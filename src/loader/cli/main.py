"""Main CLI entry point."""

import asyncio
import re
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


def clean_response(text: str) -> str:
    """Clean up response text by removing any leaked tool call syntax."""
    # Remove tool_call tags
    text = re.sub(r"</?tool_call>", "", text)
    # Remove bare JSON tool calls that might have leaked
    text = re.sub(
        r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{[^{}]*\}[^{}]*\}',
        "",
        text,
    )
    # Clean up multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@click.command()
@click.option("--model", "-m", default="llama3.1:8b", help="Model to use")
@click.option("--backend", "-b", default="ollama", help="LLM backend (ollama)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.option("--react", is_flag=True, help="Force ReAct mode (text-based tool calling)")
@click.argument("prompt", required=False)
def main(model: str, backend: str, yes: bool, react: bool, prompt: str | None) -> None:
    """Loader - Local AI coding assistant."""
    asyncio.run(_main(model, backend, yes, react, prompt))


async def _main(model: str, backend: str, yes: bool, react: bool, prompt: str | None) -> None:
    from ..llm.ollama import OllamaBackend
    from ..agent.loop import Agent, AgentConfig
    from ..tools.base import create_default_registry

    mode_str = "ReAct" if react else "Native"
    confirm_str = "off" if yes else "on"

    console.print(Panel.fit(
        "[bold blue]Loader[/bold blue] - Local AI Coding Assistant\n"
        f"Model: {model} | Backend: {backend} | Mode: {mode_str} | Confirm: {confirm_str}",
        border_style="blue",
    ))

    # Initialize backend
    llm = OllamaBackend(model=model, force_react=react)

    # Check health
    if not await llm.health_check():
        console.print("[red]Error: Cannot connect to Ollama. Is it running?[/red]")
        console.print("Start it with: ollama serve")
        return

    # Create registry with confirmation setting
    registry = create_default_registry()
    registry.skip_confirmation = yes

    config = AgentConfig(force_react=react)
    agent = Agent(backend=llm, registry=registry, config=config)

    if prompt:
        # Single prompt mode
        await run_once(agent, prompt, skip_confirmation=yes)
    else:
        # Interactive mode
        await run_interactive(agent, skip_confirmation=yes)


def _format_tool_args(args: dict | None) -> str:
    """Format tool arguments for display."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 30:
            v = v[:27] + "..."
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


async def run_once(agent, prompt: str, skip_confirmation: bool = False) -> None:
    """Run a single prompt."""
    from ..tools.base import ConfirmationRequired

    def on_event(event):
        if event.type == "thinking":
            console.print("[dim]thinking...[/dim]")
        elif event.type == "tool_call":
            args_str = _format_tool_args(event.tool_args)
            console.print(f"[cyan]> {event.tool_name}[/cyan]({args_str})")
        elif event.type == "tool_result":
            # Show result in a compact panel
            lines = event.content.splitlines()
            preview = "\n".join(lines[:10])
            if len(lines) > 10:
                preview += f"\n[dim]... ({len(lines) - 10} more lines)[/dim]"
            console.print(Panel(preview, border_style="dim"))
        elif event.type == "response":
            pass  # We'll print the full response at the end

    try:
        response = await agent.run(prompt, on_event=on_event)
        console.print(Markdown(clean_response(response)))
    except ConfirmationRequired as e:
        console.print(f"\n[yellow]Confirmation required:[/yellow] {e.message}")
        if e.details:
            console.print(f"[dim]{e.details}[/dim]")
        if Confirm.ask("Proceed?"):
            agent.registry.skip_confirmation = True
            response = await agent.run("Continue with the previous action.", on_event=on_event)
            console.print(Markdown(clean_response(response)))
            agent.registry.skip_confirmation = skip_confirmation
        else:
            console.print("[red]Aborted.[/red]")


async def run_interactive(agent, skip_confirmation: bool = False) -> None:
    """Run interactive chat loop."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from ..tools.base import ConfirmationRequired
    import os

    history_file = os.path.expanduser("~/.loader_history")
    session = PromptSession(history=FileHistory(history_file))

    console.print("[dim]Type 'exit' to quit, 'clear' to reset conversation[/dim]\n")

    while True:
        try:
            user_input = await asyncio.to_thread(
                session.prompt,
                "You: ",
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            console.print("Goodbye!")
            break

        if user_input.lower() == "clear":
            agent.clear_history()
            console.print("[dim]Conversation cleared[/dim]")
            continue

        def on_event(event):
            if event.type == "thinking":
                console.print("[dim].[/dim]", end="")
            elif event.type == "tool_call":
                args_str = _format_tool_args(event.tool_args)
                console.print(f"\n[cyan]> {event.tool_name}[/cyan]({args_str})")
            elif event.type == "tool_result":
                # Show compact result
                lines = event.content.splitlines()
                if len(lines) <= 3:
                    preview = event.content
                else:
                    preview = "\n".join(lines[:3]) + f"\n[dim]... ({len(lines) - 3} more lines)[/dim]"
                console.print(f"[dim]{preview}[/dim]")

        try:
            response = await agent.run(user_input, on_event=on_event)
            console.print()
            console.print(Markdown(clean_response(response)))
            console.print()
        except ConfirmationRequired as e:
            console.print(f"\n[yellow]Confirmation required:[/yellow] {e.message}")
            if e.details:
                console.print(f"[dim]{e.details}[/dim]")
            if Confirm.ask("Proceed?"):
                agent.registry.skip_confirmation = True
                try:
                    response = await agent.run("Continue with the previous action.", on_event=on_event)
                    console.print()
                    console.print(Markdown(clean_response(response)))
                    console.print()
                finally:
                    agent.registry.skip_confirmation = skip_confirmation
            else:
                console.print("[red]Aborted.[/red]\n")


if __name__ == "__main__":
    main()
