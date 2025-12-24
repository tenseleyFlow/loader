"""Main CLI entry point."""

import asyncio
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


@click.command()
@click.option("--model", "-m", default="llama3.1:8b", help="Model to use")
@click.option("--backend", "-b", default="ollama", help="LLM backend (ollama)")
@click.argument("prompt", required=False)
def main(model: str, backend: str, prompt: str | None) -> None:
    """Loader - Local AI coding assistant."""
    asyncio.run(_main(model, backend, prompt))


async def _main(model: str, backend: str, prompt: str | None) -> None:
    from ..llm.ollama import OllamaBackend
    from ..agent.loop import Agent

    console.print(Panel.fit(
        "[bold blue]Loader[/bold blue] - Local AI Coding Assistant\n"
        f"Model: {model} | Backend: {backend}",
        border_style="blue",
    ))

    # Initialize backend
    llm = OllamaBackend(model=model)

    # Check health
    if not await llm.health_check():
        console.print("[red]Error: Cannot connect to Ollama. Is it running?[/red]")
        console.print("Start it with: ollama serve")
        return

    agent = Agent(backend=llm)

    if prompt:
        # Single prompt mode
        await run_once(agent, prompt)
    else:
        # Interactive mode
        await run_interactive(agent)


async def run_once(agent, prompt: str) -> None:
    """Run a single prompt."""
    def on_event(event):
        if event.type == "thinking":
            console.print("[dim]Thinking...[/dim]")
        elif event.type == "tool_call":
            console.print(f"[yellow]Tool: {event.tool_name}[/yellow]")
        elif event.type == "tool_result":
            console.print(Panel(event.content[:500], title=event.tool_name, border_style="dim"))
        elif event.type == "response":
            pass  # We'll print the full response at the end

    response = await agent.run(prompt, on_event=on_event)
    console.print(Markdown(response))


async def run_interactive(agent) -> None:
    """Run interactive chat loop."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
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
                console.print("[dim]...[/dim]", end="")
            elif event.type == "tool_call":
                console.print(f"\n[yellow]> {event.tool_name}[/yellow]", end="")
            elif event.type == "tool_result":
                # Show truncated result
                preview = event.content[:200].replace("\n", " ")
                if len(event.content) > 200:
                    preview += "..."
                console.print(f" [dim]{preview}[/dim]")

        response = await agent.run(user_input, on_event=on_event)
        console.print()
        console.print(Markdown(response))
        console.print()


if __name__ == "__main__":
    main()
