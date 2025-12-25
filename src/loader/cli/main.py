"""Main CLI entry point."""

import asyncio
import re
import click
import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

console = Console()


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


async def select_model_interactive() -> str | None:
    """Show interactive model selection menu.

    Returns:
        Selected model name, or None if cancelled/no models.
    """
    from ..llm.ollama import OllamaBackend
    from ..config import get_last_model
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter

    # Create a temporary client to list models
    backend = OllamaBackend(model="")
    models = await backend.list_models()
    await backend.close()

    if not models:
        console.print("[red]No models found. Is Ollama running?[/red]")
        console.print("Start with: [cyan]ollama serve[/cyan]")
        console.print("Pull a model: [cyan]ollama pull llama3.1:8b[/cyan]")
        return None

    # Get last used model for highlighting
    last_model = get_last_model()

    # Sort by size (largest first) for better display
    models.sort(key=lambda m: m["size"], reverse=True)

    # Display available models
    console.print("\n[bold]Available Models:[/bold]\n")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="green")
    table.add_column("Size", justify="right")
    table.add_column("", width=6)  # For "last" indicator

    model_names = []
    default_idx = 0
    for i, model in enumerate(models, 1):
        name = model["name"]
        model_names.append(name)
        size = format_size(model["size"])
        # Mark last used model
        indicator = "[yellow]← last[/yellow]" if name == last_model else ""
        if name == last_model:
            default_idx = i - 1
        table.add_row(str(i), name, size, indicator)

    console.print(table)
    console.print()

    # Create completer for model names
    completer = WordCompleter(model_names + [str(i) for i in range(1, len(models) + 1)])

    # Build prompt with default hint
    default_model = models[default_idx]["name"] if last_model else models[0]["name"]
    prompt_text = f"Select model [default: {default_model}]: "

    session = PromptSession(completer=completer)
    try:
        choice = await asyncio.to_thread(
            session.prompt,
            prompt_text,
        )
        choice = choice.strip()

        if not choice:
            return default_model  # Use last-used or first model

        # Try as number first
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]["name"]
        except ValueError:
            pass

        # Try as name (partial match)
        for name in model_names:
            if choice in name or name in choice:
                return name

        console.print(f"[yellow]Unknown model: {choice}[/yellow]")
        return None

    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Cancelled[/dim]")
        return None


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
@click.option("--model", "-m", default=None, help="Model to use (default: llama3.1:8b)")
@click.option("--select-model", "-s", is_flag=True, help="Interactively select model from available")
@click.option("--backend", "-b", default="ollama", help="LLM backend (ollama)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.option("--react", is_flag=True, help="Force ReAct mode (text-based tool calling)")
@click.option("--no-context", is_flag=True, help="Skip auto-detecting project context")
@click.option("--plan", is_flag=True, help="Enable auto-planning for complex tasks (off by default)")
@click.option("--no-recover", is_flag=True, help="Disable auto-recovery from tool errors")
@click.option("--no-tui", is_flag=True, help="Use simple Rich output instead of full TUI")
@click.option("--ctx", type=int, default=8192, help="Context window size (default: 8192, smaller = faster)")
@click.option("--gpu", type=int, default=-1, help="GPU layers (default: -1 = all, 0 = CPU only)")
@click.option("--timeout", type=int, default=None, help="Request timeout in seconds (default: auto based on model size)")
# Reasoning options
@click.option("--decompose", is_flag=True, help="Enable task decomposition (break complex tasks into subtasks)")
@click.option("--critique", is_flag=True, help="Enable self-critique (review responses before finalizing)")
@click.option("--confidence", is_flag=True, help="Enable confidence scoring (rate certainty before actions)")
@click.option("--verify", is_flag=True, help="Enable post-action verification (check results)")
@click.option("--reason", is_flag=True, help="Enable all reasoning stages (decompose + critique + confidence + verify)")
@click.argument("prompt", required=False)
def main(
    model: str | None,
    select_model: bool,
    backend: str,
    yes: bool,
    react: bool,
    no_context: bool,
    plan: bool,
    no_recover: bool,
    no_tui: bool,
    ctx: int,
    gpu: int,
    timeout: int | None,
    decompose: bool,
    critique: bool,
    confidence: bool,
    verify: bool,
    reason: bool,
    prompt: str | None,
) -> None:
    """Loader - Local AI coding assistant."""
    asyncio.run(_main(
        model, select_model, backend, yes, react, no_context, plan, no_recover, no_tui,
        ctx, gpu, timeout, decompose, critique, confidence, verify, reason, prompt
    ))


async def _main(
    model: str | None,
    select_model: bool,
    backend: str,
    yes: bool,
    react: bool,
    no_context: bool,
    plan: bool,
    no_recover: bool,
    no_tui: bool,
    ctx: int | None,
    gpu: int | None,
    timeout: float | None,
    decompose: bool,
    critique: bool,
    confidence: bool,
    verify: bool,
    reason: bool,
    prompt: str | None,
) -> None:
    from ..llm.ollama import OllamaBackend
    from ..agent.loop import Agent, AgentConfig, ReasoningConfig
    from ..tools.base import create_default_registry
    from ..config import get_default_model, set_last_model, get_last_model

    # Handle model selection
    if select_model:
        # Explicitly requested model selection
        selected = await select_model_interactive()
        if selected is None:
            return
        model = selected
    elif model is None:
        # No model specified - use saved default or fallback
        model = get_default_model()
        saved = get_last_model()
        if saved:
            console.print(f"[dim]Using saved model: {model}[/dim]")

    mode_str = "ReAct" if react else "Native"

    # Initialize backend with performance options
    llm = OllamaBackend(
        model=model,
        force_react=react,
        num_ctx=ctx,
        num_gpu=gpu,
        timeout=timeout,
    )

    # Check health
    if not await llm.health_check():
        console.print("[red]Error: Cannot connect to Ollama. Is it running?[/red]")
        console.print("Start it with: ollama serve")
        console.print(f"\nOr pull the model: [cyan]ollama pull {model}[/cyan]")
        # Offer to select a different model
        console.print("\nTry [cyan]loader --select-model[/cyan] to choose from available models.")
        return

    # Save this model as the new default
    set_last_model(model)

    # Create registry with confirmation setting
    registry = create_default_registry()
    registry.skip_confirmation = yes

    # Configure reasoning stages
    # --reason enables all, otherwise use individual flags
    reasoning_config = ReasoningConfig(
        decomposition=reason or decompose,
        self_critique=reason or critique,
        confidence_scoring=reason or confidence,
        verification=reason or verify,
    )

    config = AgentConfig(
        force_react=react,
        auto_context=not no_context,
        auto_plan=plan,  # Off by default, enable with --plan
        auto_recover=not no_recover,
        reasoning=reasoning_config,
    )
    agent = Agent(backend=llm, registry=registry, config=config)

    # Show reasoning status if enabled
    reasoning_active = []
    if reasoning_config.decomposition:
        reasoning_active.append("decompose")
    if reasoning_config.self_critique:
        reasoning_active.append("critique")
    if reasoning_config.confidence_scoring:
        reasoning_active.append("confidence")
    if reasoning_config.verification:
        reasoning_active.append("verify")
    if reasoning_active:
        console.print(f"[dim]Reasoning: {', '.join(reasoning_active)}[/dim]")

    # Single prompt mode always uses simple output
    if prompt:
        # Build status line for non-TUI mode
        timeout_mins = int(llm.timeout / 60)
        status_parts = [f"Model: {model}", f"Mode: {mode_str}"]
        if agent.project_context:
            status_parts.append(f"Project: {agent.project_context.project_type}")
        status_parts.append(f"Timeout: {timeout_mins}m")
        if yes:
            status_parts.append("Confirm: off")

        console.print(Panel.fit(
            "[bold blue]Loader[/bold blue]\n" + " | ".join(status_parts),
            border_style="blue",
        ))
        await run_once(agent, prompt, skip_confirmation=yes)
        return

    # Interactive mode - use TUI unless --no-tui
    if no_tui:
        # Build status line for non-TUI mode
        timeout_mins = int(llm.timeout / 60)
        status_parts = [f"Model: {model}", f"Mode: {mode_str}"]
        if agent.project_context:
            status_parts.append(f"Project: {agent.project_context.project_type}")
        status_parts.append(f"Timeout: {timeout_mins}m")
        if yes:
            status_parts.append("Confirm: off")

        console.print(Panel.fit(
            "[bold blue]Loader[/bold blue]\n" + " | ".join(status_parts),
            border_style="blue",
        ))
        console.print("[dim]Type 'exit' to quit, 'clear' to reset conversation[/dim]\n")
        await run_interactive(agent, skip_confirmation=yes)
    else:
        # Launch TUI
        from ..ui.app import LoaderApp

        app = LoaderApp(
            agent=agent,
            model_name=model,
            mode=mode_str,
        )
        await app.run_async()


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
    import time

    thinking_start = None
    streamed_response = False

    def on_event(event):
        nonlocal thinking_start, streamed_response
        if event.type == "thinking":
            thinking_start = time.time()
            console.print("[dim]Generating...[/dim]", end="")
        elif event.type == "stream":
            if thinking_start:
                elapsed = time.time() - thinking_start
                console.print(f" [dim]({elapsed:.1f}s)[/dim]")
                thinking_start = None
                streamed_response = True
                console.print()
            console.print(event.content, end="", markup=False)
            if event.is_stream_end:
                console.print()
        elif event.type == "plan":
            console.print(Panel(event.content, title="[bold]Plan[/bold]", border_style="blue"))
        elif event.type == "step":
            console.print(f"\n[bold yellow]{event.step_info}[/bold yellow]")
        elif event.type == "tool_call":
            if thinking_start:
                elapsed = time.time() - thinking_start
                console.print(f" [dim]({elapsed:.1f}s)[/dim]")
                thinking_start = None
            args_str = _format_tool_args(event.tool_args)
            console.print(f"[cyan]> {event.tool_name}[/cyan]({args_str})")
        elif event.type == "tool_result":
            # Show result in a compact panel
            lines = event.content.splitlines()
            preview = "\n".join(lines[:10])
            if len(lines) > 10:
                preview += f"\n[dim]... ({len(lines) - 10} more lines)[/dim]"
            console.print(Panel(preview, border_style="dim"))
        elif event.type == "recovery":
            console.print(f"[yellow]Recovering from error ({event.recovery_attempt}/3)...[/yellow]")
        elif event.type == "error":
            console.print(Panel(event.content, title="[red]Error[/red]", border_style="red"))
        elif event.type == "response":
            pass  # We'll print the full response at the end

    try:
        response = await agent.run(prompt, on_event=on_event)
        if not streamed_response:
            console.print(Markdown(clean_response(response)))
    except httpx.ReadTimeout:
        console.print("\n[red]Request timed out.[/red]")
        console.print("[dim]The model is taking too long. Try a smaller model or simpler prompt.[/dim]")
        return
    except ConfirmationRequired as e:
        console.print(f"\n[yellow]Confirmation required:[/yellow] {e.message}")
        if e.details:
            console.print(f"[dim]{e.details}[/dim]")
        if Confirm.ask("Proceed?"):
            agent.registry.skip_confirmation = True
            streamed_response = False  # Reset for continuation
            response = await agent.run("Continue with the previous action.", on_event=on_event)
            if not streamed_response:
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

        import time
        thinking_start = None
        streaming_started = False
        streamed_response = False  # Track if we displayed via streaming

        def on_event(event):
            nonlocal thinking_start, streaming_started, streamed_response
            if event.type == "thinking":
                thinking_start = time.time()
                streaming_started = False
                console.print("[dim]Generating...[/dim]", end="")
            elif event.type == "stream":
                if thinking_start and not streaming_started:
                    # First stream chunk - clear the "Generating..." and start fresh
                    elapsed = time.time() - thinking_start
                    console.print(f" [dim]({elapsed:.1f}s)[/dim]")
                    thinking_start = None
                    streaming_started = True
                    streamed_response = True
                    console.print()  # New line before streamed content
                # Print the chunk without newline
                console.print(event.content, end="", markup=False)
                if event.is_stream_end:
                    console.print()  # Final newline
            elif event.type == "plan":
                if thinking_start:
                    elapsed = time.time() - thinking_start
                    console.print(f" [dim]({elapsed:.1f}s)[/dim]")
                    thinking_start = None
                console.print(Panel(event.content, title="[bold]Plan[/bold]", border_style="blue"))
            elif event.type == "step":
                console.print(f"\n[bold yellow]{event.step_info}[/bold yellow]")
            elif event.type == "tool_call":
                if thinking_start:
                    elapsed = time.time() - thinking_start
                    console.print(f" [dim]({elapsed:.1f}s)[/dim]")
                    thinking_start = None
                if streaming_started:
                    console.print()  # New line after any streamed content
                    streaming_started = False
                args_str = _format_tool_args(event.tool_args)
                console.print(f"[cyan]> {event.tool_name}[/cyan]({args_str})")
            elif event.type == "tool_result":
                # Show compact result
                lines = event.content.splitlines()
                if len(lines) <= 3:
                    preview = event.content
                else:
                    preview = "\n".join(lines[:3]) + f"\n[dim]... ({len(lines) - 3} more lines)[/dim]"
                console.print(f"[dim]{preview}[/dim]")
            elif event.type == "recovery":
                console.print(f"\n[yellow]Recovering from error ({event.recovery_attempt}/3)...[/yellow]")
            elif event.type == "error":
                console.print(Panel(event.content, title="[red]Error[/red]", border_style="red"))

        try:
            response = await agent.run(user_input, on_event=on_event)
            console.print()
            # Only print markdown response if we didn't stream it
            if not streamed_response:
                console.print(Markdown(clean_response(response)))
            console.print()
        except httpx.ReadTimeout:
            console.print("\n[red]Request timed out.[/red]")
            console.print("[dim]The model is taking too long to respond. Try:[/dim]")
            console.print("  • A smaller model (e.g., [cyan]loader -m llama3.1:8b[/cyan])")
            console.print("  • A simpler prompt")
            console.print("  • Check if Ollama is overloaded")
            continue
        except ConfirmationRequired as e:
            console.print(f"\n[yellow]Confirmation required:[/yellow] {e.message}")
            if e.details:
                console.print(f"[dim]{e.details}[/dim]")
            if Confirm.ask("Proceed?"):
                agent.registry.skip_confirmation = True
                streamed_response = False  # Reset for continuation
                try:
                    response = await agent.run("Continue with the previous action.", on_event=on_event)
                    console.print()
                    if not streamed_response:
                        console.print(Markdown(clean_response(response)))
                    console.print()
                finally:
                    agent.registry.skip_confirmation = skip_confirmation
            else:
                console.print("[red]Aborted.[/red]\n")


if __name__ == "__main__":
    main()
