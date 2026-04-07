"""Main CLI entry point."""

import asyncio
import json
import re
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ..runtime.inspection import (
    CheckStatus,
    DoctorReport,
    PermissionCheckResult,
    PermissionSnapshot,
    PromptPreview,
    StatusSnapshot,
    WorkflowTimelineSnapshot,
    collect_doctor_report,
    collect_permission_snapshot,
    collect_prompt_preview,
    collect_status_snapshot,
    collect_workflow_timeline,
    dry_run_permission_check,
    list_session_summaries,
    load_session_detail,
)
from ..runtime.permissions import PermissionMode
from .options import inject_resume_target
from .rendering import (
    format_dod_status,
    format_permission_mode,
    format_workflow_mode,
)

console = Console()
SPECIAL_COMMANDS = {
    "doctor",
    "status",
    "session",
    "explore",
    "permissions",
    "prompt",
    "workflow",
}

try:
    import httpx

    HttpReadTimeout = httpx.ReadTimeout
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    httpx = None

    class HttpReadTimeout(Exception):
        """Fallback timeout type when httpx is unavailable at import time."""


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
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter

    from ..config import get_last_model
    from ..llm.ollama import OllamaBackend

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
@click.option(
    "--permission-mode",
    type=click.Choice(
        ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"],
        case_sensitive=False,
    ),
    default="workspace-write",
    show_default=True,
    help="Runtime permission mode for tool execution",
)
@click.option("--react", is_flag=True, help="Force ReAct mode (text-based tool calling)")
@click.option("--no-context", is_flag=True, help="Skip auto-detecting project context")
@click.option("--plan", is_flag=True, help="Start the task in plan mode")
@click.option("--clarify", is_flag=True, help="Start the task in clarify mode")
@click.option("--resume-target", hidden=True, default=None)
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
def cli(
    model: str | None,
    select_model: bool,
    backend: str,
    yes: bool,
    permission_mode: str,
    react: bool,
    no_context: bool,
    plan: bool,
    clarify: bool,
    resume_target: str | None,
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
        model, select_model, backend, yes, permission_mode, react, no_context, plan, clarify, resume_target, no_recover,
        no_tui, ctx, gpu, timeout, decompose, critique, confidence, verify, reason, prompt
    ))


def main() -> None:
    """Entry-point wrapper that supports `--resume [session-id]` syntax."""

    argv = inject_resume_target(sys.argv[1:])
    if argv and argv[0] in {"-h", "--help"}:
        click.echo(_loader_help_text())
        return

    if argv and argv[0] == "session" and len(argv) >= 3 and argv[1] == "resume":
        cli.main(
            args=["--resume-target", argv[2], *argv[3:]],
            prog_name="loader",
        )
        return

    if argv and argv[0] in SPECIAL_COMMANDS:
        _run_special_command(argv)
        return

    cli.main(args=argv, prog_name="loader")


async def _main(
    model: str | None,
    select_model: bool,
    backend: str,
    yes: bool,
    permission_mode: str,
    react: bool,
    no_context: bool,
    plan: bool,
    clarify: bool,
    resume_target: str | None,
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
    from ..agent.loop import Agent, AgentConfig, ReasoningConfig
    from ..config import get_default_model, get_last_model, set_last_model
    from ..llm.ollama import OllamaBackend
    from ..tools.base import create_default_registry

    if plan and clarify:
        raise click.UsageError("Choose only one of --plan or --clarify.")

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

    await llm.describe_model()

    # Determine actual mode based on resolved model capabilities (not just CLI flag)
    mode_str = "ReAct" if react or not llm.supports_native_tools() else "Native"

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
        auto_plan=False,
        auto_recover=not no_recover,
        permission_mode=PermissionMode.from_str(permission_mode),
        workflow_mode_override="clarify" if clarify else ("plan" if plan else None),
        reasoning=reasoning_config,
    )
    try:
        agent = Agent(backend=llm, registry=registry, config=config)
    except ValueError as exc:
        console.print(f"[red]Permission policy error:[/red] {exc}")
        return
    resumed = False
    if resume_target is not None:
        session_id = None if resume_target == "__latest__" else resume_target
        resumed = agent.resume_session(session_id)
        if not resumed and session_id is None:
            console.print("[yellow]No previous session found; starting a new session.[/yellow]")
        elif not resumed:
            console.print(f"[red]Session not found:[/red] {session_id}")
            return
        else:
            console.print(f"[dim]Resumed session: {agent.session.session_id}[/dim]")

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
        status_parts = [
            f"Model: {model}",
            f"Mode: {mode_str}",
            (
                "Capabilities: "
                f"{agent.capability_profile.preferred_tool_call_format}/"
                f"{agent.capability_profile.verification_strictness}"
            ),
            f"Workflow: {format_workflow_mode(agent.workflow_mode)}",
            f"Permissions: {format_permission_mode(agent.active_permission_mode)}",
            f"Session: {agent.session.session_id}",
        ]
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
        status_parts = [
            f"Model: {model}",
            f"Mode: {mode_str}",
            (
                "Capabilities: "
                f"{agent.capability_profile.preferred_tool_call_format}/"
                f"{agent.capability_profile.verification_strictness}"
            ),
            f"Workflow: {format_workflow_mode(agent.workflow_mode)}",
            f"Permissions: {format_permission_mode(agent.active_permission_mode)}",
            f"Session: {agent.session.session_id}",
        ]
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
            capability_profile=(
                f"{agent.capability_profile.preferred_tool_call_format}/"
                f"{agent.capability_profile.verification_strictness}"
            ),
            session_id=agent.session.session_id,
            workflow_mode=agent.workflow_mode,
            turn_phase=agent.session.active_turn_phase or "",
            permission_mode=agent.active_permission_mode,
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
    import time

    from ..tools.base import ConfirmationRequired

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
        elif event.type == "workflow_mode":
            console.print(
                f"[dim]Workflow: {format_workflow_mode(event.workflow_mode or 'execute')}[/dim]"
            )
        elif event.type == "artifact":
            console.print(
                Panel(
                    f"{event.content}\n[dim]{event.artifact_path}[/dim]",
                    title=f"[bold cyan]{event.artifact_kind or 'artifact'}[/bold cyan]",
                    border_style="cyan",
                )
            )
        elif event.type == "step":
            console.print(f"\n[bold yellow]{event.step_info}[/bold yellow]")
        elif event.type == "tool_call":
            if thinking_start:
                elapsed = time.time() - thinking_start
                console.print(f" [dim]({elapsed:.1f}s)[/dim]")
                thinking_start = None
            args_str = _format_tool_args(event.tool_args)
            tool_label = (
                f"verify {event.tool_name}"
                if event.phase == "verification"
                else event.tool_name
            )
            console.print(f"[cyan]> {tool_label}[/cyan]({args_str})")
        elif event.type == "tool_result":
            # Show result in a compact panel
            lines = event.content.splitlines()
            preview = "\n".join(lines[:10])
            if len(lines) > 10:
                preview += f"\n[dim]... ({len(lines) - 10} more lines)[/dim]"
            border_style = "magenta" if event.phase == "verification" else "dim"
            console.print(Panel(preview, border_style=border_style))
        elif event.type == "dod_status":
            console.print(f"[dim]{format_dod_status(event)}[/dim]")
        elif event.type == "recovery":
            console.print(f"[yellow]Recovering from error ({event.recovery_attempt}/3)...[/yellow]")
        elif event.type == "error":
            console.print(Panel(event.content, title="[red]Error[/red]", border_style="red"))
        elif event.type == "response":
            pass  # We'll print the full response at the end

    try:
        response = await agent.run(
            prompt,
            on_event=on_event,
            on_user_question=_ask_user_question_cli,
        )
        if not streamed_response:
            console.print(Markdown(clean_response(response)))
    except HttpReadTimeout:
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
            response = await agent.run(
                "Continue with the previous action.",
                on_event=on_event,
                on_user_question=_ask_user_question_cli,
            )
            if not streamed_response:
                console.print(Markdown(clean_response(response)))
            agent.registry.skip_confirmation = skip_confirmation
        else:
            console.print("[red]Aborted.[/red]")


async def run_interactive(agent, skip_confirmation: bool = False) -> None:
    """Run interactive chat loop."""
    import os

    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    from ..tools.base import ConfirmationRequired

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
            elif event.type == "workflow_mode":
                console.print(
                    f"\n[dim]Workflow: {format_workflow_mode(event.workflow_mode or 'execute')}[/dim]"
                )
            elif event.type == "artifact":
                console.print(
                    Panel(
                        f"{event.content}\n[dim]{event.artifact_path}[/dim]",
                        title=f"[bold cyan]{event.artifact_kind or 'artifact'}[/bold cyan]",
                        border_style="cyan",
                    )
                )
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
                tool_label = (
                    f"verify {event.tool_name}"
                    if event.phase == "verification"
                    else event.tool_name
                )
                console.print(f"[cyan]> {tool_label}[/cyan]({args_str})")
            elif event.type == "tool_result":
                # Show compact result
                lines = event.content.splitlines()
                if len(lines) <= 3:
                    preview = event.content
                else:
                    preview = "\n".join(lines[:3]) + f"\n[dim]... ({len(lines) - 3} more lines)[/dim]"
                style = "magenta" if event.phase == "verification" else "dim"
                console.print(f"[{style}]{preview}[/{style}]")
            elif event.type == "dod_status":
                console.print(f"\n[dim]{format_dod_status(event)}[/dim]")
            elif event.type == "recovery":
                console.print(f"\n[yellow]Recovering from error ({event.recovery_attempt}/3)...[/yellow]")
            elif event.type == "error":
                console.print(Panel(event.content, title="[red]Error[/red]", border_style="red"))

        try:
            response = await agent.run(
                user_input,
                on_event=on_event,
                on_user_question=_ask_user_question_cli,
            )
            console.print()
            # Only print markdown response if we didn't stream it
            if not streamed_response:
                console.print(Markdown(clean_response(response)))
            console.print()
        except HttpReadTimeout:
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
                    response = await agent.run(
                        "Continue with the previous action.",
                        on_event=on_event,
                        on_user_question=_ask_user_question_cli,
                    )
                    console.print()
                    if not streamed_response:
                        console.print(Markdown(clean_response(response)))
                    console.print()
                finally:
                    agent.registry.skip_confirmation = skip_confirmation
            else:
                console.print("[red]Aborted.[/red]\n")


async def _ask_user_question_cli(
    question: str,
    options: list[str] | None,
) -> str:
    """Prompt the CLI user for an AskUserQuestion response."""

    console.print()
    console.print(
        Panel(question, title="[bold cyan]Question[/bold cyan]", border_style="cyan")
    )
    if options:
        for index, option in enumerate(options, start=1):
            console.print(f"  [cyan]{index}.[/cyan] {option}")
        answer = await asyncio.to_thread(Prompt.ask, "Answer", default="1")
        answer = answer.strip()
        if answer.isdigit():
            selected = int(answer) - 1
            if 0 <= selected < len(options):
                return options[selected]
        return answer

    return await asyncio.to_thread(Prompt.ask, "Answer")


@click.command(name="doctor")
@click.option("--model", "-m", default=None, help="Model to inspect (default: saved model)")
@click.option("--backend", "-b", default="ollama", help="Backend to inspect")
@click.option(
    "--permission-mode",
    type=click.Choice(
        ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"],
        case_sensitive=False,
    ),
    default="workspace-write",
    show_default=True,
    help="Permission mode to audit against tool requirements",
)
def doctor_cli(
    model: str | None,
    backend: str,
    permission_mode: str,
) -> None:
    """Inspect Loader runtime health without entering the agent loop."""

    asyncio.run(_doctor_main(model=model, backend=backend, permission_mode=permission_mode))


@click.command(name="status")
@click.option("--model", "-m", default=None, help="Model to summarize (default: saved model)")
@click.option(
    "--permission-mode",
    type=click.Choice(
        ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"],
        case_sensitive=False,
    ),
    default="workspace-write",
    show_default=True,
    help="Fallback permission mode when no session exists",
)
def status_cli(
    model: str | None,
    permission_mode: str,
) -> None:
    """Show the current Loader status from persisted state."""

    _status_main(model=model, permission_mode=permission_mode)


@click.command(name="explore")
@click.option("--model", "-m", default=None, help="Model to use for explore mode")
@click.option("--select-model", "-s", is_flag=True, help="Interactively select model")
@click.option("--backend", "-b", default="ollama", help="LLM backend (ollama)")
@click.option("--react", is_flag=True, help="Force ReAct mode")
@click.option("--no-context", is_flag=True, help="Skip auto-detecting project context")
@click.option("--ctx", type=int, default=8192, help="Context window size")
@click.option("--gpu", type=int, default=-1, help="GPU layers (-1 = all, 0 = CPU only)")
@click.option("--timeout", type=int, default=None, help="Request timeout in seconds")
@click.argument("prompt")
def explore_cli(
    model: str | None,
    select_model: bool,
    backend: str,
    react: bool,
    no_context: bool,
    ctx: int,
    gpu: int,
    timeout: int | None,
    prompt: str,
) -> None:
    """Run a read-only lookup query through the explore lane."""

    asyncio.run(
        _explore_main(
            model=model,
            select_model=select_model,
            backend=backend,
            react=react,
            no_context=no_context,
            ctx=ctx,
            gpu=gpu,
            timeout=timeout,
            prompt=prompt,
        )
    )


@click.group(name="session")
def session_cli() -> None:
    """Inspect persisted Loader sessions."""


@click.group(name="permissions")
def permissions_cli() -> None:
    """Inspect and dry-run Loader permission policy."""


@click.group(name="prompt")
def prompt_cli() -> None:
    """Inspect Loader prompt construction without a live turn."""


@click.group(name="workflow")
def workflow_cli() -> None:
    """Inspect persisted Loader workflow history."""


@session_cli.command("list")
def session_list_cli() -> None:
    """List persisted sessions."""

    _session_list_main()


@session_cli.command("show")
@click.argument("session_id")
def session_show_cli(session_id: str) -> None:
    """Show one persisted session in detail."""

    _session_show_main(session_id)


@permissions_cli.command("show")
@click.option(
    "--permission-mode",
    type=click.Choice(
        ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"],
        case_sensitive=False,
    ),
    default="workspace-write",
    show_default=True,
    help="Permission mode to inspect",
)
def permissions_show_cli(permission_mode: str) -> None:
    """Show the active permission rules and normalized policy state."""

    _permissions_show_main(permission_mode=permission_mode)


@permissions_cli.command("check")
@click.option(
    "--permission-mode",
    type=click.Choice(
        ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"],
        case_sensitive=False,
    ),
    default="workspace-write",
    show_default=True,
    help="Permission mode to dry-run against",
)
@click.option(
    "--args",
    "args_json",
    default=None,
    help="JSON object of tool arguments to dry-run",
)
@click.argument("tool_name")
@click.argument("input_value", required=False)
def permissions_check_cli(
    permission_mode: str,
    args_json: str | None,
    tool_name: str,
    input_value: str | None,
) -> None:
    """Dry-run one hypothetical tool request against the active permission policy."""

    _permissions_check_main(
        tool_name=tool_name,
        input_value=input_value,
        args_json=args_json,
        permission_mode=permission_mode,
    )


@prompt_cli.command("show")
@click.option("--model", "-m", default=None, help="Model to preview (default: saved model)")
@click.option(
    "--workflow-mode",
    type=click.Choice(["clarify", "plan", "execute", "verify"], case_sensitive=False),
    default=None,
    help="Override the workflow mode used for the preview",
)
@click.option(
    "--permission-mode",
    type=click.Choice(
        ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"],
        case_sensitive=False,
    ),
    default=None,
    help="Override the permission mode used for the preview",
)
@click.option("--react", is_flag=True, help="Force ReAct formatting for the preview")
@click.argument("current_task", required=False)
def prompt_show_cli(
    model: str | None,
    workflow_mode: str | None,
    permission_mode: str | None,
    react: bool,
    current_task: str | None,
) -> None:
    """Render the current prompt contract without sending a model request."""

    _prompt_show_main(
        model=model,
        workflow_mode=workflow_mode,
        permission_mode=permission_mode,
        react=react,
        current_task=current_task,
    )


@workflow_cli.command("show")
@click.option(
    "--mode",
    type=click.Choice(["clarify", "plan", "execute", "verify"], case_sensitive=False),
    default=None,
    help="Filter timeline entries to one workflow mode",
)
@click.option(
    "--kind",
    type=click.Choice(
        [
            "route",
            "handoff",
            "reentry",
            "clarify_continue",
            "clarify_exit",
            "plan_refresh",
            "verify_skip",
        ],
        case_sensitive=False,
    ),
    default=None,
    help="Filter timeline entries to one event kind",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Show only the most recent matching entries",
)
@click.argument("session_id", required=False)
def workflow_show_cli(
    mode: str | None,
    kind: str | None,
    limit: int,
    session_id: str | None,
) -> None:
    """Show the persisted workflow timeline for the latest or named session."""

    _workflow_show_main(
        session_id=session_id,
        mode=mode,
        kind=kind,
        limit=limit,
    )


def _run_special_command(argv: list[str]) -> None:
    command = argv[0]
    if command == "doctor":
        doctor_cli.main(args=argv[1:], prog_name="loader doctor")
        return
    if command == "status":
        status_cli.main(args=argv[1:], prog_name="loader status")
        return
    if command == "explore":
        explore_cli.main(args=argv[1:], prog_name="loader explore")
        return
    if command == "permissions":
        if len(argv) == 1:
            click.echo(_permissions_help_text())
            return
        permissions_cli.main(args=argv[1:], prog_name="loader permissions")
        return
    if command == "prompt":
        if len(argv) == 1:
            click.echo(_prompt_help_text())
            return
        prompt_cli.main(args=argv[1:], prog_name="loader prompt")
        return
    if command == "workflow":
        if len(argv) == 1:
            click.echo(_workflow_help_text())
            return
        workflow_cli.main(args=argv[1:], prog_name="loader workflow")
        return
    if command == "session":
        if len(argv) == 1:
            click.echo(_session_help_text())
            return
        session_cli.main(args=argv[1:], prog_name="loader session")
        return


def _loader_help_text() -> str:
    ctx = click.Context(cli, info_name="loader")
    base_help = cli.get_help(ctx)
    extra = "\n".join(
        [
            "",
            "Additional Commands:",
            "  loader doctor              Inspect backend, workspace, and state health",
            "  loader status              Show persisted runtime status",
            "  loader explore <prompt>    Run a fast read-only lookup query",
            "  loader prompt show         Preview the current prompt contract",
            "  loader permissions show    Display normalized permission rules",
            "  loader permissions check   Dry-run one permission decision",
            "  loader workflow show       Show the persisted workflow timeline",
            "  loader session list        List persisted sessions",
            "  loader session show <id>   Show one persisted session",
            "  loader session resume <id> Resume a persisted session through the main runtime",
        ]
    )
    return base_help + extra


def _permissions_help_text() -> str:
    return "\n".join(
        [
            "Usage: loader permissions [COMMAND]",
            "",
            "Commands:",
            "  show                 Display normalized permission rules and source metadata",
            "  check <tool> [input] Dry-run one permission decision for a tool request",
        ]
    )


def _prompt_help_text() -> str:
    return "\n".join(
        [
            "Usage: loader prompt [COMMAND]",
            "",
            "Commands:",
            "  show [task]  Render the current prompt contract without a model call",
        ]
    )


def _workflow_help_text() -> str:
    return "\n".join(
        [
            "Usage: loader workflow [COMMAND]",
            "",
            "Commands:",
            "  show [id]  Show the persisted workflow timeline with optional filters",
        ]
    )


def _session_help_text() -> str:
    return "\n".join(
        [
            "Usage: loader session [COMMAND]",
            "",
            "Commands:",
            "  list          List persisted sessions",
            "  show <id>     Show one persisted session",
            "  resume <id>   Resume a persisted session through the main runtime",
        ]
    )


def _status_color(status: CheckStatus) -> str:
    return {
        CheckStatus.PASS: "green",
        CheckStatus.WARN: "yellow",
        CheckStatus.FAIL: "red",
    }[status]


def _render_check_status(status: CheckStatus) -> str:
    color = _status_color(status)
    return f"[{color}]{status.value}[/{color}]"


async def _doctor_main(
    *,
    model: str | None,
    backend: str,
    permission_mode: str,
) -> None:
    report = await collect_doctor_report(
        model=model,
        backend=backend,
        permission_mode=permission_mode,
    )
    _print_doctor_report(report)


async def _explore_main(
    *,
    model: str | None,
    select_model: bool,
    backend: str,
    react: bool,
    no_context: bool,
    ctx: int,
    gpu: int,
    timeout: int | None,
    prompt: str,
) -> None:
    from ..agent.loop import Agent, AgentConfig
    from ..config import get_default_model, get_last_model, set_last_model
    from ..llm.ollama import OllamaBackend
    from ..runtime.permissions import PermissionMode

    if select_model:
        selected = await select_model_interactive()
        if selected is None:
            return
        model = selected
    elif model is None:
        model = get_default_model()
        if get_last_model():
            console.print(f"[dim]Using saved model: {model}[/dim]")

    llm = OllamaBackend(
        model=model,
        force_react=react,
        num_ctx=ctx,
        num_gpu=gpu,
        timeout=timeout,
    )
    if not await llm.health_check():
        console.print("[red]Error: Cannot connect to Ollama. Is it running?[/red]")
        console.print("Start it with: ollama serve")
        console.print(f"\nOr pull the model: [cyan]ollama pull {model}[/cyan]")
        return

    await llm.describe_model()
    set_last_model(model)

    try:
        agent = Agent(
            backend=llm,
            config=AgentConfig(
                auto_context=not no_context,
                force_react=react,
                permission_mode=PermissionMode.READ_ONLY,
                stream=False,
            ),
        )
    except ValueError as exc:
        console.print(f"[red]Permission policy error:[/red] {exc}")
        return
    mode_str = "ReAct" if agent.use_react else "Native"
    console.print(
        Panel.fit(
            "[bold blue]Loader Explore[/bold blue]\n"
            + " | ".join(
                [
                    f"Model: {model}",
                    f"Mode: {mode_str}",
                    "Lane: explore",
                    "Permissions: read-only",
                ]
            ),
            border_style="blue",
        )
    )

    def on_event(event) -> None:
        if event.type == "tool_call":
            args_str = _format_tool_args(event.tool_args)
            console.print(f"[cyan]> {event.tool_name}[/cyan]({args_str})")
        elif event.type == "tool_result":
            lines = event.content.splitlines()
            preview = "\n".join(lines[:8])
            if len(lines) > 8:
                preview += f"\n[dim]... ({len(lines) - 8} more lines)[/dim]"
            console.print(Panel(preview, border_style="dim"))

    response = await agent.run_explore(prompt, on_event=on_event)
    console.print(Markdown(clean_response(response)))


def _print_doctor_report(report: DoctorReport) -> None:
    overall_color = _status_color(report.overall_status)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Model:[/bold] {report.model}",
                    f"[bold]Workspace:[/bold] {report.project_root}",
                    f"[bold]Capabilities:[/bold] {report.capability_profile.model_name} / {report.capability_profile.preferred_tool_call_format}",
                    f"[bold]Permission Mode:[/bold] {report.permission_mode}",
                    (
                        "[bold]Permission Prompting:[/bold] "
                        + ("enabled" if report.permission_prompting_enabled else "disabled")
                    ),
                    (
                        "[bold]Permission Rules:[/bold] "
                        f"{report.permission_rule_counts['allow']} allow / "
                        f"{report.permission_rule_counts['deny']} deny / "
                        f"{report.permission_rule_counts['ask']} ask"
                    ),
                    f"[bold]Rules Source:[/bold] {report.permission_rules_source}",
                    f"[bold]Overall:[/bold] [{overall_color}]{report.overall_status.value}[/{overall_color}]",
                ]
            ),
            title="[bold blue]Loader Doctor[/bold blue]",
            border_style=overall_color,
        )
    )

    checks = Table(show_header=True, header_style="bold cyan")
    checks.add_column("Check", style="white")
    checks.add_column("Status", width=8)
    checks.add_column("Message", style="white")
    checks.add_column("Remediation", style="dim")
    for check in report.checks:
        checks.add_row(
            check.name,
            _render_check_status(check.status),
            check.message,
            check.remediation,
        )
    console.print(checks)

    permissions = Table(show_header=True, header_style="bold cyan")
    permissions.add_column("Tool", style="white")
    permissions.add_column("Required", style="white")
    permissions.add_column("Resolution", style="white")
    for item in report.tool_permissions:
        resolution = {
            "allow": "[green]allow[/green]",
            "deny": "[red]deny[/red]",
            "ask": "[magenta]ask[/magenta]",
        }.get(item.resolution, item.resolution)
        permissions.add_row(item.tool_name, item.required_mode, resolution)
    console.print()
    console.print(permissions)


def _status_main(
    *,
    model: str | None,
    permission_mode: str,
) -> None:
    snapshot = collect_status_snapshot(model=model, permission_mode=permission_mode)
    _print_status_snapshot(snapshot)


def _print_status_snapshot(snapshot: StatusSnapshot) -> None:
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Workspace", str(snapshot.project_root))
    table.add_row("Project", snapshot.project_type)
    table.add_row("Model", snapshot.model)
    table.add_row("Capabilities", f"{snapshot.capability_profile.preferred_tool_call_format} / {snapshot.capability_profile.verification_strictness}")
    table.add_row("Session", snapshot.active_session_id or "none")
    table.add_row("Workflow", snapshot.workflow_mode)
    if snapshot.workflow_decision_kind:
        table.add_row("Decision Kind", snapshot.workflow_decision_kind)
    if snapshot.workflow_reason_summary or snapshot.workflow_reason_code:
        table.add_row(
            "Workflow Reason",
            _format_workflow_reason(
                summary=snapshot.workflow_reason_summary,
                code=snapshot.workflow_reason_code,
            ),
        )
    if snapshot.workflow_scheduled_next_mode:
        table.add_row("Scheduled Next", snapshot.workflow_scheduled_next_mode)
    table.add_row("Phase", snapshot.active_turn_phase or "idle")
    if snapshot.last_turn_transition_summary:
        table.add_row("Last Transition", snapshot.last_turn_transition_summary)
    table.add_row("Permission Mode", snapshot.permission_mode)
    table.add_row("Prompt Format", snapshot.prompt_format or "unknown")
    table.add_row(
        "Prompt Sections",
        ", ".join(snapshot.prompt_sections) if snapshot.prompt_sections else "none",
    )
    table.add_row(
        "Permission Rules",
        (
            f"{snapshot.permission_rule_counts['allow']} allow / "
            f"{snapshot.permission_rule_counts['deny']} deny / "
            f"{snapshot.permission_rule_counts['ask']} ask"
        ),
    )
    table.add_row(
        "Permission Prompting",
        "enabled" if snapshot.permission_prompting_enabled else "disabled",
    )
    table.add_row(
        "Rules Status",
        "valid" if snapshot.permission_rules_valid else "invalid",
    )
    table.add_row("Rules Source", snapshot.permission_rules_source)
    table.add_row("Task", snapshot.current_task or "none")
    table.add_row("Messages", str(snapshot.message_count))
    table.add_row("DoD", snapshot.dod_status or "none")
    table.add_row("Pending", str(snapshot.dod_pending_items_count))
    table.add_row("Last Verify", snapshot.last_verification_result or "none")
    if snapshot.usage:
        table.add_row(
            "Usage",
            ", ".join(f"{key}={value}" for key, value in sorted(snapshot.usage.items())),
        )
    table.add_row("Compactions", str(snapshot.compaction_count))

    console.print(
        Panel.fit(
            table,
            title="[bold blue]Loader Status[/bold blue]",
            border_style="blue",
        )
    )

    if snapshot.recent_verification:
        evidence = Table(show_header=True, header_style="bold cyan")
        evidence.add_column("Result", width=8)
        evidence.add_column("Kind", width=10)
        evidence.add_column("Command", style="white")
        evidence.add_column("Detail", style="dim")
        for item in snapshot.recent_verification:
            result = "[green]pass[/green]" if item.passed else "[red]fail[/red]"
            evidence.add_row(result, item.kind, item.command, item.detail or "-")
        console.print(evidence)


def _session_list_main() -> None:
    entries = list_session_summaries()
    if not entries:
        console.print("[yellow]No persisted sessions found.[/yellow]")
        return

    for index, entry in enumerate(entries):
        policy_summary = (
            f"{entry.permission_rule_counts['allow']} allow / "
            f"{entry.permission_rule_counts['deny']} deny / "
            f"{entry.permission_rule_counts['ask']} ask"
        )
        if entry.permission_prompting_enabled:
            policy_summary = f"{policy_summary} (prompting enabled)"
        else:
            policy_summary = f"{policy_summary} (prompting disabled)"

        table = Table(show_header=False, box=None)
        table.add_column("Field", style="bold cyan")
        table.add_column("Value", style="white")
        table.add_row("Current", "yes" if entry.is_current else "no")
        table.add_row("Created", entry.created_at)
        table.add_row("Updated", entry.updated_at)
        table.add_row("Messages", str(entry.message_count))
        table.add_row("Workflow", entry.workflow_mode)
        if entry.workflow_decision_kind:
            table.add_row("Decision Kind", entry.workflow_decision_kind)
        if entry.workflow_reason_summary or entry.workflow_reason_code:
            table.add_row(
                "Workflow Reason",
                _format_workflow_reason(
                    summary=entry.workflow_reason_summary,
                    code=entry.workflow_reason_code,
                ),
            )
        table.add_row("Phase", entry.active_turn_phase or "idle")
        if entry.last_turn_transition_summary:
            table.add_row("Last Transition", entry.last_turn_transition_summary)
        table.add_row("Permission Mode", entry.permission_mode)
        table.add_row("Prompt", entry.prompt_format or "unknown")
        table.add_row("Permission Rules", policy_summary)
        table.add_row("Rules Source", entry.permission_rules_source or "none")
        table.add_row("DoD", entry.dod_status or "none")
        table.add_row("Task", entry.current_task or "none")
        console.print(
            Panel.fit(
                table,
                title=f"[bold blue]{entry.session_id}[/bold blue]",
                border_style="blue",
            )
        )
        if index < len(entries) - 1:
            console.print()


def _session_show_main(session_id: str) -> None:
    try:
        detail = load_session_detail(session_id)
    except FileNotFoundError:
        console.print(f"[red]Session not found:[/red] {session_id}")
        raise SystemExit(1) from None

    snapshot = detail.snapshot
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Session", snapshot.session_id)
    table.add_row("Current", "yes" if detail.is_current else "no")
    table.add_row("Created", snapshot.created_at)
    table.add_row("Updated", snapshot.updated_at)
    table.add_row("Messages", str(len(snapshot.messages)))
    table.add_row("Workflow", snapshot.workflow_mode)
    if snapshot.workflow_decision_kind:
        table.add_row("Decision Kind", snapshot.workflow_decision_kind)
    if snapshot.workflow_reason_summary or snapshot.workflow_reason_code:
        table.add_row(
            "Workflow Reason",
            _format_workflow_reason(
                summary=snapshot.workflow_reason_summary,
                code=snapshot.workflow_reason_code,
            ),
        )
    table.add_row("Phase", snapshot.active_turn_phase or "idle")
    if snapshot.last_turn_transition_summary:
        table.add_row("Last Transition", snapshot.last_turn_transition_summary)
    table.add_row("Permission Mode", snapshot.permission_mode)
    table.add_row("Prompt Format", snapshot.prompt_format or "unknown")
    table.add_row(
        "Prompt Sections",
        ", ".join(snapshot.prompt_sections) if snapshot.prompt_sections else "none",
    )
    table.add_row(
        "Permission Prompting",
        "enabled" if snapshot.permission_prompting_enabled else "disabled",
    )
    table.add_row(
        "Permission Rules",
        (
            f"{snapshot.permission_rule_counts['allow']} allow / "
            f"{snapshot.permission_rule_counts['deny']} deny / "
            f"{snapshot.permission_rule_counts['ask']} ask"
        ),
    )
    table.add_row("Rules Source", snapshot.permission_rules_source or "none")
    table.add_row("Task", snapshot.current_task or "none")
    table.add_row("Active DoD", snapshot.active_dod_path or "none")
    if snapshot.usage:
        table.add_row(
            "Usage",
            ", ".join(f"{key}={value}" for key, value in sorted(snapshot.usage.items())),
        )
    if snapshot.compaction is not None:
        table.add_row("Compactions", str(snapshot.compaction.count))

    console.print(
        Panel.fit(
            table,
            title="[bold blue]Loader Session[/bold blue]",
            border_style="blue",
        )
    )

    if detail.definition_of_done is not None:
        dod_table = Table(show_header=False, box=None)
        dod_table.add_column("Field", style="bold magenta")
        dod_table.add_column("Value", style="white")
        dod_table.add_row("Status", detail.definition_of_done.status)
        dod_table.add_row("Pending", ", ".join(detail.definition_of_done.pending_items) or "none")
        dod_table.add_row("Completed", ", ".join(detail.definition_of_done.completed_items) or "none")
        dod_table.add_row(
            "Last Verify",
            detail.definition_of_done.last_verification_result or "none",
        )
        console.print(dod_table)

    if snapshot.workflow_timeline:
        console.print()
        _print_workflow_timeline_entries(
            snapshot.workflow_timeline,
            title="[bold blue]Workflow Timeline[/bold blue]",
            limit=5,
        )


def _workflow_show_main(
    *,
    session_id: str | None,
    mode: str | None,
    kind: str | None,
    limit: int | None,
) -> None:
    try:
        snapshot: WorkflowTimelineSnapshot = collect_workflow_timeline(
            session_id=session_id,
            mode=mode,
            kind=kind,
            limit=limit,
        )
    except FileNotFoundError:
        console.print(f"[red]Session not found:[/red] {session_id}")
        raise SystemExit(1) from None

    if snapshot.session_id is None:
        console.print("[yellow]No persisted workflow timeline found.[/yellow]")
        return

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Workspace", str(snapshot.project_root))
    table.add_row("Session", snapshot.session_id or "none")
    table.add_row("Current", "yes" if snapshot.is_current else "no")
    table.add_row("Workflow", snapshot.workflow_mode)
    table.add_row("Task", snapshot.current_task or "none")
    table.add_row("Entries", f"{len(snapshot.entries)} shown / {snapshot.total_entries} total")
    table.add_row(
        "Filters",
        _format_workflow_filters(
            mode=snapshot.selected_mode,
            kind=snapshot.selected_kind,
            limit=snapshot.entry_limit,
        ),
    )
    console.print(
        Panel.fit(
            table,
            title="[bold blue]Loader Workflow[/bold blue]",
            border_style="blue",
        )
    )
    if snapshot.highlights:
        console.print()
        _print_workflow_highlights(
            snapshot.highlights,
            title="[bold blue]Workflow Answers[/bold blue]",
        )
    console.print()
    _print_workflow_timeline_entries(
        snapshot.entries,
        title="[bold blue]Workflow Timeline[/bold blue]",
    )


def _permissions_show_main(*, permission_mode: str) -> None:
    snapshot = collect_permission_snapshot(permission_mode=permission_mode)
    _print_permission_snapshot(snapshot)


def _prompt_show_main(
    *,
    model: str | None,
    workflow_mode: str | None,
    permission_mode: str | None,
    react: bool,
    current_task: str | None,
) -> None:
    preview = collect_prompt_preview(
        model=model,
        workflow_mode=workflow_mode,
        permission_mode=permission_mode,
        current_task=current_task,
        force_react=react,
    )
    _print_prompt_preview(preview)


def _permissions_check_main(
    *,
    tool_name: str,
    input_value: str | None,
    args_json: str | None,
    permission_mode: str,
) -> None:
    from ..tools.base import create_default_registry

    registry = create_default_registry()
    registry.configure_workspace_root(".")
    tool = registry.get(tool_name)
    if tool is None:
        available = ", ".join(sorted(item.name for item in registry.list_tools()))
        raise click.ClickException(
            f"Unknown tool `{tool_name}`. Available tools: {available}"
        )

    arguments = _coerce_permission_check_arguments(
        tool,
        input_value=input_value,
        args_json=args_json,
    )
    try:
        result = dry_run_permission_check(
            tool_name,
            arguments,
            permission_mode=permission_mode,
            registry=registry,
        )
    except ValueError as exc:
        raise click.ClickException(
            f"{exc}. Run `loader permissions show` after repairing the rule file."
        ) from exc
    _print_permission_check_result(result)


def _print_permission_snapshot(snapshot: PermissionSnapshot) -> None:
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Workspace", str(snapshot.project_root))
    table.add_row("Permission Mode", snapshot.active_mode)
    table.add_row(
        "Permission Prompting",
        "enabled" if snapshot.prompting_enabled else "disabled",
    )
    table.add_row(
        "Permission Rules",
        (
            f"{snapshot.rule_counts['allow']} allow / "
            f"{snapshot.rule_counts['deny']} deny / "
            f"{snapshot.rule_counts['ask']} ask"
        ),
    )
    table.add_row("Rules Status", "valid" if snapshot.rules_valid else "invalid")
    table.add_row("Rules Source", snapshot.rules_source)

    border_style = "blue" if snapshot.rules_valid else "red"
    console.print(
        Panel.fit(
            table,
            title="[bold blue]Loader Permissions[/bold blue]",
            border_style=border_style,
        )
    )

    if snapshot.rules_error:
        console.print(
            Panel(
                snapshot.rules_error,
                title="[bold red]Rule Error[/bold red]",
                border_style="red",
            )
        )

    rules_table = Table(show_header=True, header_style="bold cyan")
    rules_table.add_column("Disposition", style="white")
    rules_table.add_column("Tool", style="white")
    rules_table.add_column("Contains", style="white")
    rules_table.add_column("Path Contains", style="white")

    has_rules = False
    for disposition in ("allow", "deny", "ask"):
        color = {
            "allow": "green",
            "deny": "red",
            "ask": "magenta",
        }[disposition]
        for item in snapshot.normalized_rules.get(disposition, []):
            has_rules = True
            rules_table.add_row(
                f"[{color}]{disposition}[/{color}]",
                item.tool_name or "*",
                item.contains or "-",
                item.path_contains or "-",
            )

    if has_rules:
        console.print(rules_table)
    else:
        console.print("[dim]No allow/deny/ask rules configured.[/dim]")


def _print_permission_check_result(result: PermissionCheckResult) -> None:
    decision = {
        "allow": "[green]allow[/green]",
        "deny": "[red]deny[/red]",
        "ask": "[magenta]ask[/magenta]",
    }.get(result.decision, result.decision)

    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Workspace", str(result.project_root))
    table.add_row("Tool", result.tool_name)
    table.add_row("Permission Mode", result.active_mode)
    table.add_row(
        "Permission Prompting",
        "enabled" if result.prompting_enabled else "disabled",
    )
    table.add_row("Required Mode", result.required_mode)
    table.add_row("Decision", decision)
    table.add_row("Input Summary", result.input_summary)
    table.add_row("Path Hint", result.path_hint or "none")
    table.add_row("Matched Rule", result.matched_rule or "none")
    table.add_row("Rule Disposition", result.matched_disposition or "none")
    table.add_row("Reason", result.reason or "none")
    table.add_row("Rules Source", result.rules_source)
    console.print(
        Panel.fit(
            table,
            title="[bold blue]Permission Check[/bold blue]",
            border_style="blue",
        )
    )


def _print_prompt_preview(preview: PromptPreview) -> None:
    table = Table(show_header=False, box=None)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Workspace", str(preview.project_root))
    table.add_row("Model", preview.model)
    table.add_row(
        "Capabilities",
        (
            f"{preview.capability_profile.preferred_tool_call_format} / "
            f"{preview.capability_profile.verification_strictness}"
        ),
    )
    table.add_row("Session", preview.active_session_id or "none")
    table.add_row("Workflow", preview.workflow_mode)
    if preview.workflow_decision_kind:
        table.add_row("Decision Kind", preview.workflow_decision_kind)
    if preview.workflow_reason_summary or preview.workflow_reason_code:
        table.add_row(
            "Workflow Reason",
            _format_workflow_reason(
                summary=preview.workflow_reason_summary,
                code=preview.workflow_reason_code,
            ),
        )
    table.add_row("Permission Mode", preview.permission_mode)
    table.add_row("Prompt Format", preview.prompt_format)
    table.add_row(
        "Dynamic Sections",
        ", ".join(preview.prompt_sections) if preview.prompt_sections else "none",
    )
    table.add_row("Task", preview.current_task or "none")

    console.print(
        Panel.fit(
            table,
            title="[bold blue]Prompt Preview[/bold blue]",
            border_style="blue",
        )
    )
    console.print(
        Panel(
            Markdown(preview.content),
            title="[bold blue]Prompt Body[/bold blue]",
            border_style="blue",
        )
    )


def _print_workflow_timeline_entries(
    entries: list,
    *,
    title: str,
    limit: int | None = None,
) -> None:
    if not entries:
        console.print("[dim]No workflow timeline entries recorded.[/dim]")
        return

    visible_entries = list(entries[-limit:] if limit is not None else entries)
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Time", style="white")
    table.add_column("Kind", style="white")
    table.add_column("Mode", style="white")
    table.add_column("Summary", style="white")
    table.add_column("Context", style="dim")

    for entry in reversed(visible_entries):
        table.add_row(
            entry.timestamp or "-",
            entry.kind,
            entry.mode,
            entry.summary,
            _format_workflow_timeline_context(entry),
        )

    console.print(Panel(table, title=title, border_style="blue"))


def _format_workflow_timeline_context(entry) -> str:
    parts: list[str] = []
    if entry.reason_code:
        parts.append(f"code={entry.reason_code}")
    if entry.decision_kind:
        parts.append(entry.decision_kind)
    if entry.scheduled_next_mode:
        parts.append(f"next={entry.scheduled_next_mode}")
    if entry.runner_up_mode:
        parts.append(f"runner-up={entry.runner_up_mode}")
    if entry.prompt_format:
        parts.append(f"prompt={entry.prompt_format}")
    if entry.prompt_sections:
        parts.append(f"sections={len(entry.prompt_sections)}")
    if entry.unresolved_questions:
        parts.append(f"open={len(entry.unresolved_questions)}")
        parts.append(f"next-question={entry.unresolved_questions[0]}")
    if entry.signal_summary:
        parts.append(f"signals={'; '.join(entry.signal_summary[:2])}")
    if entry.artifact_paths:
        parts.append(f"artifacts={len(entry.artifact_paths)}")
    return ", ".join(parts) or "-"


def _print_workflow_highlights(
    highlights: list[str],
    *,
    title: str,
) -> None:
    lines = [f"- {item}" for item in highlights]
    console.print(
        Panel(
            "\n".join(lines),
            title=title,
            border_style="blue",
        )
    )


def _format_workflow_filters(
    *,
    mode: str | None,
    kind: str | None,
    limit: int | None,
) -> str:
    parts: list[str] = []
    if mode:
        parts.append(f"mode={mode}")
    if kind:
        parts.append(f"kind={kind}")
    if limit is not None:
        parts.append(f"limit={limit}")
    return ", ".join(parts) or "none"


def _format_workflow_reason(*, summary: str | None, code: str | None) -> str:
    if summary and code:
        return f"{summary} ({code})"
    return summary or code or "none"


def _coerce_permission_check_arguments(
    tool,
    *,
    input_value: str | None,
    args_json: str | None,
) -> dict[str, object]:
    arguments = _parse_permission_args_json(args_json)
    if input_value is not None:
        target_key = _simple_permission_input_key(tool)
        if target_key is None:
            raise click.ClickException(
                "This tool does not support a simple positional input. "
                "Provide structured arguments with `--args`."
            )
        if target_key in arguments:
            raise click.ClickException(
                f"`{target_key}` was provided twice; use either the positional input or `--args`."
            )
        arguments[target_key] = input_value

    required_fields = list(tool.parameters.get("required", []))
    missing = [field for field in required_fields if field not in arguments]
    if missing:
        raise click.ClickException(
            f"Tool `{tool.name}` is missing required arguments: {', '.join(missing)}. "
            "Provide them with `--args` as a JSON object."
        )
    return arguments


def _parse_permission_args_json(args_json: str | None) -> dict[str, object]:
    if not args_json:
        return {}
    try:
        payload = json.loads(args_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"`--args` must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise click.ClickException("`--args` must decode to a JSON object.")
    return payload


def _simple_permission_input_key(tool) -> str | None:
    explicit_keys = {
        "bash": "command",
        "read": "file_path",
        "write": "file_path",
        "edit": "file_path",
        "patch": "file_path",
        "glob": "pattern",
        "grep": "pattern",
        "git": "action",
        "project_memory_read": "section",
        "notepad_read": "section",
    }
    if tool.name in explicit_keys:
        return explicit_keys[tool.name]

    parameters = tool.parameters
    properties = parameters.get("properties", {})
    required_fields = list(parameters.get("required", []))
    if len(required_fields) != 1:
        return None
    candidate = required_fields[0]
    schema = properties.get(candidate, {})
    if schema.get("type") == "string":
        return candidate
    return None


if __name__ == "__main__":
    main()
