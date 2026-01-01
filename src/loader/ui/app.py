"""Main Textual application for Loader TUI."""

import asyncio
import time
from pathlib import Path

from rich.markup import escape

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Footer, Input, Static
from textual import work
from textual.worker import Worker, get_current_worker

from ..agent.loop import Agent, AgentEvent
from .adapter import (
    ClearStream,
    CompletionCheckPerformed,
    ConfidenceAssessed,
    CritiquePerformed,
    DecompositionCreated,
    ErrorOccurred,
    EventAdapter,
    PlanCreated,
    ResponseComplete,
    RollbackSummary,
    RollbackTracked,
    SteeringReceived,
    StepStarted,
    StreamChunk,
    SubtaskStarted,
    ThinkingStarted,
    ToolCallCompleted,
    ToolCallStarted,
    VerificationPerformed,
)
from .widgets import ApprovalBar, ConfirmationModal, DiffWidget, InputArea, StatusLine, StreamingText, ToolCallWidget


class LoaderApp(App):
    """Main Textual application for Loader."""

    CSS_PATH = Path(__file__).parent / "styles" / "theme.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_messages", "Clear", show=True),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    # Reactive state
    is_generating: reactive[bool] = reactive(False)

    def __init__(
        self,
        agent: Agent,
        model_name: str = "",
        mode: str = "Native",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.model_name = model_name
        self.mode = mode
        self.adapter = EventAdapter(self)
        self._start_time: float = 0.0
        self._current_streaming: StreamingText | None = None
        self._streamed_content: bool = False  # Track if any content was streamed
        self._tool_widget_queue: list[ToolCallWidget] = []  # Queue of pending tool widgets
        self._timer_handle = None
        # Approval bar state
        self._pending_confirmation: asyncio.Future | None = None
        self._pending_command: str = ""

    def _debug_log(self, message: str) -> None:
        """Write debug message to log file."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"{message}\n")
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(id="message-area"),
            ApprovalBar(id="approval-bar"),
            InputArea(id="input-area"),
            StatusLine(id="status-line"),
            id="main-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize on mount."""
        # Set status line info
        status = self.query_one(StatusLine)
        status.model = self.model_name
        status.mode = self.mode

        # Focus input
        self.query_one(InputArea).focus_input()

        # Show welcome message
        self._add_message(
            "[dim]Type a message to get started. "
            "Press Ctrl+C to quit, Ctrl+L to clear.[/dim]"
        )
        self._add_message(
            "[dim]Commands: /help, /model, /clear, /exit[/dim]"
        )

    def _add_message(self, content: str, classes: str = "") -> None:
        """Add a message to the message area."""
        msg_area = self.query_one("#message-area", ScrollableContainer)
        widget = Static(content, classes=classes)
        msg_area.mount(widget)
        msg_area.scroll_end(animate=False)

    def _add_user_message(self, content: str) -> None:
        """Add a user message to the display."""
        self._add_message(f"[bold blue]You:[/bold blue] {escape(content)}", "user-message")

    def _start_timer(self) -> None:
        """Start the elapsed time timer."""
        self._start_time = time.time()
        self._timer_handle = self.set_interval(0.5, self._update_elapsed)  # Update every 500ms

    def _stop_timer(self) -> None:
        """Stop the elapsed time timer."""
        if self._timer_handle:
            self._timer_handle.stop()
            self._timer_handle = None

    def _update_elapsed(self) -> None:
        """Update elapsed time in status line."""
        if self.is_generating:
            elapsed = time.time() - self._start_time
            self.query_one(StatusLine).update_elapsed(elapsed)

    # Event handlers for input
    def on_input_area_submitted(self, message: InputArea.Submitted) -> None:
        """Handle user input submission."""
        user_input = message.value

        # Handle slash commands
        if user_input.startswith("/"):
            self._handle_command(user_input)
            return

        # Handle legacy commands (without slash) for backwards compat
        if user_input.lower() in ("exit", "quit"):
            self.exit()
            return

        if user_input.lower() == "clear":
            self.action_clear_messages()
            return

        # If agent is running, this is a steering message
        if self.is_generating and self.agent.is_running:
            # Finalize current streaming so new content appears below user's message
            if self._current_streaming is not None:
                self._current_streaming.stop_streaming()
                self._current_streaming = None
            self._add_steering_message(user_input)
            self.agent.steer(user_input)
            return

        # Add user message to display
        self._add_user_message(user_input)

        # Show generating status immediately (before async work starts)
        self.is_generating = True
        self._start_timer()
        self.query_one(StatusLine).set_generating(True)

        # Start agent task
        self.run_agent(user_input)

    def _add_steering_message(self, content: str) -> None:
        """Add a steering message to the display."""
        self._add_message(
            f"[bold magenta]↪ Steering:[/bold magenta] {escape(content)}",
            "steering-message"
        )

    def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command[1:].split(maxsplit=1)  # Remove leading /
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("exit", "quit", "q"):
            self._add_message("[dim]Goodbye![/dim]")
            self.exit()

        elif cmd in ("clear", "c"):
            self.action_clear_messages()

        elif cmd in ("help", "h", "?"):
            self._show_help()

        elif cmd in ("model", "m"):
            self._handle_model_command(args)

        elif cmd == "models":
            self._handle_model_command("")  # List models

        else:
            self._add_message(f"[red]Unknown command: /{cmd}[/red]\nType /help for available commands.")

    def _show_help(self) -> None:
        """Show help message with available commands."""
        help_text = """[bold]Available Commands:[/bold]

[cyan]/help[/cyan], [cyan]/h[/cyan]          Show this help message
[cyan]/exit[/cyan], [cyan]/q[/cyan]          Exit the application
[cyan]/clear[/cyan], [cyan]/c[/cyan]         Clear the conversation
[cyan]/model[/cyan], [cyan]/models[/cyan]    Open model selector (fzf-style)
[cyan]/model[/cyan] [dim]<name>[/dim]     Switch to a specific model

[bold]Shortcuts:[/bold]
[dim]Ctrl+C[/dim]            Exit
[dim]Ctrl+L[/dim]            Clear conversation"""
        self._add_message(help_text)

    def _handle_model_command(self, args: str) -> None:
        """Handle /model command - switch or show selector."""
        if not args:
            # Show interactive model selector
            self._show_model_selector()
        else:
            # Switch to specified model directly
            self._switch_model(args.strip())

    def _show_model_selector(self) -> None:
        """Show interactive model selection modal."""
        # Use Textual's worker to fetch models then show modal
        self.run_worker(self._fetch_and_show_selector())

    async def _fetch_and_show_selector(self) -> None:
        """Fetch models and show the selection modal."""
        from .widgets import ModelSelectModal

        try:
            models = []
            if hasattr(self.agent.backend, "list_models"):
                models = await self.agent.backend.list_models()

            if models:
                current = self.agent.backend.model if hasattr(self.agent.backend, "model") else ""

                def on_select(selected: str | None) -> None:
                    if selected:
                        self._switch_model(selected)

                # Schedule on main thread - workers can use call_later
                self.call_later(
                    lambda: self.push_screen(ModelSelectModal(models, current), on_select)
                )
            else:
                self._add_message("[yellow]No models found. Is Ollama running?[/yellow]")
        except Exception as e:
            self._add_message(f"[red]Error listing models: {e}[/red]")

    def _switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        if hasattr(self.agent.backend, "model"):
            old_model = self.agent.backend.model
            self.agent.backend.model = model_name
            self.model_name = model_name
            # Update status line
            self.query_one(StatusLine).model = model_name
            # Update mode based on new model's capabilities
            if hasattr(self.agent.backend, "supports_native_tools"):
                supports_native = self.agent.backend.supports_native_tools()
                self.mode = "Native" if supports_native else "ReAct"
                self.query_one(StatusLine).mode = self.mode
            self._add_message(f"[green]Switched model:[/green] {old_model} → [bold]{model_name}[/bold]")
        else:
            self._add_message("[red]Model switching not supported for this backend[/red]")

    async def _request_confirmation(
        self,
        tool_name: str,
        message: str,
        details: str,
    ) -> bool:
        """Show approval bar and wait for user response."""
        # Create a future to wait on
        loop = asyncio.get_event_loop()
        self._pending_confirmation = loop.create_future()
        self._pending_command = details

        # Show the approval bar - must use call_from_thread since we're in worker
        approval_bar = self.query_one("#approval-bar", ApprovalBar)

        def show_bar():
            try:
                approval_bar.show_approval(tool_name, message, details)
                # Debug log
                with open("/tmp/loader_debug.log", "a") as f:
                    f.write(f"[approval] Bar shown, waiting for user input\n")
            except Exception as e:
                with open("/tmp/loader_debug.log", "a") as f:
                    f.write(f"[approval] Error showing bar: {e}\n")

        self.call_from_thread(show_bar)

        # Wait for user response with timeout
        try:
            result = await asyncio.wait_for(self._pending_confirmation, timeout=300.0)
            try:
                with open("/tmp/loader_debug.log", "a") as f:
                    f.write(f"[approval] Got result: {result}\n")
            except Exception:
                pass
            return result
        except asyncio.TimeoutError:
            try:
                with open("/tmp/loader_debug.log", "a") as f:
                    f.write(f"[approval] Timeout waiting for user\n")
            except Exception:
                pass
            return False
        finally:
            self._pending_confirmation = None
            self._pending_command = ""
            # Hide the bar
            self.call_from_thread(approval_bar.hide_approval)

    def on_approval_bar_approved(self, event: ApprovalBar.Approved) -> None:
        """Handle approval from the bar."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"[approval] Approved handler called\n")
        except Exception:
            pass
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(True)

    def on_approval_bar_rejected(self, event: ApprovalBar.Rejected) -> None:
        """Handle rejection from the bar."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"[approval] Rejected handler called\n")
        except Exception:
            pass
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(False)
        # Refocus input
        self.query_one(InputArea).focus_input()

    def on_approval_bar_edit_requested(self, event: ApprovalBar.EditRequested) -> None:
        """Handle edit request - put command in input for editing."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"[approval] Edit handler called\n")
        except Exception:
            pass
        if self._pending_confirmation and not self._pending_confirmation.done():
            self._pending_confirmation.set_result(False)
        # Put the command in the input field for editing
        input_area = self.query_one(InputArea)
        input_widget = input_area.query_one("#user-input", Input)
        input_widget.value = event.command
        input_widget.cursor_position = len(event.command)
        input_area.focus_input()

    @work(exclusive=True)
    async def run_agent(self, user_input: str) -> str:
        """Run the agent asynchronously."""
        import asyncio
        import httpx

        worker = get_current_worker()

        async def on_event(event: AgentEvent) -> None:
            if not worker.is_cancelled:
                self.adapter.handle_event(event)
                # Yield control to let UI update
                await asyncio.sleep(0)

        async def on_confirmation(tool_name: str, message: str, details: str) -> bool:
            """Handle confirmation requests from agent."""
            if worker.is_cancelled:
                return False
            return await self._request_confirmation(tool_name, message, details)

        try:
            return await self.agent.run(
                user_input,
                on_event=on_event,
                on_confirmation=on_confirmation,
            )
        except httpx.ReadTimeout:
            self._add_message(
                "[bold red]Request timed out.[/bold red] The model is taking too long.\n"
                "[dim]Try: smaller model, simpler prompt, or increase --ctx[/dim]"
            )
            return ""
        except httpx.ConnectError:
            self._add_message(
                "[bold red]Connection error.[/bold red] Cannot reach Ollama.\n"
                "[dim]Is Ollama running? Try: ollama serve[/dim]"
            )
            return ""
        except Exception as e:
            import traceback
            error_msg = f"[bold red]Error:[/bold red] {escape(str(e))}"
            # Show traceback in debug scenarios
            tb = traceback.format_exc()
            if "Traceback" in tb:
                error_msg += f"\n[dim]{escape(tb[-500:])}[/dim]"
            self._add_message(error_msg)
            return ""

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes."""
        if event.state.name in ("SUCCESS", "ERROR", "CANCELLED"):
            self.is_generating = False
            self._stop_timer()
            self.query_one(StatusLine).set_generating(False)

    # Message handlers from adapter
    def on_thinking_started(self, message: ThinkingStarted) -> None:
        """Handle thinking started (may be called multiple times per task)."""
        # Finalize any previous streaming widget to prevent appending to old content
        if self._current_streaming is not None:
            self._current_streaming.stop_streaming()
            self._current_streaming = None

        # Reset streamed content flag for this response iteration
        self._streamed_content = False

        # Status is already set in on_input_area_submitted, but ensure it stays on
        if not self.is_generating:
            self.is_generating = True
            self._start_timer()
            self.query_one(StatusLine).set_generating(True)

    def on_stream_chunk(self, message: StreamChunk) -> None:
        """Handle streaming content."""
        msg_area = self.query_one("#message-area", ScrollableContainer)

        if self._current_streaming is None:
            # Create new streaming widget
            self._current_streaming = StreamingText()
            self._current_streaming.start_streaming()
            msg_area.mount(self._current_streaming)

        # Filter content through safeguards before displaying
        # This removes bracket tool calls, code blocks, etc. from stream
        filtered_content = self.agent.safeguards.filter_stream_chunk(message.content)

        if filtered_content:
            self._current_streaming.append(filtered_content)
            msg_area.scroll_end(animate=False)

        # Track that we've shown actual content
        if message.content.strip():
            self._streamed_content = True

        if message.is_end:
            self._current_streaming.stop_streaming()
            self._current_streaming = None

    def on_clear_stream(self, message: ClearStream) -> None:
        """Clear/remove the current streaming content (used when raw tool calls are detected)."""
        if self._current_streaming is not None:
            # Remove the streaming widget entirely - it contained raw JSON tool calls
            self._current_streaming.remove()
            self._current_streaming = None
            self._streamed_content = False

    def on_tool_call_started(self, message: ToolCallStarted) -> None:
        """Handle tool call start."""
        msg_area = self.query_one("#message-area", ScrollableContainer)

        # Finalize any ongoing streaming - tool calls interrupt thinking
        if self._current_streaming is not None:
            self._current_streaming.stop_streaming()
            self._current_streaming = None

        # Create tool widget
        widget = ToolCallWidget(
            tool_name=message.tool_name,
            tool_args=message.tool_args,
        )
        msg_area.mount(widget)
        widget.set_running()  # Must be after mount() so children exist
        self._tool_widget_queue.append(widget)  # Add to queue
        msg_area.scroll_end(animate=False)

    def on_tool_call_completed(self, message: ToolCallCompleted) -> None:
        """Handle tool call completion."""
        msg_area = self.query_one("#message-area", ScrollableContainer)

        # Debug: log what we received
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"on_tool_call_completed: tool={message.tool_name}, new_string={bool(message.new_string)}, old_string={bool(message.old_string)}, file_path={message.file_path}\n")
        except Exception:
            pass

        # Get the corresponding tool widget from queue (FIFO)
        tool_widget = self._tool_widget_queue.pop(0) if self._tool_widget_queue else None

        # Check if this is an edit tool with diff info
        # Note: old_string can be empty string (inserting), so check `is not None`
        if message.tool_name == "edit" and message.new_string and message.old_string is not None:
            # Replace tool widget with diff widget
            self._debug_log(f"  -> showing EDIT diff widget (old={len(message.old_string)} chars, new={len(message.new_string)} chars)")
            if tool_widget:
                tool_widget.remove()

            diff_widget = DiffWidget(
                file_path=message.file_path or "",
                old_string=message.old_string,
                new_string=message.new_string,
            )
            msg_area.mount(diff_widget)
        # Check if this is a write tool - show as diff (new file)
        elif message.tool_name == "write" and message.new_string:
            self._debug_log(f"  -> showing WRITE diff widget ({len(message.new_string)} chars)")
            if tool_widget:
                tool_widget.remove()

            diff_widget = DiffWidget(
                file_path=message.file_path or "",
                old_string="",  # Empty = new file
                new_string=message.new_string,
            )
            msg_area.mount(diff_widget)
        elif tool_widget:
            # Update existing tool widget with result
            self._debug_log("  -> showing regular tool widget result")
            tool_widget.set_result(
                message.content, is_error=message.is_error
            )

        msg_area.scroll_end(animate=False)

    def on_plan_created(self, message: PlanCreated) -> None:
        """Handle plan creation."""
        msg_area = self.query_one("#message-area", ScrollableContainer)
        plan_widget = Static(
            f"[bold blue]Plan[/bold blue]\n{escape(message.content)}",
            classes="plan-container",
        )
        msg_area.mount(plan_widget)
        msg_area.scroll_end(animate=False)

    def on_step_started(self, message: StepStarted) -> None:
        """Handle step start."""
        self._add_message(
            f"[bold yellow]{escape(message.step_info)}[/bold yellow]", "step-progress"
        )

    def on_error_occurred(self, message: ErrorOccurred) -> None:
        """Handle errors."""
        self._add_message(f"[bold red]Error:[/bold red] {escape(message.content)}")

    def on_response_complete(self, message: ResponseComplete) -> None:
        """Handle response completion."""
        # If no content was streamed but we have a response, display it
        # This handles cases like empty LLM responses with fallback messages
        if not self._streamed_content and message.content.strip():
            self._add_message(message.content)

    def on_steering_received(self, message: SteeringReceived) -> None:
        """Handle steering message being processed by agent."""
        # Don't display anything - auto-steering is internal
        # Only show if this was user-initiated (but we don't track that currently)
        pass

    # Reasoning event handlers
    def on_decomposition_created(self, message: DecompositionCreated) -> None:
        """Handle task decomposition."""
        msg_area = self.query_one("#message-area", ScrollableContainer)
        decomp = message.decomposition
        if decomp:
            lines = [
                f"[bold cyan]📋 Task Decomposed[/bold cyan] ({len(decomp.subtasks)} subtasks)"
            ]
            for i, st in enumerate(decomp.subtasks, 1):
                status_icon = {
                    "pending": "○",
                    "in_progress": "◐",
                    "completed": "●",
                    "failed": "✗",
                }.get(st.status, "?")
                deps = f" [dim](after: {', '.join(st.dependencies)})[/dim]" if st.dependencies else ""
                lines.append(f"  {status_icon} {i}. {st.description}{deps}")
            widget = Static("\n".join(lines), classes="decomposition-container")
            msg_area.mount(widget)
        else:
            self._add_message(f"[cyan]📋 {escape(message.content)}[/cyan]")
        msg_area.scroll_end(animate=False)

    def on_subtask_started(self, message: SubtaskStarted) -> None:
        """Handle subtask start."""
        self._add_message(
            f"[bold yellow]▸ {escape(message.content)}[/bold yellow]",
            "subtask-progress"
        )

    def on_confidence_assessed(self, message: ConfidenceAssessed) -> None:
        """Handle confidence assessment."""
        confidence = message.confidence
        if confidence:
            level = confidence.level.name
            score = confidence.score
            # Color based on confidence level
            color = {
                1: "red",      # VERY_LOW
                2: "orange1",  # LOW
                3: "yellow",   # MEDIUM
                4: "green",    # HIGH
                5: "bright_green",  # VERY_HIGH
            }.get(score, "white")

            content = f"[{color}]🎯 Confidence: {level} ({score}/5)[/{color}]"
            if confidence.reasoning:
                content += f"\n[dim]   {confidence.reasoning}[/dim]"
            if confidence.risks:
                content += f"\n[dim red]   Risks: {', '.join(confidence.risks[:2])}[/dim red]"
            self._add_message(content, "confidence-assessment")
        else:
            self._add_message(f"[yellow]🎯 {escape(message.content)}[/yellow]")

    def on_critique_performed(self, message: CritiquePerformed) -> None:
        """Handle self-critique."""
        critique = message.critique
        if critique and critique.issues_found:
            lines = ["[bold magenta]🔍 Self-Critique[/bold magenta]"]
            for issue in critique.issues_found[:3]:
                lines.append(f"  [yellow]⚠[/yellow] {issue}")
            if critique.suggestions:
                lines.append("  [dim]Suggestions:[/dim]")
                for suggestion in critique.suggestions[:2]:
                    lines.append(f"    → {suggestion}")
            if critique.should_revise:
                lines.append("  [italic]Revising response...[/italic]")
            self._add_message("\n".join(lines), "critique-container")
        elif critique:
            self._add_message(
                "[green]🔍 Self-critique: No issues found[/green]",
                "critique-container"
            )
        else:
            self._add_message(f"[magenta]🔍 {escape(message.content)}[/magenta]")

    def on_verification_performed(self, message: VerificationPerformed) -> None:
        """Handle post-action verification."""
        verification = message.verification
        if verification:
            if verification.verified:
                self._add_message(
                    f"[green]✓ Verified: {message.tool_name}[/green]",
                    "verification-success"
                )
            else:
                lines = [f"[red]✗ Verification failed: {message.tool_name}[/red]"]
                if verification.discrepancies:
                    for disc in verification.discrepancies[:2]:
                        lines.append(f"  [dim]{disc}[/dim]")
                if verification.correction_suggestion:
                    lines.append(f"  [yellow]→ {verification.correction_suggestion}[/yellow]")
                self._add_message("\n".join(lines), "verification-failed")
        else:
            self._add_message(f"[blue]✓ {escape(message.content)}[/blue]")

    def on_completion_check_performed(self, message: CompletionCheckPerformed) -> None:
        """Handle task completion check."""
        # Don't display anything - completion checks are internal steering
        # The agent will continue automatically if needed
        pass

    def on_rollback_tracked(self, message: RollbackTracked) -> None:
        """Handle rollback action tracking (verbose mode only)."""
        action = message.rollback_action
        if action:
            self._add_message(
                f"[dim]↩ Rollback: {action.description}[/dim]",
                "rollback-tracked"
            )

    def on_rollback_summary(self, message: RollbackSummary) -> None:
        """Handle rollback plan summary at task completion."""
        plan = message.rollback_plan
        if plan and plan.actions:
            lines = [f"[dim cyan]↩ Rollback available ({len(plan.actions)} actions)[/dim cyan]"]
            # Show first few rollback steps
            steps = plan.get_rollback_steps()[:3]
            for step in steps:
                lines.append(f"  [dim]• {step}[/dim]")
            if len(plan.actions) > 3:
                lines.append(f"  [dim]... and {len(plan.actions) - 3} more[/dim]")
            if not plan.can_rollback:
                lines.append("  [dim yellow]⚠ Some actions cannot be undone[/dim yellow]")
            self._add_message("\n".join(lines), "rollback-summary")

    # Actions
    def action_clear_messages(self) -> None:
        """Clear all messages."""
        msg_area = self.query_one("#message-area", ScrollableContainer)
        msg_area.remove_children()
        self.agent.clear_history()
        self._add_message("[dim]Conversation cleared.[/dim]")

    def action_cancel(self) -> None:
        """Cancel current operation."""
        # Cancel any running workers
        self.workers.cancel_all()
        self.is_generating = False
        self._stop_timer()
        self.query_one(StatusLine).set_generating(False)
