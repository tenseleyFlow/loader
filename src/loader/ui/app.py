"""Main Textual application for Loader TUI."""

import time
from pathlib import Path

from rich.markup import escape

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Footer, Static
from textual import work
from textual.worker import Worker, get_current_worker

from ..agent.loop import Agent, AgentEvent
from .adapter import (
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
from .widgets import ConfirmationModal, DiffWidget, InputArea, StatusLine, StreamingText, ToolCallWidget


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
        self._tool_widget_queue: list[ToolCallWidget] = []  # Queue of pending tool widgets
        self._timer_handle = None

    def compose(self) -> ComposeResult:
        yield Container(
            ScrollableContainer(id="message-area"),
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

        # Handle special commands
        if user_input.lower() == "exit":
            self.exit()
            return

        if user_input.lower() == "clear":
            self.action_clear_messages()
            return

        # If agent is running, this is a steering message
        if self.is_generating and self.agent.is_running:
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

    async def _request_confirmation(
        self,
        tool_name: str,
        message: str,
        details: str,
    ) -> bool:
        """Show confirmation modal and wait for user response."""
        modal = ConfirmationModal(
            tool_name=tool_name,
            message=message,
            details=details,
        )
        return await self.push_screen_wait(modal)

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

        self._current_streaming.append(message.content)
        msg_area.scroll_end(animate=False)

        if message.is_end:
            self._current_streaming.stop_streaming()
            self._current_streaming = None

    def on_tool_call_started(self, message: ToolCallStarted) -> None:
        """Handle tool call start."""
        msg_area = self.query_one("#message-area", ScrollableContainer)

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

        # Get the corresponding tool widget from queue (FIFO)
        tool_widget = self._tool_widget_queue.pop(0) if self._tool_widget_queue else None

        # Check if this is an edit tool with diff info
        if message.tool_name == "edit" and message.old_string and message.new_string:
            # Replace tool widget with diff widget
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
        # Response was already streamed, nothing extra needed
        pass

    def on_steering_received(self, message: SteeringReceived) -> None:
        """Handle steering message being processed by agent."""
        # The steering message was already displayed when sent,
        # this confirms it was injected into the agent's context
        self._add_message(
            "[dim italic]↪ Steering message injected, agent will incorporate it...[/dim italic]"
        )

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
        check = message.completion_check
        if check and not check.is_complete:
            lines = ["[bold yellow]⚡ Task not complete yet![/bold yellow]"]
            if check.accomplished:
                lines.append(f"  [green]Done:[/green] {', '.join(check.accomplished[:3])}")
            if check.suggested_next_steps:
                lines.append("  [yellow]Next steps:[/yellow]")
                for step in check.suggested_next_steps[:2]:
                    lines.append(f"    → {step}")
            lines.append("  [italic]Continuing...[/italic]")
            self._add_message("\n".join(lines), "completion-check")
        else:
            self._add_message(f"[dim]{escape(message.content)}[/dim]")

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
