"""Main Textual application for Loader TUI."""

import time
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Footer, Static
from textual import work
from textual.worker import Worker, get_current_worker

from ..agent.loop import Agent, AgentEvent
from .adapter import (
    ErrorOccurred,
    EventAdapter,
    PlanCreated,
    ResponseComplete,
    StepStarted,
    StreamChunk,
    ThinkingStarted,
    ToolCallCompleted,
    ToolCallStarted,
)
from .widgets import DiffWidget, InputArea, StatusLine, StreamingText, ToolCallWidget


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
        self._current_tool_widget: ToolCallWidget | None = None
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
        self._add_message(f"[bold blue]You:[/bold blue] {content}", "user-message")

    def _start_timer(self) -> None:
        """Start the elapsed time timer."""
        self._start_time = time.time()
        self._timer_handle = self.set_interval(0.1, self._update_elapsed)

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

        # Add user message to display
        self._add_user_message(user_input)

        # Start agent task
        self.run_agent(user_input)

    @work(exclusive=True, thread=True)
    def run_agent(self, user_input: str) -> str:
        """Run the agent in a worker thread."""
        worker = get_current_worker()

        def on_event(event: AgentEvent) -> None:
            if not worker.is_cancelled:
                self.adapter.handle_event(event)

        # Run synchronously from thread
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.agent.run(user_input, on_event=on_event)
            )
        finally:
            loop.close()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker state changes."""
        if event.state.name == "SUCCESS":
            self.is_generating = False
            self._stop_timer()
            self.query_one(StatusLine).set_generating(False)

    # Message handlers from adapter
    def on_thinking_started(self, message: ThinkingStarted) -> None:
        """Handle thinking started."""
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
        widget.set_running()
        msg_area.mount(widget)
        self._current_tool_widget = widget
        msg_area.scroll_end(animate=False)

    def on_tool_call_completed(self, message: ToolCallCompleted) -> None:
        """Handle tool call completion."""
        msg_area = self.query_one("#message-area", ScrollableContainer)

        # Check if this is an edit tool with diff info
        if message.tool_name == "edit" and message.old_string and message.new_string:
            # Replace tool widget with diff widget
            if self._current_tool_widget:
                self._current_tool_widget.remove()

            diff_widget = DiffWidget(
                file_path=message.file_path or "",
                old_string=message.old_string,
                new_string=message.new_string,
            )
            msg_area.mount(diff_widget)
        elif self._current_tool_widget:
            # Update existing tool widget with result
            self._current_tool_widget.set_result(
                message.content, is_error=message.is_error
            )

        self._current_tool_widget = None
        msg_area.scroll_end(animate=False)

    def on_plan_created(self, message: PlanCreated) -> None:
        """Handle plan creation."""
        msg_area = self.query_one("#message-area", ScrollableContainer)
        plan_widget = Static(
            f"[bold blue]Plan[/bold blue]\n{message.content}",
            classes="plan-container",
        )
        msg_area.mount(plan_widget)
        msg_area.scroll_end(animate=False)

    def on_step_started(self, message: StepStarted) -> None:
        """Handle step start."""
        self._add_message(
            f"[bold yellow]{message.step_info}[/bold yellow]", "step-progress"
        )

    def on_error_occurred(self, message: ErrorOccurred) -> None:
        """Handle errors."""
        self._add_message(f"[bold red]Error:[/bold red] {message.content}")

    def on_response_complete(self, message: ResponseComplete) -> None:
        """Handle response completion."""
        # Response was already streamed, nothing extra needed
        pass

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
