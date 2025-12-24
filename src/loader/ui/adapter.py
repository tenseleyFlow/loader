"""Event adapter bridging Agent events to Textual messages."""

from dataclasses import dataclass

from textual.message import Message

from ..agent.loop import AgentEvent


# Custom Textual messages for TUI updates
@dataclass
class ThinkingStarted(Message):
    """LLM is generating a response."""

    pass


@dataclass
class StreamChunk(Message):
    """Chunk of streaming response."""

    content: str
    is_end: bool = False


@dataclass
class ToolCallStarted(Message):
    """Tool execution started."""

    tool_name: str
    tool_args: dict


@dataclass
class ToolCallCompleted(Message):
    """Tool execution completed."""

    tool_name: str
    content: str
    is_error: bool = False
    # For edit tool diffs
    old_string: str | None = None
    new_string: str | None = None
    file_path: str | None = None


@dataclass
class PlanCreated(Message):
    """Plan was generated for a complex task."""

    content: str


@dataclass
class StepStarted(Message):
    """A plan step has started."""

    step_info: str


@dataclass
class RecoveryAttempted(Message):
    """Error recovery was attempted."""

    attempt: int
    max_attempts: int = 3


@dataclass
class ErrorOccurred(Message):
    """An error occurred."""

    content: str


@dataclass
class ResponseComplete(Message):
    """Final response is complete."""

    content: str


class EventAdapter:
    """Adapts Agent callback events to Textual messages."""

    def __init__(self, app: "LoaderApp") -> None:  # noqa: F821
        self.app = app
        self._current_tool_args: dict | None = None

    def handle_event(self, event: AgentEvent) -> None:
        """Convert AgentEvent to appropriate Textual message and post it."""
        match event.type:
            case "thinking":
                self.app.call_from_thread(
                    self.app.post_message, ThinkingStarted()
                )

            case "stream":
                self.app.call_from_thread(
                    self.app.post_message,
                    StreamChunk(content=event.content, is_end=event.is_stream_end),
                )

            case "plan":
                self.app.call_from_thread(
                    self.app.post_message, PlanCreated(content=event.content)
                )

            case "step":
                self.app.call_from_thread(
                    self.app.post_message, StepStarted(step_info=event.step_info or "")
                )

            case "tool_call":
                # Store args for potential diff display
                self._current_tool_args = event.tool_args
                self.app.call_from_thread(
                    self.app.post_message,
                    ToolCallStarted(
                        tool_name=event.tool_name or "",
                        tool_args=event.tool_args or {},
                    ),
                )

            case "tool_result":
                # Check if this was an edit tool for diff display
                old_string = None
                new_string = None
                file_path = None

                if event.tool_name == "edit" and self._current_tool_args:
                    old_string = self._current_tool_args.get("old_string")
                    new_string = self._current_tool_args.get("new_string")
                    file_path = self._current_tool_args.get("file_path")

                self.app.call_from_thread(
                    self.app.post_message,
                    ToolCallCompleted(
                        tool_name=event.tool_name or "",
                        content=event.content,
                        is_error=False,  # TODO: pass error state from agent
                        old_string=old_string,
                        new_string=new_string,
                        file_path=file_path,
                    ),
                )
                self._current_tool_args = None

            case "recovery":
                self.app.call_from_thread(
                    self.app.post_message,
                    RecoveryAttempted(
                        attempt=event.recovery_attempt or 1,
                        max_attempts=3,
                    ),
                )

            case "error":
                self.app.call_from_thread(
                    self.app.post_message, ErrorOccurred(content=event.content)
                )

            case "response":
                self.app.call_from_thread(
                    self.app.post_message, ResponseComplete(content=event.content)
                )
