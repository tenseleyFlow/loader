"""Event adapter bridging Agent events to Textual messages."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual.message import Message

from ..agent.loop import AgentEvent

if TYPE_CHECKING:
    from ..agent.reasoning import (
        TaskDecomposition,
        Subtask,
        SelfCritique,
        ConfidenceAssessment,
        ActionVerification,
        TaskCompletionCheck,
        RollbackPlan,
        RollbackAction,
    )


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
class ConfirmationRequired(Message):
    """A destructive operation needs user confirmation."""

    tool_name: str
    confirm_message: str
    details: str = ""


@dataclass
class ResponseComplete(Message):
    """Final response is complete."""

    content: str


@dataclass
class SteeringReceived(Message):
    """A steering message was received and injected into the conversation."""

    content: str


@dataclass
class DecompositionCreated(Message):
    """Task was decomposed into subtasks."""

    content: str
    decomposition: "TaskDecomposition | None" = None


@dataclass
class SubtaskStarted(Message):
    """A subtask has started."""

    content: str
    subtask: "Subtask | None" = None


@dataclass
class ConfidenceAssessed(Message):
    """Confidence was assessed for an action."""

    content: str
    tool_name: str
    confidence: "ConfidenceAssessment | None" = None


@dataclass
class CritiquePerformed(Message):
    """Self-critique was performed."""

    content: str
    critique: "SelfCritique | None" = None


@dataclass
class VerificationPerformed(Message):
    """Post-action verification was performed."""

    content: str
    tool_name: str
    verification: "ActionVerification | None" = None


@dataclass
class CompletionCheckPerformed(Message):
    """Task completion was checked."""

    content: str
    completion_check: "TaskCompletionCheck | None" = None


@dataclass
class RollbackTracked(Message):
    """A rollback action was tracked."""

    content: str
    rollback_action: "RollbackAction | None" = None


@dataclass
class RollbackSummary(Message):
    """Summary of rollback plan at task completion."""

    content: str
    rollback_plan: "RollbackPlan | None" = None


class EventAdapter:
    """Adapts Agent callback events to Textual messages."""

    DEBUG_LOG_FILE = "/tmp/loader_debug.log"

    def __init__(self, app: "LoaderApp") -> None:  # noqa: F821
        self.app = app
        self._tool_args_queue: list[tuple[str, dict]] = []  # Queue of (tool_name, args)
        # Clear debug log on start
        try:
            with open(self.DEBUG_LOG_FILE, "w") as f:
                f.write("=== Loader Debug Log ===\n")
        except Exception:
            pass

    def _debug_log(self, message: str) -> None:
        """Write debug message to log file."""
        try:
            with open(self.DEBUG_LOG_FILE, "a") as f:
                f.write(f"{message}\n")
        except Exception:
            pass

    def handle_event(self, event: AgentEvent) -> None:
        """Convert AgentEvent to appropriate Textual message and post it."""
        self._debug_log(f"handle_event: type={event.type}")
        match event.type:
            case "thinking":
                self.app.post_message(ThinkingStarted())

            case "stream":
                self.app.post_message(
                    StreamChunk(content=event.content, is_end=event.is_stream_end)
                )

            case "plan":
                self.app.post_message(PlanCreated(content=event.content))

            case "step":
                self.app.post_message(StepStarted(step_info=event.step_info or ""))

            case "tool_call":
                # Queue args for matching with result (FIFO)
                tool_name = event.tool_name or ""
                tool_args = event.tool_args or {}
                self._tool_args_queue.append((tool_name, tool_args))

                # Debug: log tool args for edit/write (helps diagnose diff view issues)
                self._debug_log(f"tool_call '{tool_name}': queued, keys={list(tool_args.keys())}")
                if tool_name == "write":
                    content = tool_args.get("content", "")
                    self._debug_log(f"  write content: {len(content) if content else 0} chars")
                elif tool_name == "edit":
                    self._debug_log(f"  edit old_string: {bool(tool_args.get('old_string'))}, new_string: {bool(tool_args.get('new_string'))}")

                self.app.post_message(
                    ToolCallStarted(
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
                )

            case "tool_result":
                # Get matching args from queue (FIFO)
                tool_name = event.tool_name or ""
                tool_args = {}

                # Find matching tool_call in queue (should be FIFO but handle mismatch)
                if self._tool_args_queue:
                    # Try to find matching tool by name, fallback to FIFO
                    for i, (queued_name, queued_args) in enumerate(self._tool_args_queue):
                        if queued_name == tool_name:
                            tool_args = queued_args
                            self._tool_args_queue.pop(i)
                            self._debug_log(f"tool_result '{tool_name}': matched in queue, keys={list(tool_args.keys())}")
                            break
                    else:
                        # No match found, use FIFO
                        popped_name, tool_args = self._tool_args_queue.pop(0)
                        self._debug_log(f"tool_result '{tool_name}': no match, used FIFO (got '{popped_name}'), keys={list(tool_args.keys())}")
                else:
                    self._debug_log(f"tool_result '{tool_name}': queue was EMPTY!")

                # Extract diff info for edit/write tools
                old_string = None
                new_string = None
                file_path = None

                if tool_name == "edit":
                    if tool_args:
                        # Try multiple key names that models might use
                        old_string = (
                            tool_args.get("old_string")
                            or tool_args.get("old")
                            or tool_args.get("original")
                            or tool_args.get("search")
                            or tool_args.get("find")
                        )
                        new_string = (
                            tool_args.get("new_string")
                            or tool_args.get("new")
                            or tool_args.get("replacement")
                            or tool_args.get("replace")
                        )
                        file_path = (
                            tool_args.get("file_path")
                            or tool_args.get("path")
                            or tool_args.get("filename")
                            or tool_args.get("file")
                        )
                        self._debug_log(f"  edit extracted: old={bool(old_string)} ({len(old_string) if old_string else 0} chars), new={bool(new_string)} ({len(new_string) if new_string else 0} chars), path={file_path}")
                    else:
                        self._debug_log(f"  edit: tool_args was empty!")
                elif tool_name == "write":
                    # For writes, content is the new file content
                    # Try multiple key names that models might use
                    if tool_args:
                        new_string = (
                            tool_args.get("content")
                            or tool_args.get("contents")
                            or tool_args.get("text")
                            or tool_args.get("data")
                        )
                        file_path = (
                            tool_args.get("file_path")
                            or tool_args.get("path")
                            or tool_args.get("filename")
                        )
                        self._debug_log(f"  write extracted: new={bool(new_string)} ({len(new_string) if new_string else 0} chars), path={file_path}")
                    else:
                        self._debug_log(f"  write: tool_args was empty!")

                self.app.post_message(
                    ToolCallCompleted(
                        tool_name=tool_name,
                        content=event.content,
                        is_error=event.is_error,
                        old_string=old_string,
                        new_string=new_string,
                        file_path=file_path,
                    )
                )

            case "recovery":
                self.app.post_message(
                    RecoveryAttempted(
                        attempt=event.recovery_attempt or 1,
                        max_attempts=3,
                    )
                )

            case "error":
                self.app.post_message(ErrorOccurred(content=event.content))

            case "response":
                self.app.post_message(ResponseComplete(content=event.content))

            case "confirmation":
                # Confirmation is handled via async callback, but we can post a message
                # for UI updates if needed
                self.app.post_message(
                    ConfirmationRequired(
                        tool_name=event.tool_name or "",
                        confirm_message=event.confirm_message or "",
                        details=event.confirm_details or "",
                    )
                )

            case "steering":
                # Steering message was injected into the conversation
                self.app.post_message(SteeringReceived(content=event.content))

            case "decomposition":
                # Task was decomposed into subtasks
                self.app.post_message(DecompositionCreated(
                    content=event.content,
                    decomposition=event.decomposition,
                ))

            case "subtask":
                # A subtask has started
                self.app.post_message(SubtaskStarted(
                    content=event.content,
                    subtask=event.subtask,
                ))

            case "confidence":
                # Confidence was assessed for an action
                self.app.post_message(ConfidenceAssessed(
                    content=event.content,
                    tool_name=event.tool_name or "",
                    confidence=event.confidence,
                ))

            case "critique":
                # Self-critique was performed
                self.app.post_message(CritiquePerformed(
                    content=event.content,
                    critique=event.critique,
                ))

            case "verification":
                # Post-action verification was performed
                self.app.post_message(VerificationPerformed(
                    content=event.content,
                    tool_name=event.tool_name or "",
                    verification=event.verification,
                ))

            case "completion_check":
                # Task completion was checked
                self.app.post_message(CompletionCheckPerformed(
                    content=event.content,
                    completion_check=event.completion_check,
                ))

            case "rollback":
                # A rollback action was tracked
                self.app.post_message(RollbackTracked(
                    content=event.content,
                    rollback_action=event.rollback_action,
                ))

            case "rollback_summary":
                # Summary of rollback plan
                self.app.post_message(RollbackSummary(
                    content=event.content,
                    rollback_plan=event.rollback_plan,
                ))
