"""Event adapter bridging Agent events to Textual messages."""

from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

from textual.message import Message

from ..runtime.events import AgentEvent
from ..utils.file_mutations import build_file_mutation_preview_dict

if TYPE_CHECKING:
    from ..runtime.reasoning_types import (
        ActionVerification,
        ConfidenceAssessment,
        SelfCritique,
        Subtask,
        TaskCompletionCheck,
        TaskDecomposition,
    )
    from ..runtime.rollback import RollbackAction, RollbackPlan


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
    tool_call_id: str | None = None
    phase: str | None = None


@dataclass
class ToolCallCompleted(Message):
    """Tool execution completed."""

    tool_name: str
    content: str
    is_error: bool = False
    phase: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] | None = None
    # For edit tool diffs
    old_string: str | None = None
    new_string: str | None = None
    file_path: str | None = None
    mutation_preview: dict[str, Any] | None = None


@dataclass
class PlanCreated(Message):
    """Plan was generated for a complex task."""

    content: str


@dataclass
class StepStarted(Message):
    """A plan step has started."""

    step_info: str


@dataclass
class TodoListUpdated(Message):
    """Agent updated its todo list."""

    todos: list


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
    preview: dict[str, Any] | None = None


@dataclass
class ResponseComplete(Message):
    """Final response is complete."""

    content: str


@dataclass
class SteeringReceived(Message):
    """A steering message was received and injected into the conversation."""

    content: str


@dataclass
class ClearStream(Message):
    """Clear the current streaming content (used when raw tool calls are detected)."""

    pass


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


@dataclass
class DefinitionOfDoneUpdated(Message):
    """Definition-of-done status changed."""

    content: str
    dod_status: str
    pending_items_count: int = 0
    last_verification_result: str | None = None
    verification_attempt: str | None = None


@dataclass
class WorkflowModeChanged(Message):
    """Workflow mode changed."""

    workflow_mode: str
    content: str


@dataclass
class TurnPhaseChanged(Message):
    """Turn phase changed."""

    turn_phase: str
    content: str


@dataclass
class ArtifactCreated(Message):
    """A workflow artifact was created."""

    content: str
    artifact_kind: str
    artifact_path: str


class EventAdapter:
    """Adapts Agent callback events to Textual messages."""

    DEBUG_LOG_FILE = "/tmp/loader_debug.log"

    def __init__(self, app: "LoaderApp") -> None:  # noqa: F821
        self.app = app
        self._tool_args_queue: list[tuple[str | None, str, dict[str, Any]]] = []
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

    @staticmethod
    def _extract_todos(content: str, tool_args: dict) -> list[dict]:
        """Extract todo items from TodoWrite result or args."""
        import json

        # Try parsing the content as JSON (may be wrapped in Observation prefix)
        for candidate in [content, content.split("Result: ", 1)[-1] if "Result:" in content else ""]:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    todos = data.get("new_todos", [])
                    if isinstance(todos, list):
                        return todos
            except (json.JSONDecodeError, TypeError):
                continue

        # Fall back to the original tool args
        todos = tool_args.get("todos", [])
        return todos if isinstance(todos, list) else []

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

            case "clear_stream":
                self.app.post_message(ClearStream())

            case "plan":
                self.app.post_message(PlanCreated(content=event.content))

            case "step":
                self.app.post_message(StepStarted(step_info=event.step_info or ""))

            case "tool_call":
                tool_name = event.tool_name or ""
                tool_call_id = event.tool_call_id
                tool_args = event.tool_args or {}
                self._tool_args_queue.append((tool_call_id, tool_name, tool_args))

                # Debug: log tool args for edit/write (helps diagnose diff view issues)
                self._debug_log(
                    f"tool_call '{tool_name}' ({tool_call_id}): queued, "
                    f"keys={list(tool_args.keys())}"
                )
                if tool_name == "write":
                    content = tool_args.get("content", "")
                    self._debug_log(f"  write content: {len(content) if content else 0} chars")
                elif tool_name == "edit":
                    self._debug_log(f"  edit old_string: {bool(tool_args.get('old_string'))}, new_string: {bool(tool_args.get('new_string'))}")

                self.app.post_message(
                    ToolCallStarted(
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        tool_args=tool_args,
                        phase=event.phase,
                    )
                )

            case "tool_result":
                tool_name = event.tool_name or ""
                tool_call_id = event.tool_call_id
                tool_args = {}

                if self._tool_args_queue:
                    for i, (queued_id, queued_name, queued_args) in enumerate(self._tool_args_queue):
                        if tool_call_id is not None and queued_id == tool_call_id:
                            tool_args = queued_args
                            self._tool_args_queue.pop(i)
                            self._debug_log(
                                f"tool_result '{tool_name}' ({tool_call_id}): "
                                f"matched by id, keys={list(tool_args.keys())}"
                            )
                            break
                    else:
                        for i, (_, queued_name, queued_args) in enumerate(self._tool_args_queue):
                            if queued_name == tool_name:
                                tool_args = queued_args
                                self._tool_args_queue.pop(i)
                                self._debug_log(
                                    f"tool_result '{tool_name}' ({tool_call_id}): "
                                    f"matched by name, keys={list(tool_args.keys())}"
                                )
                                break
                        else:
                            popped_id, popped_name, tool_args = self._tool_args_queue.pop(0)
                            self._debug_log(
                                f"tool_result '{tool_name}' ({tool_call_id}): no match, "
                                f"used FIFO (got '{popped_name}' / {popped_id}), "
                                f"keys={list(tool_args.keys())}"
                            )
                else:
                    self._debug_log(
                        f"tool_result '{tool_name}' ({tool_call_id}): queue was EMPTY!"
                    )

                # Extract diff info for edit/write tools
                old_string = None
                new_string = None
                file_path = None
                mutation_preview = build_file_mutation_preview_dict(
                    tool_name,
                    tool_args=tool_args,
                    metadata=event.tool_metadata,
                )

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
                        self._debug_log("  edit: tool_args was empty!")
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
                        self._debug_log("  write: tool_args was empty!")

                self.app.post_message(
                    ToolCallCompleted(
                        tool_name=tool_name,
                        content=event.content,
                        is_error=event.is_error,
                        phase=event.phase,
                        tool_call_id=tool_call_id,
                        metadata=event.tool_metadata,
                        old_string=old_string,
                        new_string=new_string,
                        file_path=file_path,
                        mutation_preview=mutation_preview,
                    )
                )

                # Update the todo list widget when TodoWrite succeeds
                if tool_name == "TodoWrite" and not event.is_error:
                    metadata_todos = (event.tool_metadata or {}).get("new_todos", [])
                    new_todos = (
                        metadata_todos
                        if isinstance(metadata_todos, list) and metadata_todos
                        else self._extract_todos(event.content, tool_args)
                    )
                    if new_todos:
                        self.app.post_message(TodoListUpdated(todos=new_todos))

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

            case "todo_update":
                if event.todo_items is not None:
                    self.app.post_message(TodoListUpdated(todos=event.todo_items))

            case "confirmation":
                # Confirmation is handled via async callback, but we can post a message
                # for UI updates if needed
                self.app.post_message(
                    ConfirmationRequired(
                        tool_name=event.tool_name or "",
                        confirm_message=event.confirm_message or "",
                        details=event.confirm_details or "",
                        preview=event.confirm_preview,
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

            case "dod_status":
                self.app.post_message(
                    DefinitionOfDoneUpdated(
                        content=event.content,
                        dod_status=event.dod_status or "",
                        pending_items_count=event.pending_items_count or 0,
                        last_verification_result=event.last_verification_result,
                        verification_attempt=_definition_of_done_verification_attempt(
                            event.definition_of_done
                        ),
                    )
                )

            case "workflow_mode":
                self.app.post_message(
                    WorkflowModeChanged(
                        workflow_mode=event.workflow_mode or "",
                        content=event.content,
                    )
                )

            case "turn_phase":
                self.app.post_message(
                    TurnPhaseChanged(
                        turn_phase=event.turn_phase or "",
                        content=event.content,
                    )
                )

            case "artifact":
                self.app.post_message(
                    ArtifactCreated(
                        content=event.content,
                        artifact_kind=event.artifact_kind or "",
                        artifact_path=event.artifact_path or "",
                    )
                )


def _definition_of_done_verification_attempt(dod) -> str | None:
    """Render one compact verification-attempt label from DoD state."""

    if dod is None:
        return None
    active_number = getattr(dod, "active_verification_attempt_number", None)
    if active_number is None:
        return None
    if getattr(dod, "last_verification_result", None) == "stale" and active_number > 1:
        return f"attempt {active_number - 1} -> attempt {active_number}"
    return f"attempt {active_number}"
