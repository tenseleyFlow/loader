"""Runtime-owned helpers for public shell prompt and session lifecycle."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..context.project import ProjectContext
from ..llm.base import Message, Role
from ..tools.base import ToolRegistry
from .dod import DefinitionOfDoneStore
from .events import TurnSummary
from .permissions import PermissionConfigStatus, PermissionPolicy
from .prompt_history import PromptSnapshot
from .prompting import build_system_prompt_result
from .session import ConversationSession


@dataclass(slots=True)
class RuntimePromptState:
    """Rendered runtime prompt plus the session metadata it updates."""

    system_message: Message
    prompt_format: str
    prompt_sections: list[str]


@dataclass(slots=True)
class RestoredSessionState:
    """Mutable agent-shell state reconstructed from a persisted session."""

    messages: list[Message]
    current_task: str | None
    workflow_mode: str
    permission_mode: str
    prompt_format: str | None
    prompt_sections: list[str]
    last_completion_decision_code: str | None
    last_completion_decision_summary: str | None
    last_turn_summary: TurnSummary | None


@dataclass(slots=True)
class RuntimeSessionInstall:
    """A persisted session plus the shell state restored from it."""

    session: ConversationSession
    restored: RestoredSessionState


class SteeringMailbox:
    """Small public-shell owner for steering and running-state bookkeeping."""

    def __init__(self) -> None:
        self._pending: deque[str] = deque()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Return whether the owning shell is currently running."""

        return self._is_running

    def mark_running(self) -> None:
        """Mark the owning shell as currently running."""

        self._is_running = True

    def mark_idle(self) -> None:
        """Mark the owning shell as currently idle."""

        self._is_running = False

    def steer(self, message: str) -> bool:
        """Queue one steering message only when the owner is running."""

        if not self._is_running:
            return False
        self.queue(message)
        return True

    def queue(self, message: str) -> None:
        """Queue one steering message regardless of running state."""

        self._pending.append(message)

    def drain(self) -> list[str]:
        """Drain all pending steering messages in FIFO order."""

        drained = list(self._pending)
        self._pending.clear()
        return drained

    def clear(self) -> None:
        """Drop queued steering state and mark the owner idle."""

        self._pending.clear()
        self._is_running = False


def create_runtime_session(
    *,
    project_root: Path,
    messages: list[Message] | None,
    permission_policy: PermissionPolicy,
    permission_config_status: PermissionConfigStatus,
    prompt_format: str | None,
    prompt_sections: list[str],
    workflow_mode: str,
    rotate_after_bytes: int,
    auto_compaction_input_tokens_threshold: int,
    compaction_keep_last_messages: int,
    system_message_factory: Callable[[], Message],
    few_shot_factory: Callable[[], list[Message]],
) -> ConversationSession:
    """Create one persisted conversation session from public shell state."""

    return ConversationSession(
        system_message_factory=system_message_factory,
        few_shot_factory=few_shot_factory,
        project_root=project_root,
        messages=messages or [],
        permission_mode=permission_policy.active_mode.as_str(),
        permission_prompting_enabled=permission_policy.prompting_enabled,
        permission_rule_counts=_copy_rule_counts(permission_policy.rule_counts()),
        permission_rules_source=str(permission_config_status.source_path),
        prompt_format=prompt_format,
        prompt_sections=list(prompt_sections),
        workflow_mode=workflow_mode,
        rotate_after_bytes=rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=compaction_keep_last_messages,
    )


def create_runtime_session_install(
    *,
    project_root: Path,
    messages: list[Message] | None,
    permission_policy: PermissionPolicy,
    permission_config_status: PermissionConfigStatus,
    prompt_format: str | None,
    prompt_sections: list[str],
    workflow_mode: str,
    rotate_after_bytes: int,
    auto_compaction_input_tokens_threshold: int,
    compaction_keep_last_messages: int,
    system_message_factory: Callable[[], Message],
    few_shot_factory: Callable[[], list[Message]],
) -> RuntimeSessionInstall:
    """Create a fresh persisted session and its restored shell view."""

    session = create_runtime_session(
        project_root=project_root,
        messages=messages,
        permission_policy=permission_policy,
        permission_config_status=permission_config_status,
        prompt_format=prompt_format,
        prompt_sections=prompt_sections,
        workflow_mode=workflow_mode,
        rotate_after_bytes=rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=compaction_keep_last_messages,
        system_message_factory=system_message_factory,
        few_shot_factory=few_shot_factory,
    )
    return RuntimeSessionInstall(
        session=session,
        restored=restore_runtime_session_state(
            project_root=project_root,
            session=session,
        ),
    )


def restore_runtime_session_state(
    *,
    project_root: Path,
    session: ConversationSession,
) -> RestoredSessionState:
    """Reconstruct public shell state from one persisted runtime session."""

    last_turn_summary: TurnSummary | None = None
    if session.active_dod_path:
        dod_path = Path(session.active_dod_path)
        if dod_path.exists():
            dod = DefinitionOfDoneStore(project_root).load(dod_path)
            last_turn_summary = TurnSummary(
                final_response="",
                definition_of_done=dod,
                workflow_mode=session.workflow_mode,
                workflow_reason_code=session.workflow_reason_code,
                workflow_reason_summary=session.workflow_reason_summary,
                workflow_decision_kind=session.workflow_decision_kind,
                completion_decision_code=session.last_completion_decision_code,
                completion_decision_summary=session.last_completion_decision_summary,
                completion_trace=list(session.completion_trace),
                workflow_timeline=list(session.workflow_timeline),
                session_id=session.session_id,
                cumulative_usage=dict(session.usage_totals),
            )

    return RestoredSessionState(
        messages=session.messages,
        current_task=session.current_task,
        workflow_mode=session.workflow_mode,
        permission_mode=session.permission_mode,
        prompt_format=session.prompt_format,
        prompt_sections=list(session.prompt_sections),
        last_completion_decision_code=session.last_completion_decision_code,
        last_completion_decision_summary=session.last_completion_decision_summary,
        last_turn_summary=last_turn_summary,
    )


def load_runtime_session_install(
    *,
    project_root: Path,
    system_message_factory: Callable[[], Message],
    few_shot_factory: Callable[[], list[Message]],
    session_id: str | None = None,
    rotate_after_bytes: int,
    auto_compaction_input_tokens_threshold: int,
    compaction_keep_last_messages: int,
) -> RuntimeSessionInstall | None:
    """Load the latest or named session together with restored shell state."""

    session = ConversationSession.load(
        project_root=project_root,
        system_message_factory=system_message_factory,
        few_shot_factory=few_shot_factory,
        session_id=session_id,
        rotate_after_bytes=rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=compaction_keep_last_messages,
    )
    if session is None:
        return None
    return RuntimeSessionInstall(
        session=session,
        restored=restore_runtime_session_state(
            project_root=project_root,
            session=session,
        ),
    )


def build_runtime_system_message(
    *,
    registry: ToolRegistry,
    use_react: bool,
    project_context: ProjectContext | None,
    workflow_mode: str,
    permission_mode: str,
    cwd: Path,
    current_task: str | None,
    session: ConversationSession,
) -> RuntimePromptState:
    """Build and persist the active runtime system prompt contract."""

    prompt_result = build_system_prompt_result(
        tools=registry.get_schemas(),
        use_react=use_react,
        project_context=project_context,
        workflow_mode=workflow_mode,
        permission_mode=permission_mode,
        cwd=cwd,
        current_task=current_task,
    )
    prompt_sections = list(prompt_result.dynamic_section_names)
    session.update_runtime_state(
        prompt_format=prompt_result.prompt_format,
        prompt_sections=prompt_sections,
    )
    session.append_prompt_snapshot(
        PromptSnapshot.create(
            workflow_mode=workflow_mode,
            permission_mode=permission_mode,
            current_task=current_task,
            prompt_format=prompt_result.prompt_format,
            prompt_sections=prompt_sections,
            content=prompt_result.content,
        )
    )
    return RuntimePromptState(
        system_message=Message(
            role=Role.SYSTEM,
            content=prompt_result.content,
        ),
        prompt_format=prompt_result.prompt_format,
        prompt_sections=prompt_sections,
    )


def build_runtime_few_shot_examples(*, use_react: bool) -> list[Message]:
    """Return the runtime-owned few-shot examples for the active tool format."""

    if use_react:
        return [
            Message(
                role=Role.USER,
                content="Create a file called hello.py that prints hello",
            ),
            Message(
                role=Role.ASSISTANT,
                content=(
                    '<tool_call>\n'
                    '{"name": "write", "arguments": {"file_path": "hello.py", '
                    '"content": "print(\'hello\')"}}\n'
                    "</tool_call>"
                ),
            ),
            Message(role=Role.TOOL, content="Created hello.py"),
            Message(role=Role.ASSISTANT, content="Done."),
        ]
    return [
        Message(
            role=Role.USER,
            content="Create a file called hello.py that prints hello",
        ),
        Message(
            role=Role.ASSISTANT,
            content='[write: file_path="hello.py", content="print(\'hello\')"]',
        ),
        Message(role=Role.TOOL, content="Created hello.py"),
        Message(role=Role.ASSISTANT, content="Done."),
    ]


def _copy_rule_counts(rule_counts: dict[str, int]) -> dict[str, int]:
    return {
        "allow": int(rule_counts.get("allow", 0)),
        "deny": int(rule_counts.get("deny", 0)),
        "ask": int(rule_counts.get("ask", 0)),
    }
