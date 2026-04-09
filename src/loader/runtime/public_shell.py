"""Runtime-owned helpers for public shell prompt and session lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..context.project import ProjectContext
from ..llm.base import Message, Role
from ..tools.base import ToolRegistry
from .capabilities import CapabilityProfile, resolve_backend_capability_profile
from .dod import DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .launcher import build_runtime_launcher
from .permissions import PermissionConfigStatus, PermissionMode, PermissionPolicy
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


@dataclass(slots=True)
class CapabilityRefresh:
    """Result of recomputing the active capability profile."""

    capability_profile: CapabilityProfile
    prompt_reset_required: bool


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


class RuntimeShellConfigProtocol(Protocol):
    """Typed view of shell config used for session lifecycle helpers."""

    force_react: bool
    session_rotate_after_bytes: int
    session_auto_compaction_input_tokens_threshold: int
    session_compaction_keep_last_messages: int


class RuntimeShellOwner(Protocol):
    """Typed public-shell owner for session lifecycle helpers."""

    project_root: Path
    backend: Any
    registry: ToolRegistry
    session: ConversationSession
    messages: list[Message]
    config: RuntimeShellConfigProtocol
    project_context: ProjectContext | None
    permission_policy: PermissionPolicy
    permission_config_status: PermissionConfigStatus
    capability_profile: CapabilityProfile
    workflow_mode: str
    prompt_format: str | None
    prompt_sections: list[str]
    current_task: str | None
    last_turn_summary: TurnSummary | None
    steering: SteeringMailbox
    safeguards: Any
    _system_message: Message | None
    _use_react: bool | None

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode."""

    def queue_steering_message(self, message: str) -> None:
        """Queue one steering message for the runtime."""

    def drain_steering_messages(self) -> list[str]:
        """Drain queued steering messages."""

    def refresh_capability_profile(self) -> None:
        """Refresh the active capability profile."""

    def _get_system_message(self) -> Message:
        """Build the active system message."""

    def _get_few_shot_examples(self) -> list[Message]:
        """Build the active few-shot examples."""


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


def apply_runtime_session_install(
    owner: RuntimeShellOwner,
    install: RuntimeSessionInstall,
) -> None:
    """Apply one restored runtime session onto the public shell owner."""

    owner.steering.clear()
    owner.session = install.session
    owner.messages = install.restored.messages
    owner.current_task = install.restored.current_task
    owner.set_workflow_mode(install.restored.workflow_mode)
    owner.permission_policy.active_mode = PermissionMode.from_str(
        install.restored.permission_mode
    )
    owner.prompt_format = install.restored.prompt_format
    owner.prompt_sections = list(install.restored.prompt_sections)
    owner.last_turn_summary = install.restored.last_turn_summary
    owner._system_message = None


def build_fresh_runtime_session_install(
    owner: RuntimeShellOwner,
    *,
    messages: list[Message] | None = None,
    workflow_mode: str | None = None,
) -> RuntimeSessionInstall:
    """Build a fresh runtime session install from the current public shell."""

    return create_runtime_session_install(
        project_root=owner.project_root,
        messages=messages,
        permission_policy=owner.permission_policy,
        permission_config_status=owner.permission_config_status,
        prompt_format=owner.prompt_format,
        prompt_sections=list(owner.prompt_sections),
        workflow_mode=workflow_mode or owner.workflow_mode,
        rotate_after_bytes=owner.config.session_rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            owner.config.session_auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=owner.config.session_compaction_keep_last_messages,
        system_message_factory=owner._get_system_message,
        few_shot_factory=owner._get_few_shot_examples,
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


def resume_runtime_shell_session(
    owner: RuntimeShellOwner,
    *,
    session_id: str | None = None,
) -> bool:
    """Resume the latest or named persisted session onto the public shell."""

    loaded = load_runtime_session_install(
        project_root=owner.project_root,
        system_message_factory=owner._get_system_message,
        few_shot_factory=owner._get_few_shot_examples,
        session_id=session_id,
        rotate_after_bytes=owner.config.session_rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            owner.config.session_auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=owner.config.session_compaction_keep_last_messages,
    )
    if loaded is None:
        return False
    apply_runtime_session_install(owner, loaded)
    return True


def clear_runtime_shell_history(owner: RuntimeShellOwner) -> None:
    """Reset the public shell onto a fresh runtime session."""

    owner.messages = []
    owner.prompt_format = None
    owner.prompt_sections = []
    owner.current_task = None
    owner.last_turn_summary = None
    owner.set_workflow_mode("execute")
    apply_runtime_session_install(
        owner,
        build_fresh_runtime_session_install(
            owner,
            messages=owner.messages,
            workflow_mode="execute",
        ),
    )
    owner._system_message = None
    owner.safeguards.reset()


def resolve_runtime_shell_use_react(owner: RuntimeShellOwner) -> bool:
    """Resolve the active prompt/tool format for the public shell."""

    if owner._use_react is not None:
        return owner._use_react

    if owner.config.force_react:
        owner._use_react = True
        return True

    owner._use_react = not owner.capability_profile.supports_native_tools
    return owner._use_react


def set_runtime_shell_workflow_mode(
    owner: RuntimeShellOwner,
    workflow_mode: str,
) -> None:
    """Update workflow mode and invalidate prompt state when it changes."""

    if workflow_mode == owner.workflow_mode:
        return
    owner.workflow_mode = workflow_mode
    owner._system_message = None


def get_runtime_shell_system_message(owner: RuntimeShellOwner) -> Message:
    """Build or reuse the cached runtime system message for the public shell."""

    if owner._system_message is None:
        prompt_state = build_runtime_system_message(
            registry=owner.registry,
            use_react=resolve_runtime_shell_use_react(owner),
            project_context=owner.project_context,
            workflow_mode=owner.workflow_mode,
            permission_mode=owner.permission_policy.active_mode.as_str(),
            cwd=owner.project_root,
            current_task=owner.current_task,
            session=owner.session,
        )
        owner.prompt_format = prompt_state.prompt_format
        owner.prompt_sections = list(prompt_state.prompt_sections)
        owner._system_message = prompt_state.system_message
    return owner._system_message


def get_runtime_shell_few_shot_examples(owner: RuntimeShellOwner) -> list[Message]:
    """Return few-shot examples for the owner's active shell tool format."""

    return build_runtime_few_shot_examples(
        use_react=resolve_runtime_shell_use_react(owner)
    )


def build_event_emitter(
    on_event: Callable[[AgentEvent], None] | Callable[[AgentEvent], Awaitable[None]] | None,
) -> Callable[[AgentEvent], Awaitable[None]]:
    """Normalize public-shell event callbacks into one async emitter."""

    async def emit(event: AgentEvent) -> None:
        if on_event is None:
            return
        result = on_event(event)
        if inspect.iscoroutine(result):
            await result

    return emit


async def run_runtime_shell(
    owner: RuntimeShellOwner,
    user_message: str,
    *,
    on_event: Callable[[AgentEvent], None]
    | Callable[[AgentEvent], Awaitable[None]]
    | None = None,
    on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None = None,
    on_user_question: Callable[[str, list[str] | None], Awaitable[str]] | None = None,
    use_plan: bool | None = None,
) -> str:
    """Run one user message through the runtime-owned public shell entrypoint."""

    emit = build_event_emitter(on_event)
    owner.steering.mark_running()
    try:
        launcher = build_runtime_launcher(owner)
        return await launcher.run_user_message(
            user_message,
            emit,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            use_plan=use_plan,
        )
    finally:
        owner.steering.mark_idle()


async def stream_runtime_shell(
    owner: RuntimeShellOwner,
    user_message: str,
) -> AsyncIterator[AgentEvent]:
    """Yield the streamed event sequence from the runtime-owned public shell."""

    queue: asyncio.Queue[AgentEvent | BaseException | None] = asyncio.Queue()

    async def on_event(event: AgentEvent) -> None:
        await queue.put(event)

    async def run_owner() -> None:
        try:
            await run_runtime_shell(owner, user_message, on_event=on_event)
        except BaseException as exc:  # pragma: no cover - propagated below
            await queue.put(exc)
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_owner())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        await task
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def run_runtime_shell_explore(
    owner: RuntimeShellOwner,
    user_message: str,
    *,
    on_event: Callable[[AgentEvent], None]
    | Callable[[AgentEvent], Awaitable[None]]
    | None = None,
    fresh: bool = False,
) -> str:
    """Run one read-only explore query through the runtime-owned public shell."""

    emit = build_event_emitter(on_event)
    launcher = build_runtime_launcher(owner)
    owner.last_turn_summary = await launcher.run_explore(
        user_message,
        emit,
        fresh=fresh,
    )
    return owner.last_turn_summary.final_response


def refresh_runtime_capability_state(
    *,
    backend,
    current_profile: CapabilityProfile,
) -> CapabilityRefresh:
    """Recompute backend capability state and report whether prompts must reset."""

    refreshed_profile = resolve_backend_capability_profile(backend)
    return CapabilityRefresh(
        capability_profile=refreshed_profile,
        prompt_reset_required=refreshed_profile != current_profile,
    )


def refresh_runtime_shell_capability_profile(
    owner: RuntimeShellOwner,
) -> CapabilityRefresh:
    """Refresh backend capabilities and invalidate prompt caches as needed."""

    refresh = refresh_runtime_capability_state(
        backend=owner.backend,
        current_profile=owner.capability_profile,
    )
    owner.capability_profile = refresh.capability_profile
    if refresh.prompt_reset_required:
        owner._system_message = None
    owner._use_react = None
    return refresh


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
