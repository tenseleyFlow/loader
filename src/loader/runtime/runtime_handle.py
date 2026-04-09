"""Runtime-owned internal execution handle below the public Agent facade."""

from __future__ import annotations

from pathlib import Path

from ..context.project import ProjectContext, detect_project
from ..llm.base import LLMBackend, Message
from ..tools.base import ToolRegistry, create_default_registry
from .capabilities import resolve_backend_capability_profile
from .events import TurnSummary
from .permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from .public_shell import (
    SteeringMailbox,
    build_fresh_runtime_session_install,
    clear_runtime_shell_history,
    get_runtime_shell_few_shot_examples,
    get_runtime_shell_system_message,
    refresh_runtime_shell_capability_profile,
    resolve_runtime_shell_use_react,
    resume_runtime_shell_session,
    set_runtime_shell_workflow_mode,
)
from .safeguards import RuntimeSafeguards
from .workflow import WorkflowMode


class RuntimeHandle:
    """Runtime-first internal owner for launcher and turn-runtime tests."""

    def __init__(
        self,
        *,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        config,
        project_root: Path | str | None = None,
        project_context: ProjectContext | None = None,
        permission_mode: PermissionMode | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.project_root = Path(project_root or ".").expanduser().resolve()
        self.registry = registry or create_default_registry(self.project_root)
        self.registry.configure_workspace_root(self.project_root)
        self.permission_config_status = load_permission_rules(self.project_root)
        if not self.permission_config_status.valid:
            raise ValueError(
                "Invalid permission policy configuration at "
                f"{self.permission_config_status.source_path}: "
                f"{self.permission_config_status.error}"
            )
        active_permission_mode = permission_mode or getattr(
            self.config,
            "permission_mode",
            PermissionMode.WORKSPACE_WRITE,
        )
        self.permission_policy = build_permission_policy(
            active_mode=active_permission_mode,
            workspace_root=self.project_root,
            tool_requirements=self.registry.get_tool_requirements(),
            rules=self.permission_config_status.rules,
        )
        self.workflow_mode = WorkflowMode.EXECUTE.value
        self.messages: list[Message] = []
        self.prompt_format: str | None = None
        self.prompt_sections: list[str] = []
        self._system_message: Message | None = None
        self._use_react: bool | None = None
        self.capability_profile = resolve_backend_capability_profile(self.backend)
        self.last_turn_summary: TurnSummary | None = None
        self.steering = SteeringMailbox()
        self._current_task: str | None = None
        self.safeguards = RuntimeSafeguards()
        if project_context is not None:
            self.project_context = project_context
        elif getattr(self.config, "auto_context", False):
            self.project_context = detect_project(self.project_root)
        else:
            self.project_context = None
        self.session = build_fresh_runtime_session_install(self).session

    @property
    def current_task(self) -> str | None:
        """Expose the current top-level task through the bootstrap contract."""

        return self._current_task

    @current_task.setter
    def current_task(self, value: str | None) -> None:
        self._current_task = value

    @property
    def active_permission_mode(self) -> str:
        """Return the current runtime permission mode."""

        return self.permission_policy.active_mode.as_str()

    @property
    def active_permission_rule_counts(self) -> dict[str, int]:
        """Return rule counts for the active permission policy."""

        return self.permission_policy.rule_counts()

    @property
    def use_react(self) -> bool:
        """Determine whether to use ReAct prompting or native tools."""

        return resolve_runtime_shell_use_react(self)

    def resume_session(self, session_id: str | None = None) -> bool:
        """Resume the latest or named persisted session."""

        return resume_runtime_shell_session(self, session_id=session_id)

    def clear_history(self) -> None:
        """Reset the internal runtime handle onto a fresh session."""

        clear_runtime_shell_history(self)

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode used by the system prompt."""

        set_runtime_shell_workflow_mode(self, workflow_mode)

    def refresh_capability_profile(self) -> None:
        """Refresh the runtime capability profile from the current backend."""

        refresh_runtime_shell_capability_profile(self)

    def queue_steering_message(self, message: str) -> None:
        """Queue one runtime steering message."""

        self.steering.queue(message)

    def drain_steering_messages(self) -> list[str]:
        """Drain queued runtime steering messages."""

        return self.steering.drain()

    def _get_system_message(self) -> Message:
        """Get the system message with current context."""

        return get_runtime_shell_system_message(self)

    def _get_few_shot_examples(self) -> list[Message]:
        """Get few-shot examples demonstrating proper tool use."""

        return get_runtime_shell_few_shot_examples(self)
