"""Typed runtime context and service seams for turn execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..context.project import ProjectContext
from ..llm.base import LLMBackend, Message
from ..tools.base import ToolRegistry
from .capabilities import CapabilityProfile
from .permissions import PermissionConfigStatus, PermissionPolicy
from .reasoning_types import ActionVerification, ConfidenceAssessment
from .recovery import RecoveryContext
from .session import ConversationSession
from .steering import SteeringDirective


class ReasoningConfigProtocol(Protocol):
    """Typed view of reasoning-stage settings the runtime consumes."""

    decomposition: bool
    rollback: bool
    show_rollback_plan: bool
    completion_check: bool
    use_quick_completion: bool
    max_continuation_prompts: int
    self_critique: bool
    confidence_scoring: bool
    min_confidence_for_action: int
    verification: bool


class RuntimeConfigProtocol(Protocol):
    """Typed view of top-level agent config used by the runtime."""

    max_iterations: int
    clarify_max_rounds: int
    temperature: float
    max_tokens: int
    force_react: bool
    auto_plan: bool
    auto_recover: bool
    max_recovery_attempts: int
    verification_retry_budget: int
    workflow_mode_override: str | None
    stream: bool
    reasoning: ReasoningConfigProtocol


class CodeFilterProtocol(Protocol):
    """Typed view of the code-block filter used during streaming."""

    def reset(self) -> None:
        """Reset filter state for a new assistant turn."""


class RuntimeSafeguardsProtocol(Protocol):
    """Typed view of runtime safeguards used during migration."""

    action_tracker: Any
    validator: Any
    code_filter: CodeFilterProtocol

    def filter_stream_chunk(self, content: str) -> str:
        """Filter one streamed chunk before it reaches the UI."""

    def filter_complete_content(self, content: str) -> str:
        """Filter a non-streaming assistant response."""

    def should_steer(self) -> bool:
        """Return whether a steering nudge should be queued."""

    def get_steering_message(self) -> str | None:
        """Return the current steering message if one is pending."""

    def record_response(self, content: str) -> None:
        """Record a completed assistant response for safeguard bookkeeping."""


class RuntimeReasoningServiceProtocol(Protocol):
    """Typed action-reasoning surface the runtime can rely on."""

    async def assess_confidence(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: str = "",
    ) -> ConfidenceAssessment:
        """Assess confidence in a planned tool action."""

    async def verify_action(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        """Verify that a tool action produced the desired result."""


@dataclass(slots=True)
class RuntimeContext:
    """Typed state and services shared across runtime helpers."""

    project_root: Path
    backend: LLMBackend
    registry: ToolRegistry
    session: ConversationSession
    config: RuntimeConfigProtocol
    capability_profile: CapabilityProfile
    project_context: ProjectContext | None
    permission_policy: PermissionPolicy
    permission_config_status: PermissionConfigStatus
    workflow_mode: str
    safeguards: RuntimeSafeguardsProtocol
    reasoning: RuntimeReasoningServiceProtocol | None = None
    recovery_context: RecoveryContext | None = None
    prompt_format: str | None = None
    prompt_sections: list[str] = field(default_factory=list)
    set_workflow_mode_callback: Callable[[str], None] | None = None
    drain_steering_messages_callback: Callable[[], list[SteeringDirective]] | None = None
    queue_steering_message_callback: Callable[[str], None] | None = None
    queue_ephemeral_steering_message_callback: Callable[[str], None] | None = None
    refresh_capability_profile_callback: Callable[[], None] | None = None

    @property
    def use_react(self) -> bool:
        """Determine whether the runtime should use ReAct tool prompts."""

        return bool(self.config.force_react or not self.capability_profile.supports_native_tools)

    @property
    def messages(self) -> list[Message]:
        """Expose the current session message list as runtime history."""

        return self.session.messages

    @property
    def active_permission_mode(self) -> str:
        """Return the active permission mode string."""

        return self.permission_policy.active_mode.as_str()

    @property
    def active_permission_rule_counts(self) -> dict[str, int]:
        """Return rule counts for the active permission policy."""

        return self.permission_policy.rule_counts()

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode through the runtime control seam."""

        if self.set_workflow_mode_callback is None:
            self.workflow_mode = workflow_mode
            return
        self.set_workflow_mode_callback(workflow_mode)
        self.workflow_mode = workflow_mode

    def drain_steering_messages(self) -> list[SteeringDirective]:
        """Drain pending steering messages through the runtime control seam."""

        if self.drain_steering_messages_callback is None:
            return []
        return self.drain_steering_messages_callback()

    def queue_steering_message(self, message: str) -> None:
        """Queue a steering message through the runtime control seam."""

        if self.queue_steering_message_callback is None:
            return
        self.queue_steering_message_callback(message)

    def queue_ephemeral_steering_message(self, message: str) -> None:
        """Queue a UI-visible steering message without forcing model persistence."""

        if self.queue_ephemeral_steering_message_callback is not None:
            self.queue_ephemeral_steering_message_callback(message)
            return
        if self.queue_steering_message_callback is not None:
            self.queue_steering_message_callback(message)

    def refresh_capability_profile(self) -> None:
        """Refresh the resolved capability profile through the runtime control seam."""

        if self.refresh_capability_profile_callback is None:
            return
        self.refresh_capability_profile_callback()

    async def assess_confidence(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: str = "",
    ) -> ConfidenceAssessment:
        """Assess confidence using the primary runtime reasoning service."""

        if self.reasoning is None:
            raise RuntimeError("RuntimeContext.reasoning is required for confidence checks")
        return await self.reasoning.assess_confidence(tool_name, tool_args, context)

    async def verify_action(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        """Verify a tool action using the primary runtime reasoning service."""

        if self.reasoning is None:
            raise RuntimeError("RuntimeContext.reasoning is required for verification checks")
        return await self.reasoning.verify_action(tool_name, tool_args, result, expected)
