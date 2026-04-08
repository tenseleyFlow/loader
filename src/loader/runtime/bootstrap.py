"""Shared runtime bootstrap helpers for typed runtime contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..context.project import ProjectContext
from ..llm.base import LLMBackend
from ..tools.base import ToolRegistry
from .capabilities import CapabilityProfile
from .context import (
    RuntimeConfigProtocol,
    RuntimeContext,
    RuntimeSafeguardsProtocol,
)
from .permissions import PermissionConfigStatus, PermissionPolicy
from .reasoning_service import RuntimeReasoningService
from .session import ConversationSession


class RuntimeBootstrapSource(Protocol):
    """Typed source object that can bootstrap one runtime context."""

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
    prompt_format: str | None
    prompt_sections: list[str]
    safeguards: RuntimeSafeguardsProtocol

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode."""

    def queue_steering_message(self, message: str) -> None:
        """Queue one steering message for the runtime."""

    def drain_steering_messages(self) -> list[str]:
        """Drain queued steering messages."""

    def refresh_capability_profile(self) -> None:
        """Refresh the active capability profile."""


def sync_runtime_context(
    context: RuntimeContext,
    source: RuntimeBootstrapSource,
) -> None:
    """Synchronize mutable runtime context state from the bootstrap source."""

    context.backend = source.backend
    context.registry = source.registry
    context.session = source.session
    context.config = source.config
    context.capability_profile = source.capability_profile
    context.project_context = source.project_context
    context.permission_policy = source.permission_policy
    context.permission_config_status = source.permission_config_status
    context.workflow_mode = source.workflow_mode
    context.prompt_format = source.prompt_format
    context.prompt_sections = list(source.prompt_sections)


def build_runtime_context(source: RuntimeBootstrapSource) -> RuntimeContext:
    """Build a typed runtime context from the shared bootstrap contract."""

    context: RuntimeContext | None = None

    def _set_workflow_mode(workflow_mode: str) -> None:
        source.set_workflow_mode(workflow_mode)
        if context is not None:
            sync_runtime_context(context, source)

    def _refresh_capability_profile() -> None:
        source.refresh_capability_profile()
        if context is not None:
            context.capability_profile = source.capability_profile

    context = RuntimeContext(
        project_root=source.project_root,
        backend=source.backend,
        registry=source.registry,
        session=source.session,
        config=source.config,
        capability_profile=source.capability_profile,
        project_context=source.project_context,
        permission_policy=source.permission_policy,
        permission_config_status=source.permission_config_status,
        workflow_mode=source.workflow_mode,
        safeguards=source.safeguards,
        reasoning=RuntimeReasoningService(source.backend, source.config),
        prompt_format=source.prompt_format,
        prompt_sections=list(source.prompt_sections),
        set_workflow_mode_callback=_set_workflow_mode,
        drain_steering_messages_callback=source.drain_steering_messages,
        queue_steering_message_callback=source.queue_steering_message,
        refresh_capability_profile_callback=_refresh_capability_profile,
    )
    return context
