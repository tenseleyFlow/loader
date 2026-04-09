"""Shared runtime bootstrap helpers for typed runtime contexts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..context.project import ProjectContext
from ..llm.base import LLMBackend
from ..tools.base import ToolRegistry
from .capabilities import CapabilityProfile
from .context import (
    RuntimeConfigProtocol,
    RuntimeContext,
    RuntimeSafeguardsProtocol,
)
from .events import TurnSummary
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
    current_task: str | None
    last_turn_summary: TurnSummary | None
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


@dataclass(slots=True)
class RuntimeBootstrapView:
    """Explicit runtime-facing bootstrap view over public shell state."""

    project_root: Path
    backend: LLMBackend
    registry: ToolRegistry
    session: ConversationSession
    config: RuntimeConfigProtocol
    project_context: ProjectContext | None
    permission_policy: PermissionPolicy
    permission_config_status: PermissionConfigStatus
    safeguards: RuntimeSafeguardsProtocol
    _get_capability_profile: Callable[[], CapabilityProfile]
    _get_workflow_mode: Callable[[], str]
    _set_workflow_mode: Callable[[str], None]
    _get_current_task: Callable[[], str | None]
    _set_current_task: Callable[[str | None], None]
    _get_last_turn_summary: Callable[[], TurnSummary | None]
    _set_last_turn_summary: Callable[[TurnSummary | None], None]
    _get_prompt_format: Callable[[], str | None]
    _get_prompt_sections: Callable[[], list[str]]
    _queue_steering_message: Callable[[str], None]
    _drain_steering_messages: Callable[[], list[str]]
    _refresh_capability_profile: Callable[[], None]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def capability_profile(self) -> CapabilityProfile:
        """Return the live capability profile from the public shell."""

        return self._get_capability_profile()

    @property
    def workflow_mode(self) -> str:
        """Return the active workflow mode from the public shell."""

        return self._get_workflow_mode()

    @property
    def current_task(self) -> str | None:
        """Return the current top-level task from the public shell."""

        return self._get_current_task()

    @current_task.setter
    def current_task(self, value: str | None) -> None:
        self._set_current_task(value)

    @property
    def last_turn_summary(self) -> TurnSummary | None:
        """Return the latest turn summary from the public shell."""

        return self._get_last_turn_summary()

    @last_turn_summary.setter
    def last_turn_summary(self, value: TurnSummary | None) -> None:
        self._set_last_turn_summary(value)

    @property
    def prompt_format(self) -> str | None:
        """Return the live prompt-format metadata from the public shell."""

        return self._get_prompt_format()

    @property
    def prompt_sections(self) -> list[str]:
        """Return the live prompt-section metadata from the public shell."""

        return list(self._get_prompt_sections())

    def set_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode via the public shell callback."""

        self._set_workflow_mode(workflow_mode)

    def queue_steering_message(self, message: str) -> None:
        """Queue one steering message through the public shell callback."""

        self._queue_steering_message(message)

    def drain_steering_messages(self) -> list[str]:
        """Drain steering messages through the public shell callback."""

        return self._drain_steering_messages()

    def refresh_capability_profile(self) -> None:
        """Refresh capabilities through the public shell callback."""

        self._refresh_capability_profile()


def build_runtime_bootstrap_source(source: RuntimeBootstrapSource | Any) -> RuntimeBootstrapView:
    """Coerce a public shell object into the explicit runtime bootstrap view."""

    if isinstance(source, RuntimeBootstrapView):
        return source

    return RuntimeBootstrapView(
        project_root=source.project_root,
        backend=source.backend,
        registry=source.registry,
        session=source.session,
        config=source.config,
        project_context=source.project_context,
        permission_policy=source.permission_policy,
        permission_config_status=source.permission_config_status,
        safeguards=source.safeguards,
        _get_capability_profile=lambda: source.capability_profile,
        _get_workflow_mode=lambda: source.workflow_mode,
        _set_workflow_mode=source.set_workflow_mode,
        _get_current_task=lambda: source.current_task,
        _set_current_task=lambda value: setattr(source, "current_task", value),
        _get_last_turn_summary=lambda: source.last_turn_summary,
        _set_last_turn_summary=lambda value: setattr(source, "last_turn_summary", value),
        _get_prompt_format=lambda: source.prompt_format,
        _get_prompt_sections=lambda: list(source.prompt_sections),
        _queue_steering_message=source.queue_steering_message,
        _drain_steering_messages=source.drain_steering_messages,
        _refresh_capability_profile=source.refresh_capability_profile,
        metadata={"owner_type": type(source).__name__},
    )


def sync_runtime_context(
    context: RuntimeContext,
    source: RuntimeBootstrapSource,
) -> None:
    """Synchronize mutable runtime context state from the bootstrap source."""

    source = build_runtime_bootstrap_source(source)

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

    source = build_runtime_bootstrap_source(source)
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
