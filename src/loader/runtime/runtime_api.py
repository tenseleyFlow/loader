"""Runtime-owned public shell contract and owner builder."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Literal, Protocol

from ..context.project import ProjectContext
from ..tools.base import ToolRegistry
from .capabilities import CapabilityProfile
from .events import AgentEvent
from .session import ConversationSession

RuntimeOwnerKind = Literal["runtime", "public-compat"]


class RuntimeShellOwner(Protocol):
    """Narrow public shell contract shared by CLI, UI, and integrations."""

    backend: object
    registry: ToolRegistry
    capability_profile: CapabilityProfile
    safeguards: object
    session: ConversationSession
    project_context: ProjectContext | None
    workflow_mode: str
    active_permission_mode: str
    use_react: bool
    is_running: bool

    def resume_session(self, session_id: str | None = None) -> bool:
        """Resume the latest or named persisted session."""

    def steer(self, message: str) -> bool:
        """Queue one steering message while the owner is running."""

    def refresh_capability_profile(self) -> None:
        """Refresh the active capability profile."""

    async def run(
        self,
        user_message: str,
        on_event: (
            Callable[[AgentEvent], None]
            | Callable[[AgentEvent], Awaitable[None]]
            | None
        ) = None,
        on_confirmation: Callable[[str, str, str], Awaitable[bool]] | None = None,
        on_user_question: Callable[[str, list[str] | None], Awaitable[str]] | None = None,
        use_plan: bool | None = None,
    ) -> str:
        """Run one user message through the shell owner."""

    async def run_streaming(
        self,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """Yield the streamed event sequence for one user message."""

    async def run_explore(
        self,
        user_message: str,
        on_event: (
            Callable[[AgentEvent], None]
            | Callable[[AgentEvent], Awaitable[None]]
            | None
        ) = None,
        *,
        fresh: bool = False,
    ) -> str:
        """Run one read-only explore query through the shell owner."""

    def clear_history(self) -> None:
        """Reset the owner history."""


def build_runtime_shell_owner(
    *,
    backend,
    registry,
    config,
    project_root: Path | str | None = None,
    owner_kind: RuntimeOwnerKind = "runtime",
) -> RuntimeShellOwner:
    """Build the shared shell owner for runtime-first or compatibility paths."""

    if owner_kind == "public-compat":
        from ..agent.loop import Agent

        return Agent(
            backend=backend,
            registry=registry,
            config=config,
            project_root=project_root,
        )

    from .runtime_handle import RuntimeHandle

    return RuntimeHandle(
        backend=backend,
        registry=registry,
        config=config,
        project_root=project_root,
    )
