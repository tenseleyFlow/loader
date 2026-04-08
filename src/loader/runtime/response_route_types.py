"""Shared typed surfaces for assistant-response routing."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .rollback import RollbackPlan

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


class ResponseRouteAction(StrEnum):
    """What should happen after routing one assistant response."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    FINALIZE = "finalize"


@dataclass(slots=True)
class ResponseRouteDecision:
    """Structured response-routing outcome for one assistant cycle."""

    action: ResponseRouteAction
    continuation_count: int
    consecutive_errors: int
    new_actions_taken: list[str] = field(default_factory=list)
    finalize_reason_code: str | None = None
    finalize_reason_summary: str | None = None


@dataclass(slots=True)
class ResponseRouteContext:
    """Loop state needed to route a classified assistant response."""

    task: str
    effective_task: str
    iterations: int
    max_iterations: int
    actions_taken: list[str]
    continuation_count: int
    consecutive_errors: int
    dod: DefinitionOfDone
    summary: TurnSummary
    executor: ToolExecutor
    rollback_plan: RollbackPlan | None
