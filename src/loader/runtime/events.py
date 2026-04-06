"""Typed runtime event and summary surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..agent.reasoning import (
    ActionVerification,
    ConfidenceAssessment,
    RollbackAction,
    RollbackPlan,
    SelfCritique,
    Subtask,
    TaskCompletionCheck,
    TaskDecomposition,
)
from ..llm.base import Message
from .tracing import RuntimeTraceEvent


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""

    type: str
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    step_info: str | None = None
    recovery_attempt: int | None = None
    is_stream_end: bool = False
    confirm_message: str | None = None
    confirm_details: str | None = None
    is_error: bool = False

    decomposition: TaskDecomposition | None = None
    subtask: Subtask | None = None
    critique: SelfCritique | None = None
    confidence: ConfidenceAssessment | None = None
    verification: ActionVerification | None = None
    completion_check: TaskCompletionCheck | None = None
    rollback_plan: RollbackPlan | None = None
    rollback_action: RollbackAction | None = None


@dataclass
class TurnSummary:
    """Structured summary for one completed runtime turn."""

    final_response: str
    assistant_messages: list[Message] = field(default_factory=list)
    tool_result_messages: list[Message] = field(default_factory=list)
    iterations: int = 0
    failures: list[str] = field(default_factory=list)
    verification_status: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    trace: list[RuntimeTraceEvent] = field(default_factory=list)
