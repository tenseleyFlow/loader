"""Turn-phase tracking for Loader runtime execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum

from .context import RuntimeContext
from .events import AgentEvent
from .tracing import RuntimeTracer

EventSink = Callable[[AgentEvent], Awaitable[None]]


class TurnPhase(StrEnum):
    """Named phases of one runtime turn."""

    PREPARE = "prepare"
    ASSISTANT = "assistant"
    REPAIR = "repair"
    TOOLS = "tools"
    CRITIQUE = "critique"
    COMPLETION = "completion"
    FINALIZE = "finalize"


class TurnPhaseTracker:
    """Persist and emit turn-phase transitions."""

    def __init__(self, context: RuntimeContext, tracer: RuntimeTracer) -> None:
        self.context = context
        self.tracer = tracer
        self.current_phase: str | None = None

    async def enter(
        self,
        phase: TurnPhase,
        emit: EventSink,
        *,
        detail: str | None = None,
    ) -> None:
        """Move the runtime into a named phase and emit the transition."""

        if phase.value == self.current_phase:
            return

        self.current_phase = phase.value
        self.context.session.update_runtime_state(active_turn_phase=phase.value)
        self.tracer.record("turn.phase_changed", phase=phase.value, detail=detail)
        await emit(
            AgentEvent(
                type="turn_phase",
                content=detail or f"Phase: {phase.value}",
                turn_phase=phase.value,
            )
        )

    def clear(self) -> None:
        """Clear the persisted active phase when the turn finishes."""

        self.current_phase = None
        self.context.session.update_runtime_state(active_turn_phase=None)
