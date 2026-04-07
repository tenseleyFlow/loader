"""Turn-phase tracking for Loader runtime execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

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


class TurnTransitionKind(StrEnum):
    """Classification for why one turn-state transition occurred."""

    NORMAL = "normal"
    RETRY = "retry"
    REROUTE = "reroute"
    RECOVERY = "recovery"
    TERMINAL = "terminal"


@dataclass(slots=True)
class TurnTransition:
    """One validated turn-state transition."""

    from_phase: str | None
    to_phase: str
    reason_code: str
    reason_summary: str
    kind: TurnTransitionKind

    @property
    def summary(self) -> str:
        source = self.from_phase or "start"
        return (
            f"{source} -> {self.to_phase} "
            f"[{self.kind.value}] {self.reason_summary}"
        )


class TurnStateMachine:
    """Validate allowed turn-state transitions."""

    _ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
        None: {TurnPhase.PREPARE.value},
        TurnPhase.PREPARE.value: {
            TurnPhase.ASSISTANT.value,
            TurnPhase.FINALIZE.value,
        },
        TurnPhase.ASSISTANT.value: {
            TurnPhase.REPAIR.value,
            TurnPhase.TOOLS.value,
            TurnPhase.CRITIQUE.value,
            TurnPhase.COMPLETION.value,
            TurnPhase.FINALIZE.value,
        },
        TurnPhase.REPAIR.value: {
            TurnPhase.ASSISTANT.value,
            TurnPhase.TOOLS.value,
            TurnPhase.COMPLETION.value,
            TurnPhase.FINALIZE.value,
        },
        TurnPhase.TOOLS.value: {
            TurnPhase.ASSISTANT.value,
            TurnPhase.CRITIQUE.value,
            TurnPhase.COMPLETION.value,
            TurnPhase.FINALIZE.value,
        },
        TurnPhase.CRITIQUE.value: {
            TurnPhase.ASSISTANT.value,
            TurnPhase.COMPLETION.value,
            TurnPhase.FINALIZE.value,
        },
        TurnPhase.COMPLETION.value: {
            TurnPhase.ASSISTANT.value,
            TurnPhase.FINALIZE.value,
        },
        TurnPhase.FINALIZE.value: set(),
    }

    def __init__(self) -> None:
        self.current_phase: str | None = None
        self.last_transition: TurnTransition | None = None

    def transition(
        self,
        phase: TurnPhase,
        *,
        reason_code: str,
        reason_summary: str,
        kind: TurnTransitionKind = TurnTransitionKind.NORMAL,
    ) -> TurnTransition | None:
        """Validate and record a transition to the target phase."""

        if phase.value == self.current_phase:
            return None

        allowed = self._ALLOWED_TRANSITIONS.get(self.current_phase, set())
        if phase.value not in allowed:
            raise ValueError(
                "Invalid turn-state transition: "
                f"{self.current_phase or 'start'} -> {phase.value}"
            )

        transition = TurnTransition(
            from_phase=self.current_phase,
            to_phase=phase.value,
            reason_code=reason_code,
            reason_summary=reason_summary,
            kind=kind,
        )
        self.current_phase = phase.value
        self.last_transition = transition
        return transition

    def clear(self) -> None:
        """Reset the active phase after a turn completes."""

        self.current_phase = None


class TurnPhaseTracker:
    """Persist and emit turn-phase transitions."""

    def __init__(self, agent, tracer: RuntimeTracer) -> None:
        self.agent = agent
        self.tracer = tracer
        self.state_machine = TurnStateMachine()

    async def enter(
        self,
        phase: TurnPhase,
        emit: EventSink,
        *,
        detail: str | None = None,
        reason_code: str | None = None,
        kind: TurnTransitionKind = TurnTransitionKind.NORMAL,
    ) -> None:
        """Move the runtime into a named phase and emit the transition."""

        summary = detail or f"Phase: {phase.value}"
        transition = self.state_machine.transition(
            phase,
            reason_code=reason_code or phase.value,
            reason_summary=summary,
            kind=kind,
        )
        if transition is None:
            return

        self.agent.session.update_runtime_state(
            active_turn_phase=phase.value,
            last_turn_transition_summary=transition.summary,
            last_turn_transition_kind=transition.kind.value,
            last_turn_transition_reason_code=transition.reason_code,
        )
        self.tracer.record(
            "turn.phase_changed",
            phase=phase.value,
            detail=summary,
            from_phase=transition.from_phase,
            transition_kind=transition.kind.value,
            reason_code=transition.reason_code,
        )
        await emit(
            AgentEvent(
                type="turn_phase",
                content=transition.summary,
                turn_phase=phase.value,
                transition_kind=transition.kind.value,
                transition_summary=transition.summary,
                transition_reason_code=transition.reason_code,
            )
        )

    def clear(self) -> None:
        """Clear the persisted active phase when the turn finishes."""

        self.state_machine.clear()
        self.agent.session.update_runtime_state(active_turn_phase=None)
