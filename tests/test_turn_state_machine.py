"""Tests for the validated runtime turn state machine."""

from __future__ import annotations

import pytest

from loader.runtime.phases import (
    TurnPhase,
    TurnStateMachine,
    TurnTransitionKind,
)


def test_turn_state_machine_accepts_valid_transitions() -> None:
    machine = TurnStateMachine()

    prepare = machine.transition(
        TurnPhase.PREPARE,
        reason_code="prepare_runtime",
        reason_summary="Preparing runtime state",
    )
    assistant = machine.transition(
        TurnPhase.ASSISTANT,
        reason_code="request_assistant_response",
        reason_summary="Requesting assistant response",
    )
    tools = machine.transition(
        TurnPhase.TOOLS,
        reason_code="execute_tool_batch",
        reason_summary="Executing tool batch",
    )
    finalize = machine.transition(
        TurnPhase.FINALIZE,
        reason_code="turn_complete",
        reason_summary="Finalizing completed turn",
        kind=TurnTransitionKind.TERMINAL,
    )

    assert prepare is not None
    assert prepare.from_phase is None
    assert prepare.to_phase == "prepare"
    assert assistant is not None
    assert assistant.from_phase == "prepare"
    assert tools is not None
    assert tools.from_phase == "assistant"
    assert finalize is not None
    assert finalize.kind is TurnTransitionKind.TERMINAL
    assert machine.current_phase == "finalize"
    assert machine.last_transition == finalize
    assert finalize.summary == "tools -> finalize [terminal] Finalizing completed turn"


def test_turn_state_machine_rejects_invalid_transitions() -> None:
    machine = TurnStateMachine()
    machine.transition(
        TurnPhase.PREPARE,
        reason_code="prepare_runtime",
        reason_summary="Preparing runtime state",
    )

    with pytest.raises(ValueError, match="prepare -> tools"):
        machine.transition(
            TurnPhase.TOOLS,
            reason_code="execute_tool_batch",
            reason_summary="Executing tool batch",
        )
