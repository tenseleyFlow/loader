"""Tests for runtime-owned reasoning helpers and compatibility exports."""

from __future__ import annotations

import loader.agent.reasoning as agent_reasoning
from loader.runtime.action_reasoning import parse_confidence
from loader.runtime.deliberation import (
    DECOMPOSITION_PROMPT,
    SELF_CRITIQUE_PROMPT,
    parse_decomposition,
    parse_self_critique,
    should_decompose,
)
from loader.runtime.task_classification import is_conversational
from loader.runtime.task_completion import (
    COMPLETION_CHECK_PROMPT,
    parse_completion_check,
)


def test_parse_decomposition_falls_back_to_single_subtask_on_invalid_json() -> None:
    decomposition = parse_decomposition("not valid json", "Refactor auth and add tests")

    assert decomposition.original_task == "Refactor auth and add tests"
    assert len(decomposition.subtasks) == 1
    assert decomposition.subtasks[0].description == "Refactor auth and add tests"
    assert decomposition.subtasks[0].verification == "Check completion"


def test_parse_self_critique_reads_revision_signal() -> None:
    critique = parse_self_critique(
        (
            '{"issues":["Missing edge case"],'
            '"suggestions":["Handle empty input"],'
            '"should_revise":true}'
        ),
        "Draft response",
    )

    assert critique.original_response == "Draft response"
    assert critique.should_revise is True
    assert critique.issues_found == ["Missing edge case"]
    assert critique.suggestions == ["Handle empty input"]


def test_parse_completion_check_builds_continuation_prompt() -> None:
    completion = parse_completion_check(
        (
            '{"is_complete": false,'
            '"accomplished":["Created the file"],'
            '"remaining":["Run the tests"],'
            '"next_steps":["Run pytest -q"]}'
        ),
        "Create the file and verify it works",
    )

    assert completion.is_complete is False
    assert completion.remaining == ["Run the tests"]
    assert "Run pytest -q" in completion.continuation_prompt


def test_agent_reasoning_reexports_runtime_helpers() -> None:
    assert agent_reasoning.DECOMPOSITION_PROMPT == DECOMPOSITION_PROMPT
    assert agent_reasoning.SELF_CRITIQUE_PROMPT == SELF_CRITIQUE_PROMPT
    assert agent_reasoning.COMPLETION_CHECK_PROMPT == COMPLETION_CHECK_PROMPT
    assert agent_reasoning.parse_decomposition is parse_decomposition
    assert agent_reasoning.parse_self_critique is parse_self_critique
    assert agent_reasoning.should_decompose is should_decompose
    assert agent_reasoning.parse_completion_check is parse_completion_check
    assert agent_reasoning.parse_confidence is parse_confidence
    assert agent_reasoning.is_conversational is is_conversational
