"""Tests for no-tool text completion orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.phases import TurnPhase
from loader.runtime.turn_completion import TurnCompletionAction
from tests.helpers.runtime_harness import ScriptedBackend


def non_streaming_config() -> AgentConfig:
    """Shared config for direct turn-completion tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=8)


@pytest.mark.asyncio
async def test_turn_completion_requests_continuation_for_premature_text_response(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    events = []

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task="Fix the README heading.",
        emit=capture,
        requested_mode="execute",
        original_task=None,
        on_user_question=None,
    )
    await runtime.phase_tracker.enter(
        TurnPhase.ASSISTANT,
        capture,
        detail="Requesting assistant response",
        reason_code="request_assistant_response",
    )

    decision = await runtime.turn_completion.handle_text_response(
        content="I looked into it.",
        response_content="I looked into it.",
        task=prepared.task,
        effective_task=prepared.effective_task,
        iterations=1,
        max_iterations=agent.config.max_iterations,
        actions_taken=[],
        continuation_count=0,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
    )

    assert decision.action == TurnCompletionAction.CONTINUE
    assert decision.continuation_count == 1
    assert agent.session.messages[-1].role.value == "user"
    assert "If there's more to do, continue" in agent.session.messages[-1].content
    assert any(event.type == "completion_check" for event in events)


@pytest.mark.asyncio
async def test_turn_completion_marks_non_mutating_response_done(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    events = []

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task="Explain Loader's clarify loop.",
        emit=capture,
        requested_mode="execute",
        original_task=None,
        on_user_question=None,
    )
    await runtime.phase_tracker.enter(
        TurnPhase.ASSISTANT,
        capture,
        detail="Requesting assistant response",
        reason_code="request_assistant_response",
    )

    decision = await runtime.turn_completion.handle_text_response(
        content="Loader uses a bounded clarify loop before execution.",
        response_content="Loader uses a bounded clarify loop before execution.",
        task=prepared.task,
        effective_task=prepared.effective_task,
        iterations=1,
        max_iterations=agent.config.max_iterations,
        actions_taken=[],
        continuation_count=0,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
    )

    assert decision.action == TurnCompletionAction.COMPLETE
    assert prepared.summary.final_response == (
        "Loader uses a bounded clarify loop before execution."
    )
    assert prepared.definition_of_done.status == "done"
    assert prepared.definition_of_done.last_verification_result == "skipped"
    assert any(event.type == "response" for event in events)
    assert any(
        event.type == "dod_status" and event.dod_status == "done"
        for event in events
    )
