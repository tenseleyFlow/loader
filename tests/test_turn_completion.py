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
    assert prepared.summary.completion_decision_code == "premature_completion_nudge"
    assert prepared.summary.completion_decision_summary == (
        "requested one continuation because the non-mutating response looked incomplete"
    )
    assert agent.session.last_completion_decision_code == "premature_completion_nudge"
    assert [
        entry.decision_code for entry in prepared.summary.completion_trace
    ] == ["premature_completion_nudge"]
    assert prepared.summary.completion_trace[0].stage == "continuation_check"
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
    assert prepared.summary.completion_decision_code == "non_mutating_response_accepted"
    assert prepared.summary.completion_decision_summary == (
        "accepted the response because no mutating work required verification"
    )
    assert agent.session.last_completion_decision_code == (
        "non_mutating_response_accepted"
    )
    assert [
        entry.decision_code for entry in prepared.summary.completion_trace
    ] == [
        "completion_response_accepted",
        "non_mutating_response_accepted",
    ]
    assert prepared.definition_of_done.status == "done"
    assert prepared.definition_of_done.last_verification_result == "skipped"
    assert any(event.type == "response" for event in events)
    assert any(
        event.type == "dod_status" and event.dod_status == "done"
        for event in events
    )


@pytest.mark.asyncio
async def test_turn_completion_handles_fake_tool_narration_without_reroute(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    config = non_streaming_config()
    config.reasoning.completion_check = False
    agent = Agent(
        backend=backend,
        config=config,
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    events = []

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task="Summarize the current test status.",
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

    narrated = "Used bash tool with command `pytest -q` and everything passed."
    decision = await runtime.turn_completion.handle_text_response(
        content=narrated,
        response_content=narrated,
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
    assert prepared.summary.final_response == narrated
    assert prepared.summary.completion_decision_code == "non_mutating_response_accepted"
    assert prepared.summary.completion_trace[-1].decision_code == (
        "non_mutating_response_accepted"
    )
    assert not any(
        "PRETENDING to use tools" in message.content
        for message in agent.session.messages
    )
    assert any(event.type == "response" and event.content == narrated for event in events)


@pytest.mark.asyncio
async def test_turn_completion_handles_deflection_text_without_repair_prompt(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    config = non_streaming_config()
    config.reasoning.completion_check = False
    agent = Agent(
        backend=backend,
        config=config,
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    events = []

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task="What should I verify next?",
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

    deflection = "You can run pytest -q to verify the current state."
    decision = await runtime.turn_completion.handle_text_response(
        content=deflection,
        response_content=deflection,
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
    assert prepared.summary.final_response == deflection
    assert prepared.summary.completion_decision_code == "non_mutating_response_accepted"
    assert prepared.summary.completion_trace[-1].decision_code == (
        "non_mutating_response_accepted"
    )
    assert not any(
        "Please use your tools to execute the task" in message.content
        for message in agent.session.messages
    )
    assert any(event.type == "response" and event.content == deflection for event in events)


@pytest.mark.asyncio
async def test_turn_completion_skips_self_critique_reroute(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    config = non_streaming_config()
    config.reasoning.completion_check = False
    config.reasoning.self_critique = True
    agent = Agent(
        backend=backend,
        config=config,
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

    detailed = (
        "Loader might begin with a bounded clarify pass, perhaps asking follow-up "
        "questions when the task leaves touchpoints or decision boundaries unclear. "
        "It then shifts into execution once the workflow policy is satisfied."
    )
    decision = await runtime.turn_completion.handle_text_response(
        content=detailed,
        response_content=detailed,
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
    assert prepared.summary.final_response == detailed
    assert prepared.summary.completion_decision_code == "non_mutating_response_accepted"
    assert prepared.summary.completion_trace[-1].decision_code == (
        "non_mutating_response_accepted"
    )
    assert not any("[SELF-CRITIQUE]" in message.content for message in agent.session.messages)
    assert not any(event.type == "critique" for event in events)
