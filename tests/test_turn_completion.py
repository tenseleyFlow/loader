"""Tests for no-tool text completion orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.dod import VerificationEvidence
from loader.runtime.phases import TurnPhase
from loader.runtime.turn_completion import TurnCompletionAction
from loader.runtime.verification_observations import VerificationObservationStatus
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
    assert [entry.kind for entry in prepared.summary.workflow_timeline[-1:]] == [
        "completion_continue"
    ]
    assert prepared.summary.workflow_timeline[-1].policy_stage == "continuation_check"
    assert prepared.summary.workflow_timeline[-1].policy_outcome == "continue"
    assert agent.session.messages[-1].role.value == "user"
    assert "concrete evidence" in agent.session.messages[-1].content
    assert "Carry out the requested change or command now" in agent.session.messages[-1].content
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
    policy_entries = [
        entry
        for entry in prepared.summary.workflow_timeline
        if entry.kind.startswith("completion_")
    ]
    assert [entry.kind for entry in policy_entries] == [
        "completion_check",
        "completion_complete",
    ]
    assert policy_entries[0].policy_stage == "continuation_check"
    assert policy_entries[-1].policy_stage == "definition_of_done"
    assert [item.summary for item in prepared.summary.completion_trace[-1].evidence_provenance] == [
        "verification was skipped because no mutating work required checks"
    ]
    assert [
        item.status
        for item in prepared.summary.completion_trace[-1].verification_observations
    ] == [VerificationObservationStatus.SKIPPED.value]
    assert [
        item.summary
        for item in prepared.summary.completion_trace[-1].verification_observations
    ] == ["verification was skipped because no mutating work required checks"]
    assert [item.status for item in policy_entries[-1].verification_observations] == [
        VerificationObservationStatus.SKIPPED.value
    ]
    assert prepared.definition_of_done.status == "done"
    assert prepared.definition_of_done.last_verification_result == "skipped"
    assert any(event.type == "response" for event in events)
    assert any(
        event.type == "dod_status" and event.dod_status == "done"
        for event in events
    )


@pytest.mark.asyncio
async def test_turn_completion_blocks_false_completion_without_preserving_it(
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
        task=(
            "Create a multi-file nginx guide under ~/Loader/guides/nginx "
            "with an index and chapter files."
        ),
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

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "# Implementation Plan\n\n"
        "## File Changes\n\n"
        "1. Create main index.html file:\n"
        "   - `index.html`\n\n"
        "2. Create chapter files:\n"
        "   - `chapters/01-getting-started.html`\n"
        "   - `chapters/06-troubleshooting.html`\n"
    )
    chapters_dir = temp_dir / "chapters"
    chapters_dir.mkdir()
    (chapters_dir / "01-getting-started.html").write_text("<h1>Getting Started</h1>\n")
    (temp_dir / "index.html").write_text("<h1>NGINX Guide</h1>\n")

    prepared.definition_of_done.implementation_plan = str(implementation_plan)
    prepared.definition_of_done.mutating_actions.append("write")
    prepared.definition_of_done.touched_files.extend(
        [
            str(temp_dir / "index.html"),
            str(chapters_dir / "01-getting-started.html"),
        ]
    )

    queued_messages: list[str] = []
    runtime.context.queue_steering_message_callback = queued_messages.append

    completion_claim = (
        "I've successfully completed the NGINX guide with all planned files "
        "and verified everything is done."
    )
    decision = await runtime.turn_completion.handle_text_response(
        content=completion_claim,
        response_content=completion_claim,
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
    assert prepared.summary.assistant_messages == []
    assert not any(
        message.role.value == "assistant" and message.content == completion_claim
        for message in agent.session.messages
    )
    assert agent.session.messages[-1].role.value == "user"
    assert agent.session.messages[-1].content.startswith(
        "[PLANNED ARTIFACTS STILL MISSING]"
    )
    assert "`06-troubleshooting.html`" in agent.session.messages[-1].content
    assert queued_messages
    assert "06-troubleshooting.html" in queued_messages[-1]
    assert "Do not summarize, mark completion, or write bookkeeping notes yet" in queued_messages[-1]
    assert not any(event.type == "response" for event in events)


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


@pytest.mark.asyncio
async def test_turn_completion_finalizes_when_follow_through_budget_is_exhausted(
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
        continuation_count=agent.config.reasoning.max_continuation_prompts,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
    )

    assert decision.action == TurnCompletionAction.FINALIZE
    assert decision.finalize_reason_code == "continuation_budget_exhausted"
    assert prepared.summary.final_response.startswith(
        "I stopped because I still could not show enough evidence"
    )
    assert prepared.summary.completion_decision_code == "continuation_budget_exhausted"
    assert prepared.summary.failures == [
        "missing follow-through evidence after continuation budget exhaustion"
    ]
    assert prepared.summary.completion_trace[-1].outcome == "finalize"
    assert prepared.summary.completion_trace[-1].decision_code == (
        "continuation_budget_exhausted"
    )
    assert prepared.summary.completion_trace[-1].evidence_summary == [
        "showing the requested work was actually carried out"
    ]
    assert [item.status for item in prepared.summary.completion_trace[-1].evidence_provenance] == [
        "missing"
    ]
    assert prepared.summary.workflow_timeline[-1].kind == "completion_finalize"
    assert prepared.summary.workflow_timeline[-1].evidence_summary == [
        "showing the requested work was actually carried out"
    ]
    assert [event.type for event in events[-3:]] == [
        "completion_check",
        "error",
        "response",
    ]


@pytest.mark.asyncio
async def test_turn_completion_uses_observed_verification_for_budget_exhaustion(
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
        task="Run pytest -q and make sure it works.",
        emit=capture,
        requested_mode="execute",
        original_task=None,
        on_user_question=None,
    )
    prepared.definition_of_done.verification_commands = ["pytest -q"]
    prepared.definition_of_done.evidence = [
        VerificationEvidence(
            command="pytest -q",
            passed=False,
            stderr="1 failed",
            kind="test",
        )
    ]
    prepared.definition_of_done.last_verification_result = "failed"
    await runtime.phase_tracker.enter(
        TurnPhase.ASSISTANT,
        capture,
        detail="Requesting assistant response",
        reason_code="request_assistant_response",
    )

    decision = await runtime.turn_completion.handle_text_response(
        content="The tests are done.",
        response_content="The tests are done.",
        task=prepared.task,
        effective_task=prepared.effective_task,
        iterations=1,
        max_iterations=agent.config.max_iterations,
        actions_taken=[],
        continuation_count=agent.config.reasoning.max_continuation_prompts,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
    )

    assert decision.action == TurnCompletionAction.FINALIZE
    assert decision.finalize_reason_code == "continuation_budget_exhausted"
    assert prepared.summary.final_response == (
        "I stopped because the continuation budget was exhausted and observed "
        "verification still showed: verification failed for `pytest -q` [1 failed]."
    )
    assert prepared.summary.completion_trace[-1].decision_code == (
        "continuation_budget_exhausted"
    )
    assert [
        item.status
        for item in prepared.summary.completion_trace[-1].verification_observations
    ] == [VerificationObservationStatus.FAILED.value]
    assert [
        item.summary
        for item in prepared.summary.completion_trace[-1].verification_observations
    ] == ["verification failed for `pytest -q`"]
    assert [
        item.status
        for item in prepared.summary.workflow_timeline[-1].verification_observations
    ] == [VerificationObservationStatus.FAILED.value]
