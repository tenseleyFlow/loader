"""Tests for main turn-loop orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.turn_iteration import TurnIterationAction, TurnIterationDecision
from loader.runtime.turn_preamble import TurnPreludeDecision
from tests.helpers.runtime_harness import ScriptedBackend


def non_streaming_config() -> AgentConfig:
    """Shared config for direct turn-loop tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=8)


async def _prepare_runtime(
    runtime: ConversationRuntime,
    *,
    task: str,
) -> tuple:
    events = []

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task=task,
        emit=capture,
        requested_mode="execute",
        original_task=None,
        on_user_question=None,
    )
    return prepared, events, capture


@pytest.mark.asyncio
async def test_turn_loop_retries_after_preamble_continue(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    prepared, _, capture = await _prepare_runtime(
        runtime,
        task="Explain the runtime loop shape.",
    )
    prelude_calls: list[int] = []
    iteration_calls: list[int] = []

    async def fake_preamble(**kwargs):
        prelude_calls.append(kwargs["iterations"])
        kwargs["summary"].iterations = kwargs["iterations"]
        return TurnPreludeDecision(should_continue=kwargs["iterations"] == 1)

    async def fake_iteration(**kwargs):
        iteration_calls.append(kwargs["iterations"])
        return TurnIterationDecision(
            action=TurnIterationAction.COMPLETE,
            continuation_count=0,
            empty_retry_count=0,
            extracted_iterations=0,
            consecutive_errors=0,
        )

    runtime.turn_preamble.prepare_iteration = fake_preamble
    runtime.turn_iteration.run_iteration = fake_iteration

    loop_exit = await runtime.turn_loop.run_loop(
        task=prepared.task,
        effective_task=prepared.effective_task,
        original_task=None,
        effective_max_tokens=prepared.effective_max_tokens,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=runtime._emit_confirmation(capture),
    )

    assert prelude_calls == [1, 2]
    assert iteration_calls == [2]
    assert prepared.summary.iterations == 2
    assert loop_exit.reason_code == "turn_complete"
    assert loop_exit.reason_summary == "Finalizing completed turn"


@pytest.mark.asyncio
async def test_turn_loop_carries_iteration_state_between_attempts(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    prepared, _, capture = await _prepare_runtime(
        runtime,
        task="Inspect the loop bookkeeping.",
    )
    iteration_inputs: list[tuple[int, int, int, int, int, list[str]]] = []

    async def fake_preamble(**kwargs):
        kwargs["summary"].iterations = kwargs["iterations"]
        return TurnPreludeDecision()

    async def fake_iteration(**kwargs):
        iteration_inputs.append(
            (
                kwargs["iterations"],
                kwargs["continuation_count"],
                kwargs["empty_retry_count"],
                kwargs["extracted_iterations"],
                kwargs["consecutive_errors"],
                list(kwargs["actions_taken"]),
            )
        )
        if len(iteration_inputs) == 1:
            return TurnIterationDecision(
                action=TurnIterationAction.CONTINUE,
                continuation_count=1,
                empty_retry_count=2,
                extracted_iterations=1,
                consecutive_errors=1,
                new_actions_taken=["read: {'file_path': 'README.md'}"],
            )
        return TurnIterationDecision(
            action=TurnIterationAction.FINALIZE,
            continuation_count=1,
            empty_retry_count=2,
            extracted_iterations=1,
            consecutive_errors=0,
            finalize_reason_code="tool_batch_halted",
            finalize_reason_summary="Finalizing after halted tool batch",
        )

    runtime.turn_preamble.prepare_iteration = fake_preamble
    runtime.turn_iteration.run_iteration = fake_iteration

    loop_exit = await runtime.turn_loop.run_loop(
        task=prepared.task,
        effective_task=prepared.effective_task,
        original_task=None,
        effective_max_tokens=prepared.effective_max_tokens,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=runtime._emit_confirmation(capture),
    )

    assert iteration_inputs == [
        (1, 0, 0, 0, 0, []),
        (2, 1, 2, 1, 1, ["read: {'file_path': 'README.md'}"]),
    ]
    assert loop_exit.reason_code == "tool_batch_halted"
    assert loop_exit.reason_summary == "Finalizing after halted tool batch"
