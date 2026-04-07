"""Tests for per-iteration turn prelude handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import Message, Role
from loader.runtime.conversation import ConversationRuntime
from tests.helpers.runtime_harness import ScriptedBackend


def non_streaming_config() -> AgentConfig:
    """Shared config for direct turn-preamble tests."""

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
async def test_turn_preamble_seeds_action_hint_and_drains_steering(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)

    prepared, events, capture = await _prepare_runtime(
        runtime,
        task="Create a README for the runtime controller.",
    )
    agent.messages.append(
        Message(
            role=Role.USER,
            content="Create a README for the runtime controller.",
        )
    )
    agent._steering_queue.put_nowait("Stay inside src/loader/runtime.")

    decision = await runtime.turn_preamble.prepare_iteration(
        task=prepared.task,
        original_task=None,
        iterations=1,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        on_user_question=None,
        executor=prepared.executor,
    )

    assert not decision.should_continue
    assert prepared.summary.iterations == 1
    assert any(
        message.role.value == "assistant" and message.content == "["
        for message in agent.session.messages
    )
    assert (
        agent.session.messages[-1].content
        == "[USER INTERRUPTION]: Stay inside src/loader/runtime."
    )
    assert any(
        event.type == "steering" and event.content == "Stay inside src/loader/runtime."
        for event in events
    )


@pytest.mark.asyncio
async def test_turn_preamble_skips_iteration_when_recovery_refreshes(
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
        task="Explain the runtime controller shape.",
    )
    calls: list[str] = []

    async def fake_refresh(**kwargs) -> bool:
        calls.append(kwargs["task"])
        return True

    runtime.workflow_recovery.maybe_refresh_plan_for_drift = fake_refresh

    decision = await runtime.turn_preamble.prepare_iteration(
        task=prepared.task,
        original_task="Original task text",
        iterations=2,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        on_user_question=None,
        executor=prepared.executor,
    )

    assert decision.should_continue
    assert calls == ["Original task text"]
