"""Tests for the public runtime launcher seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig, ReasoningConfig
from loader.llm.base import CompletionResponse
from loader.runtime.launcher import RuntimeLauncher, build_runtime_launcher
from tests.helpers.runtime_harness import ScriptedBackend


def test_build_runtime_launcher_returns_launcher_for_agent_source(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )

    launcher = build_runtime_launcher(agent)

    assert isinstance(launcher, RuntimeLauncher)
    assert launcher.source is agent


@pytest.mark.asyncio
async def test_runtime_launcher_runs_conversation_turn(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Hello back.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    launcher = build_runtime_launcher(agent)
    events = []

    async def emit(event) -> None:
        events.append(event)

    summary = await launcher.run_turn("Hello there", emit)

    assert summary.final_response == "Hello back."
    assert any(event.type == "response" for event in events)


@pytest.mark.asyncio
async def test_runtime_launcher_runs_explore_query(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content="Quick repo summary.")
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    launcher = build_runtime_launcher(agent)
    events = []

    async def emit(event) -> None:
        events.append(event)

    summary = await launcher.run_explore("Give me a quick repo summary.", emit)

    assert summary.workflow_mode == "explore"
    assert summary.final_response == "Quick repo summary."
    assert any(event.type == "response" for event in events)


@pytest.mark.asyncio
async def test_runtime_launcher_runs_decomposition_fallback_turn(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content=(
                    '{"subtasks": [{"id": "1", "description": "Ship the feature", '
                    '"verification": "Done"}]}'
                )
            ),
            CompletionResponse(content="Feature shipped directly."),
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(
            auto_context=False,
            stream=False,
            reasoning=ReasoningConfig(decomposition=True),
        ),
        project_root=temp_dir,
    )
    launcher = build_runtime_launcher(agent)
    events = []

    async def emit(event) -> None:
        events.append(event)

    response = await launcher.run_decomposed(
        "Ship the feature",
        emit,
        requested_mode="execute",
        original_task="Ship the feature",
    )

    assert response == "Feature shipped directly."
    assert events[0].type == "thinking"
    assert any(event.type == "response" for event in events)
    assert not any(event.type == "decomposition" for event in events)
    assert agent.session.messages[0].content == "Ship the feature"
