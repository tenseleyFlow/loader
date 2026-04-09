"""Tests for the public runtime launcher seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig, ReasoningConfig
from loader.llm.base import CompletionResponse, StreamChunk
from loader.runtime.bootstrap import RuntimeBootstrapView
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
    assert isinstance(launcher.source, RuntimeBootstrapView)
    assert launcher.source is not agent
    assert launcher.source.metadata == {"owner_type": "Agent"}


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


@pytest.mark.asyncio
async def test_runtime_launcher_routes_user_message_to_conversational_fast_path(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(
            streams=[
                [
                    StreamChunk(content="Quick ", is_done=False),
                    StreamChunk(
                        content="reply.",
                        full_content="Quick reply.",
                        is_done=True,
                    ),
                ]
            ]
        ),
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )
    launcher = build_runtime_launcher(agent)
    events = []

    async def emit(event) -> None:
        events.append(event)

    response = await launcher.run_user_message("thanks", emit)

    assert response == "Quick reply."
    assert agent.current_task is None
    assert any(event.type == "response" and event.content == "Quick reply." for event in events)


@pytest.mark.asyncio
async def test_runtime_launcher_routes_user_message_to_direct_runtime_turn(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Feature shipped directly.")]
        ),
        config=AgentConfig(
            auto_context=False,
            stream=False,
            reasoning=ReasoningConfig(completion_check=False),
        ),
        project_root=temp_dir,
    )
    launcher = build_runtime_launcher(agent)
    events = []

    async def emit(event) -> None:
        events.append(event)

    response = await launcher.run_user_message(
        "Write a short release-note style summary of what Loader does well.",
        emit,
        use_plan=False,
    )

    assert response == "Feature shipped directly."
    assert agent.current_task == "Write a short release-note style summary of what Loader does well."
    assert agent.last_turn_summary is not None
    assert agent.last_turn_summary.final_response == "Feature shipped directly."
    assert (
        agent.session.messages[0].content
        == "Write a short release-note style summary of what Loader does well."
    )
    assert any(event.type == "response" for event in events)


@pytest.mark.asyncio
async def test_runtime_launcher_routes_user_message_through_decomposition_lane(
    temp_dir: Path,
    monkeypatch,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(
            auto_context=False,
            stream=False,
            reasoning=ReasoningConfig(decomposition=True, completion_check=False),
        ),
        project_root=temp_dir,
    )
    launcher = build_runtime_launcher(agent)
    events = []
    calls = []

    async def emit(event) -> None:
        events.append(event)

    async def fake_run_decomposed(
        task: str,
        emit,
        *,
        on_confirmation=None,
        on_user_question=None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> str:
        calls.append(
            {
                "task": task,
                "requested_mode": requested_mode,
                "original_task": original_task,
            }
        )
        return "All done."

    monkeypatch.setattr(launcher, "run_decomposed", fake_run_decomposed)

    response = await launcher.run_user_message(
        "Read the spec and implement the feature",
        emit,
        use_plan=False,
    )

    assert response == "All done."
    assert agent.current_task == "Read the spec and implement the feature"
    assert calls == [
        {
            "task": "Read the spec and implement the feature",
            "requested_mode": "execute",
            "original_task": "Read the spec and implement the feature",
        }
    ]
    assert events == []
