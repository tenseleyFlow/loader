"""Tests for the runtime-owned conversational fast path."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import StreamChunk
from loader.runtime.chat_lane import CHAT_SYSTEM_PROMPT, ConversationalTurnRunner
from loader.runtime.launcher import build_runtime_launcher
from tests.helpers.runtime_harness import ScriptedBackend


@pytest.mark.asyncio
async def test_conversational_turn_runner_streams_and_persists_history(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(
            streams=[
                [
                    StreamChunk(content="Hello ", is_done=False),
                    StreamChunk(
                        content="back.",
                        full_content="Hello back.",
                        is_done=True,
                    ),
                ]
            ]
        ),
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )
    runner = ConversationalTurnRunner(agent)
    events = []

    async def emit(event) -> None:
        events.append(event)

    response = await runner.run("hello there", emit)

    assert response == "Hello back."
    assert [event.type for event in events] == ["thinking", "stream", "stream", "response"]
    assert agent.session.messages[-2].content == "hello there"
    assert agent.session.messages[-1].content == "Hello back."
    assert agent.backend.invocations[0].messages[0].content == CHAT_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_runtime_launcher_runs_conversational_fast_path(
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

    response = await launcher.run_conversational("thanks", emit)

    assert response == "Quick reply."
    assert any(event.type == "response" and event.content == "Quick reply." for event in events)
