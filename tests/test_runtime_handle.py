"""Tests for the runtime-first internal handle below Agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, StreamChunk
from loader.runtime.bootstrap import RuntimeBootstrapView, build_runtime_context
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.launcher import RuntimeLauncher, build_runtime_launcher
from loader.runtime.runtime_handle import RuntimeHandle
from tests.helpers.runtime_harness import ScriptedBackend, run_explore_scenario, run_scenario


def test_runtime_handle_builds_runtime_bootstrap_contract(
    temp_dir: Path,
) -> None:
    handle = RuntimeHandle(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )

    launcher = build_runtime_launcher(handle)
    context = build_runtime_context(handle)

    assert isinstance(launcher, RuntimeLauncher)
    assert isinstance(launcher.source, RuntimeBootstrapView)
    assert launcher.source is not handle
    assert launcher.source.metadata == {
        "owner_type": "RuntimeHandle",
        "owner_path": "runtime-handle",
    }
    assert context.project_root == temp_dir.resolve()
    assert context.backend is handle.backend
    assert context.registry is handle.registry
    assert context.session is handle.session
    assert context.permission_policy is handle.permission_policy
    assert context.workflow_mode == handle.workflow_mode


@pytest.mark.asyncio
async def test_runtime_handle_runs_conversation_runtime_without_agent(
    temp_dir: Path,
) -> None:
    handle = RuntimeHandle(
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Runtime handle reply.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(handle)
    events = []

    async def emit(event) -> None:
        events.append(event)

    summary = await runtime.run_turn(
        "Explain why runtime-first seams matter here.",
        emit,
        requested_mode="execute",
    )

    assert summary.final_response == "Runtime handle reply."
    assert runtime.source.metadata == {
        "owner_type": "RuntimeHandle",
        "owner_path": "runtime-handle",
    }
    assert any(event.type == "response" for event in events)


@pytest.mark.asyncio
async def test_runtime_handle_runs_public_shell_entrypoint_without_agent(
    temp_dir: Path,
) -> None:
    handle = RuntimeHandle(
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Runtime handle shell reply.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    events = []

    async def emit(event) -> None:
        events.append(event)

    response = await handle.run(
        "Summarize the runtime-first shell path.",
        on_event=emit,
        use_plan=False,
    )

    assert response == "Runtime handle shell reply."
    assert handle.last_turn_summary is not None
    assert handle.last_turn_summary.final_response == "Runtime handle shell reply."
    assert any(event.type == "response" for event in events)


@pytest.mark.asyncio
async def test_runtime_handle_runs_explore_and_streaming_entrypoints_without_agent(
    temp_dir: Path,
) -> None:
    handle = RuntimeHandle(
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Explore with runtime handle.")],
            streams=[
                [
                    StreamChunk(content="Streamed ", is_done=False),
                    StreamChunk(
                        content="runtime reply.",
                        full_content="Streamed runtime reply.",
                        is_done=True,
                    ),
                ]
            ],
        ),
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )

    stream_events = [event async for event in handle.run_streaming("thanks")]
    explore_response = await handle.run_explore(
        "Where should I start in this repo?",
    )

    assert any(
        event.type == "response" and event.content == "Streamed runtime reply."
        for event in stream_events
    )
    assert explore_response == "Explore with runtime handle."
    assert handle.last_turn_summary is not None
    assert handle.last_turn_summary.workflow_mode == "explore"


@pytest.mark.asyncio
async def test_runtime_harness_uses_runtime_handle_for_scripted_runs(
    temp_dir: Path,
) -> None:
    run = await run_scenario(
        "Summarize the runtime-first harness.",
        ScriptedBackend(
            completions=[CompletionResponse(content="Runtime harness reply.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    explore_run = await run_explore_scenario(
        "Where should I start?",
        ScriptedBackend(
            completions=[CompletionResponse(content="Explore harness reply.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )

    assert isinstance(run.agent, RuntimeHandle)
    assert run.response == "Runtime harness reply."
    assert isinstance(explore_run.agent, RuntimeHandle)
    assert explore_run.response == "Explore harness reply."
