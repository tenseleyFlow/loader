"""Tests for the runtime-first internal handle below Agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse
from loader.runtime.bootstrap import RuntimeBootstrapView, build_runtime_context
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.launcher import RuntimeLauncher, build_runtime_launcher
from loader.runtime.runtime_handle import RuntimeHandle
from tests.helpers.runtime_harness import ScriptedBackend


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
    assert launcher.source.metadata == {"owner_type": "RuntimeHandle"}
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
    assert runtime.source.metadata == {"owner_type": "RuntimeHandle"}
    assert any(event.type == "response" for event in events)
