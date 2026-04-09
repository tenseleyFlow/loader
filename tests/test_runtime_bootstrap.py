"""Tests for shared runtime bootstrap and context synchronization."""

from __future__ import annotations

from pathlib import Path

from loader.agent.loop import Agent, AgentConfig
from loader.runtime.bootstrap import (
    RuntimeBootstrapView,
    build_runtime_bootstrap_source,
    build_runtime_context,
    sync_runtime_context,
)
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.explore import ExploreRuntime
from loader.runtime.launcher import RuntimeLauncher, build_runtime_launcher
from tests.helpers.runtime_harness import ScriptedBackend


def test_build_runtime_context_uses_shared_bootstrap_contract(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )
    source = build_runtime_bootstrap_source(agent)

    context = build_runtime_context(source)

    assert context.project_root == temp_dir.resolve()
    assert context.backend is agent.backend
    assert context.registry is agent.registry
    assert context.session is agent.session
    assert context.config is agent.config
    assert context.capability_profile == agent.capability_profile
    assert context.permission_policy is agent.permission_policy
    assert context.permission_config_status is agent.permission_config_status
    assert context.workflow_mode == agent.workflow_mode
    assert context.prompt_format == agent.prompt_format
    assert context.prompt_sections == agent.prompt_sections
    assert source.metadata == {"owner_type": "Agent"}


def test_sync_runtime_context_refreshes_prompt_and_capability_state(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(supports_native_tools=True)
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )
    source = build_runtime_bootstrap_source(agent)
    context = build_runtime_context(source)

    agent.prompt_format = "native"
    agent.prompt_sections = ["Workflow Context", "Runtime Config"]
    agent.set_workflow_mode("clarify")
    backend._supports_native_tools = False  # type: ignore[attr-defined]
    agent.refresh_capability_profile()

    sync_runtime_context(context, source)

    assert context.workflow_mode == "clarify"
    assert context.prompt_format == "native"
    assert context.prompt_sections == ["Workflow Context", "Runtime Config"]
    assert context.capability_profile.supports_native_tools is False


def test_conversation_runtime_uses_shared_bootstrap_factory(
    temp_dir: Path,
    monkeypatch,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    calls: list[str] = []
    real_build_runtime_context = build_runtime_context

    def fake_build_runtime_context(source) -> object:
        calls.append("conversation")
        return real_build_runtime_context(source)

    monkeypatch.setattr(
        "loader.runtime.conversation.build_runtime_context",
        fake_build_runtime_context,
    )

    source = build_runtime_bootstrap_source(agent)
    runtime = ConversationRuntime(source)

    assert calls == ["conversation"]
    assert runtime.context.project_root == temp_dir.resolve()
    assert runtime.source is source


def test_explore_runtime_uses_shared_bootstrap_factory(
    temp_dir: Path,
    monkeypatch,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    calls: list[str] = []
    real_build_runtime_context = build_runtime_context

    def fake_build_runtime_context(source) -> object:
        calls.append("explore")
        return real_build_runtime_context(source)

    monkeypatch.setattr(
        "loader.runtime.explore.build_runtime_context",
        fake_build_runtime_context,
    )

    source = build_runtime_bootstrap_source(agent)
    runtime = ExploreRuntime(source)

    assert calls == ["explore"]
    assert runtime.context.project_root == temp_dir.resolve()
    assert runtime.source is source


def test_build_runtime_launcher_wraps_shared_bootstrap_source(
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
