"""Tests for typed runtime context construction."""

from __future__ import annotations

from pathlib import Path

from loader.agent.loop import Agent, AgentConfig
from loader.runtime.bootstrap import build_runtime_context
from loader.runtime.context import RuntimeContext
from loader.runtime.recovery import RecoveryContext
from tests.helpers.runtime_harness import ScriptedBackend


def test_agent_builds_typed_runtime_context(temp_dir: Path) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )

    context = build_runtime_context(agent)

    assert isinstance(context, RuntimeContext)
    assert context.project_root == temp_dir.resolve()
    assert context.backend is backend
    assert context.registry is agent.registry
    assert context.session is agent.session
    assert context.config is agent.config
    assert context.capability_profile == agent.capability_profile
    assert context.project_context is None
    assert context.permission_policy is agent.permission_policy
    assert context.permission_config_status is agent.permission_config_status
    assert context.workflow_mode == agent.workflow_mode
    assert context.safeguards is agent.safeguards
    assert context.messages is agent.session.messages
    assert context.use_react == agent.use_react
    assert context.active_permission_mode == agent.active_permission_mode
    assert context.active_permission_rule_counts == agent.active_permission_rule_counts
    assert context.reasoning is not None
    assert context.messages is agent.session.messages


def test_runtime_context_control_callbacks_stay_in_sync(temp_dir: Path) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )

    context = build_runtime_context(agent)
    context.queue_steering_message("Re-check the current task.")

    assert context.drain_steering_messages() == ["Re-check the current task."]

    context.set_workflow_mode("clarify")
    assert agent.workflow_mode == "clarify"
    assert context.workflow_mode == "clarify"

    recovery = RecoveryContext(original_tool="read", original_args={"file_path": "README.md"})
    context.recovery_context = recovery
    assert context.recovery_context is recovery

    backend = agent.backend
    backend._supports_native_tools = False  # type: ignore[attr-defined]
    context.refresh_capability_profile()
    assert context.capability_profile == agent.capability_profile
    assert context.capability_profile.supports_native_tools is False
