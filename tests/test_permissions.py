"""Tests for permission policy and tool lifecycle hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.llm.base import ToolCall
from loader.runtime.executor import ToolExecutionState, ToolExecutor
from loader.runtime.hooks import (
    BaseToolHook,
    HookDecision,
    HookManager,
    HookResult,
)
from loader.runtime.permissions import (
    PermissionMode,
    PermissionOverride,
    build_permission_policy,
)
from loader.runtime.tracing import RuntimeTracer
from loader.tools.base import create_default_registry


class RecordingHook(BaseToolHook):
    """Hook that records lifecycle events."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def pre_tool_use(self, context) -> HookResult:
        self.events.append("pre_tool_use")
        return HookResult()

    async def post_tool_use(self, context) -> HookResult:
        self.events.append("post_tool_use")
        return HookResult()

    async def post_tool_use_failure(self, context) -> HookResult:
        self.events.append("post_tool_use_failure")
        return HookResult()


class DenyInPreHook(BaseToolHook):
    """Hook that denies execution before the tool runs."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def pre_tool_use(self, context) -> HookResult:
        self.events.append("pre_tool_use")
        return HookResult(
            decision=HookDecision.DENY,
            message="[Blocked - denied by test hook]",
            terminal_state="blocked",
        )

    async def post_tool_use_failure(self, context) -> HookResult:
        self.events.append("post_tool_use_failure")
        return HookResult()


@pytest.mark.asyncio
async def test_permission_policy_honors_overrides(temp_dir: Path) -> None:
    policy = build_permission_policy(
        active_mode=PermissionMode.READ_ONLY,
        workspace_root=temp_dir,
        tool_requirements={"write": PermissionMode.WORKSPACE_WRITE},
    )

    denied = policy.authorize("write")
    allowed = policy.authorize("write", override=PermissionOverride.ALLOW)
    asked = policy.authorize("write", override=PermissionOverride.ASK)

    assert denied.decision.value == "deny"
    assert allowed.allowed
    assert asked.decision.value == "ask"


@pytest.mark.asyncio
async def test_hook_lifecycle_runs_in_order_for_success(temp_dir: Path) -> None:
    events: list[str] = []
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    executor = ToolExecutor(
        registry,
        RuntimeTracer(),
        policy,
        hooks=HookManager([RecordingHook(events)]),
    )
    target = temp_dir / "hook-success.txt"

    outcome = await executor.execute_tool_call(
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(target), "content": "hook success\n"},
        ),
        source="native",
        skip_confirmation=True,
    )

    assert outcome.state == ToolExecutionState.EXECUTED
    assert events == ["pre_tool_use", "post_tool_use"]
    assert target.read_text() == "hook success\n"


@pytest.mark.asyncio
async def test_pre_hook_deny_still_runs_failure_hook_once(temp_dir: Path) -> None:
    events: list[str] = []
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    executor = ToolExecutor(
        registry,
        RuntimeTracer(),
        policy,
        hooks=HookManager([DenyInPreHook(events)]),
    )
    target = temp_dir / "hook-denied.txt"

    outcome = await executor.execute_tool_call(
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(target), "content": "should not exist\n"},
        ),
        source="native",
        skip_confirmation=True,
    )

    assert outcome.state == ToolExecutionState.BLOCKED
    assert events == ["pre_tool_use", "post_tool_use_failure"]
    assert not target.exists()
    assert len(outcome.message.tool_results) == 1
    assert "denied by test hook" in outcome.event_content
