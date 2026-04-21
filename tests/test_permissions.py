"""Tests for permission policy and tool lifecycle hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.llm.base import ToolCall
from loader.runtime.executor import ToolExecutionState, ToolExecutor
from loader.runtime.hooks import (
    BaseToolHook,
    FilePathAliasHook,
    HookDecision,
    HookContext,
    HookManager,
    HookResult,
    SearchPathAliasHook,
)
from loader.runtime.permissions import (
    PermissionMode,
    PermissionOverride,
    PermissionRuleDisposition,
    PermissionRuleSet,
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


def test_permission_mode_parsing_supports_prompt_and_allow() -> None:
    assert PermissionMode.from_str("prompt") == PermissionMode.PROMPT
    assert PermissionMode.from_str("allow") == PermissionMode.ALLOW


def test_permission_policy_honors_rule_precedence(temp_dir: Path) -> None:
    policy = build_permission_policy(
        active_mode=PermissionMode.ALLOW,
        workspace_root=temp_dir,
        tool_requirements={"write": PermissionMode.WORKSPACE_WRITE},
        rules=PermissionRuleSet.from_dict(
            {
                "allow": [{"tool": "write", "contains": "safe change"}],
                "deny": [{"tool": "write", "path_contains": "secrets"}],
                "ask": [{"tool": "write", "path_contains": "README"}],
            }
        ),
    )

    denied = policy.authorize(
        "write",
        arguments={
            "file_path": str(temp_dir / "secrets.txt"),
            "content": "safe change\n",
        },
    )
    asked = policy.authorize(
        "write",
        arguments={
            "file_path": str(temp_dir / "README.md"),
            "content": "safe change\n",
        },
    )
    allowed = policy.authorize(
        "write",
        arguments={
            "file_path": str(temp_dir / "notes.txt"),
            "content": "safe change\n",
        },
    )

    assert denied.decision.value == "deny"
    assert denied.matched_disposition == PermissionRuleDisposition.DENY
    assert asked.decision.value == "ask"
    assert asked.matched_disposition == PermissionRuleDisposition.ASK
    assert allowed.decision.value == "allow"
    assert allowed.matched_disposition == PermissionRuleDisposition.ALLOW


@pytest.mark.asyncio
async def test_prompt_mode_executor_prompts_once_and_respects_denial(
    temp_dir: Path,
) -> None:
    prompts: list[tuple[str, str, str]] = []
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.PROMPT,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    executor = ToolExecutor(registry, RuntimeTracer(), policy)
    target = temp_dir / "prompted.txt"

    async def deny(tool_name: str, message: str, details: str) -> bool:
        prompts.append((tool_name, message, details))
        return False

    outcome = await executor.execute_tool_call(
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(target), "content": "prompted\n"},
        ),
        source="native",
        on_confirmation=deny,
    )

    assert outcome.state == ToolExecutionState.DECLINED
    assert not target.exists()
    assert len(prompts) == 1
    assert "active_mode=prompt" in prompts[0][2]
    assert "required_mode=workspace-write" in prompts[0][2]


@pytest.mark.asyncio
async def test_allow_mode_executor_skips_prompt_for_destructive_write(
    temp_dir: Path,
) -> None:
    prompts: list[str] = []
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.ALLOW,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    executor = ToolExecutor(registry, RuntimeTracer(), policy)
    target = temp_dir / "allowed.txt"

    async def unexpected(tool_name: str, message: str, details: str) -> bool:
        prompts.append(tool_name)
        return False

    outcome = await executor.execute_tool_call(
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(target), "content": "allowed\n"},
        ),
        source="native",
        on_confirmation=unexpected,
    )

    assert outcome.state == ToolExecutionState.EXECUTED
    assert target.read_text() == "allowed\n"
    assert prompts == []


@pytest.mark.asyncio
async def test_ask_rule_prompts_even_when_allow_mode(temp_dir: Path) -> None:
    prompts: list[str] = []
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.ALLOW,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
        rules=PermissionRuleSet.from_dict(
            {"ask": [{"tool": "write", "path_contains": "README"}]}
        ),
    )
    executor = ToolExecutor(registry, RuntimeTracer(), policy)
    target = temp_dir / "README.md"

    async def deny(tool_name: str, message: str, details: str) -> bool:
        prompts.append(details)
        return False

    outcome = await executor.execute_tool_call(
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(target), "content": "no thanks\n"},
        ),
        source="native",
        on_confirmation=deny,
    )

    assert outcome.state == ToolExecutionState.DECLINED
    assert not target.exists()
    assert len(prompts) == 1
    assert "matched_ask_rule=tool=write, path_contains=README" in prompts[0]


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_path"),
    [
        ("read", {"file": "notes.txt"}, "notes.txt"),
        ("write", {"filepath": "notes.txt", "content": "hello\n"}, "notes.txt"),
        (
            "edit",
            {"filePath": "notes.txt", "old_string": "before", "new_string": "after"},
            "notes.txt",
        ),
        ("patch", {"path": "notes.txt", "hunks": []}, "notes.txt"),
    ],
)
async def test_file_path_alias_hook_canonicalizes_common_aliases(
    temp_dir: Path,
    tool_name: str,
    arguments: dict[str, object],
    expected_path: str,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    hook = FilePathAliasHook()

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(id=f"{tool_name}-1", name=tool_name, arguments=arguments),
            tool=registry.get(tool_name),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.updated_arguments is not None
    assert result.updated_arguments["file_path"] == expected_path
    for alias in ("file", "filepath", "filePath", "filename", "path"):
        assert alias not in result.updated_arguments


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_path"),
    [
        ("glob", {"pattern": "*.html", "directory": "chapters"}, "chapters"),
        ("grep", {"pattern": "alpha", "dir": "src"}, "src"),
    ],
)
async def test_search_path_alias_hook_canonicalizes_common_aliases(
    temp_dir: Path,
    tool_name: str,
    arguments: dict[str, object],
    expected_path: str,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    hook = SearchPathAliasHook()

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(id=f"{tool_name}-1", name=tool_name, arguments=arguments),
            tool=registry.get(tool_name),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.updated_arguments is not None
    assert result.updated_arguments["path"] == expected_path
    for alias in ("directory", "dir", "folder"):
        assert alias not in result.updated_arguments


@pytest.mark.asyncio
async def test_search_path_alias_hook_splits_full_glob_pattern(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    hook = SearchPathAliasHook()
    chapters = temp_dir / "chapters"

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="glob-1",
                name="glob",
                arguments={"pattern": f"{chapters}/*.html"},
            ),
            tool=registry.get("glob"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.updated_arguments is not None
    assert result.updated_arguments["path"] == str(chapters)
    assert result.updated_arguments["pattern"] == "*.html"
