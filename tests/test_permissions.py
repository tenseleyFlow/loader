"""Tests for permission policy and tool lifecycle hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.llm.base import Message, Role, ToolCall
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.executor import ToolExecutionState, ToolExecutor
from loader.runtime.hooks import (
    ActiveRepairMutationScopeHook,
    ActiveRepairScopeHook,
    BaseToolHook,
    FilePathAliasHook,
    HookContext,
    HookDecision,
    HookManager,
    HookResult,
    LateReferenceDriftHook,
    RelativePathContextHook,
    SearchPathAliasHook,
)
from loader.runtime.permissions import (
    PermissionMode,
    PermissionOverride,
    PermissionRuleDisposition,
    PermissionRuleSet,
    build_permission_policy,
)
from loader.runtime.safeguard_services import ActionTracker
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


@pytest.mark.asyncio
async def test_relative_path_context_hook_remaps_workspace_mirror_of_external_root(
    temp_dir: Path,
) -> None:
    workspace_root = temp_dir / "workspace"
    workspace_root.mkdir()
    external_root = temp_dir / "external-home"
    external_fortran = external_root / "Loader" / "guides" / "fortran"
    external_fortran.mkdir(parents=True)
    (external_fortran / "index.html").write_text("<html></html>\n")
    (external_root / "Loader" / "guides").mkdir(exist_ok=True)

    registry = create_default_registry(workspace_root)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=workspace_root,
        tool_requirements=registry.get_tool_requirements(),
    )
    action_tracker = ActionTracker()
    action_tracker.record_tool_call(
        "read",
        {"file_path": str(external_fortran / "index.html")},
    )
    hook = RelativePathContextHook(action_tracker, workspace_root)

    mirrored_workspace_path = workspace_root / "Loader" / "guides" / "nginx" / "index.html"
    expected_external_path = external_root / "Loader" / "guides" / "nginx" / "index.html"

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="write-1",
                name="write",
                arguments={
                    "file_path": str(mirrored_workspace_path),
                    "content": "<html></html>\n",
                },
            ),
            tool=registry.get("write"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.updated_arguments is not None
    assert Path(result.updated_arguments["file_path"]).resolve() == expected_external_path.resolve()


class FakeSession:
    def __init__(self, *, active_dod_path: str, messages: list[Message]) -> None:
        self.active_dod_path = active_dod_path
        self.messages = messages


@pytest.mark.asyncio
async def test_active_repair_scope_hook_blocks_reference_reads_while_fixing(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "fixing"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "index.html"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/01-introduction.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{temp_dir / 'guide' / 'chapters' / '01-introduction.html'}`; otherwise remove or replace `chapters/01-introduction.html`.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(temp_dir / "reference" / "index.html")},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.DENY
    assert result.terminal_state == "blocked"
    assert result.message is not None
    assert "active repair scope" in result.message
    assert str(repair_target) in result.message


@pytest.mark.asyncio
async def test_active_repair_scope_hook_allows_reads_inside_active_artifact_set(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "fixing"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "index.html"
    chapter_path = temp_dir / "guide" / "chapters" / "01-getting-started.html"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/01-getting-started.html` in `{repair_target}`.\n"
                    f"- Fix the broken local reference `../styles.css` in `{chapter_path}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{chapter_path}`; otherwise remove or replace `chapters/01-getting-started.html`.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(chapter_path)},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_active_repair_scope_hook_allows_existing_sibling_reads_with_source_of_truth_hint(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "fixing"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "index.html"
    chapter_dir = temp_dir / "guide" / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    sibling = chapter_dir / "03-basic-usage.html"
    sibling.write_text("<h1>Basic Usage</h1>\n")
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/02-installation.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{chapter_dir / '02-installation.html'}`; otherwise remove or replace `chapters/02-installation.html`.\n"
                    "- Use the existing artifact files as the source of truth while repairing this file: "
                    f"`{repair_target}`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(sibling)},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_active_repair_scope_hook_allows_verification_source_outside_repair_target(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "chapters" / "06-troubleshooting.html"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `01-introduction.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="verify-1",
                name="read",
                arguments={"file_path": str(temp_dir / "guide" / "index.html")},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="verification",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_active_repair_scope_hook_blocks_local_rereads_outside_concrete_repair_files(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "chapters" / "05-advanced-configurations.html"
    stylesheet = temp_dir / "guide" / "styles.css"
    other_chapter = temp_dir / "guide" / "chapters" / "01-getting-started.html"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `../styles.css` in `{repair_target}`.\n"
                    f"- Fix the broken local reference `../styles.css` in `{temp_dir / 'guide' / 'chapters' / '06-troubleshooting.html'}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{stylesheet}`; otherwise remove or replace `../styles.css`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(other_chapter)},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.DENY
    assert result.terminal_state == "blocked"
    assert result.message is not None
    assert "active repair scope" in result.message
    assert str(repair_target) in result.message
    assert str(stylesheet) in result.message


@pytest.mark.asyncio
async def test_active_repair_scope_hook_blocks_repair_audit_loop_after_repeated_source_reads(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "fixing"
    dod_path = dod_store.save(dod)
    guide_root = temp_dir / "guide"
    chapter_dir = guide_root / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    repair_target = guide_root / "index.html"
    repair_target.write_text("<h1>Guide</h1>\n")
    intro = chapter_dir / "01-introduction.html"
    install = chapter_dir / "02-installation.html"
    intro.write_text("<h1>Intro</h1>\n")
    install.write_text("<h1>Install</h1>\n")
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/02-installation.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{install}`; otherwise remove or replace `chapters/02-installation.html`.\n"
                    "- Use the existing artifact files as the source of truth while repairing this file: "
                    f"`{repair_target}`, `{intro}`, `{install}`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    def make_context(index: int) -> HookContext:
        target = repair_target if index % 2 else intro
        return HookContext(
            tool_call=ToolCall(
                id=f"read-{index}",
                name="read",
                arguments={"file_path": str(target)},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )

    for index in range(1, 5):
        context = make_context(index)
        result = await hook.pre_tool_use(context)
        assert result.decision == HookDecision.CONTINUE
        await hook.post_tool_use(context)

    blocked = await hook.pre_tool_use(make_context(5))

    assert blocked.decision == HookDecision.DENY
    assert blocked.terminal_state == "blocked"
    assert blocked.message is not None
    assert "repair audit loop" in blocked.message


@pytest.mark.asyncio
async def test_active_repair_scope_hook_allows_scoped_glob_within_active_artifact_roots(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "index.html"
    guide_root = temp_dir / "guide"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/troubleshooting.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{guide_root / 'chapters' / 'troubleshooting.html'}`; otherwise remove or replace `chapters/troubleshooting.html`.\n"
                    "- Use the existing artifact files as the source of truth while repairing this file: "
                    f"`{guide_root / 'chapters' / 'introduction.html'}`, `{guide_root / 'chapters' / 'installation.html'}`, `{guide_root / 'chapters' / 'configuration.html'}`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="glob-1",
                name="glob",
                arguments={
                    "path": str(temp_dir),
                    "pattern": "**/guide/chapters/*.html",
                },
            ),
            tool=registry.get("glob"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_active_repair_scope_hook_allows_declared_missing_sibling_reads(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    guide_root = temp_dir / "guide"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    repair_target = guide_root / "index.html"
    existing_chapter = chapters / "overview.html"
    next_chapter = chapters / "installation.html"
    repair_target.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/overview.html">Overview</a>',
                '<a href="chapters/installation.html">Installation</a>',
                "</html>",
            ]
        )
        + "\n"
    )
    existing_chapter.write_text("<h1>Overview</h1>\n")

    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/overview.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{existing_chapter}`; otherwise remove or replace `chapters/overview.html`.\n"
                    "- Use the existing artifact files as the source of truth while repairing this file: "
                    f"`{existing_chapter}`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-allowed-sibling",
                name="read",
                arguments={"file_path": str(next_chapter)},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_active_repair_scope_hook_blocks_reference_reads_during_in_progress_repair(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "chapters" / "05-advanced-configurations.html"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `../styles.css` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{temp_dir / 'guide' / 'styles.css'}`; otherwise remove or replace `../styles.css`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(temp_dir / "reference" / "index.html")},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.DENY
    assert result.terminal_state == "blocked"
    assert result.message is not None
    assert "active repair scope" in result.message


@pytest.mark.asyncio
async def test_active_repair_mutation_scope_hook_blocks_writes_outside_named_repair_files(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "chapters" / "05-advanced-configurations.html"
    chapter_path = temp_dir / "guide" / "chapters" / "01-getting-started.html"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `../styles.css` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{temp_dir / 'guide' / 'styles.css'}`; otherwise remove or replace `../styles.css`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairMutationScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="edit-1",
                name="edit",
                arguments={"file_path": str(chapter_path), "old_string": "old", "new_string": "new"},
            ),
            tool=registry.get("edit"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.DENY
    assert result.terminal_state == "blocked"
    assert result.message is not None
    assert "active repair mutation scope" in result.message
    assert str(repair_target) in result.message


@pytest.mark.asyncio
async def test_active_repair_mutation_scope_hook_allows_expected_repair_file_writes(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "chapters" / "05-advanced-configurations.html"
    stylesheet = temp_dir / "guide" / "styles.css"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `../styles.css` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{stylesheet}`; otherwise remove or replace `../styles.css`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairMutationScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="write-1",
                name="write",
                arguments={"file_path": str(stylesheet), "content": "body { color: #222; }\n"},
            ),
            tool=registry.get("write"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_active_repair_mutation_scope_hook_allows_declared_missing_sibling_outputs(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    guide_root = temp_dir / "guide"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    repair_target = guide_root / "index.html"
    existing_chapter = chapters / "01-introduction.html"
    next_chapter = chapters / "02-installation.html"
    repair_target.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/01-introduction.html">Introduction</a>',
                '<a href="chapters/02-installation.html">Installation</a>',
                "</html>",
            ]
        )
        + "\n"
    )
    existing_chapter.write_text("<h1>Introduction</h1>\n")

    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/01-introduction.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{existing_chapter}`; otherwise remove or replace `chapters/01-introduction.html`.\n"
                    "- Use the existing artifact files as the source of truth while repairing this file: "
                    f"`{existing_chapter}`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairMutationScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="write-2",
                name="write",
                arguments={"file_path": str(next_chapter), "content": "<h1>Installation</h1>\n"},
            ),
            tool=registry.get("write"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_active_repair_mutation_scope_hook_blocks_broad_mutating_bash(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Repair the active artifact set")
    dod.status = "in_progress"
    dod_path = dod_store.save(dod)
    repair_target = temp_dir / "guide" / "chapters" / "05-advanced-configurations.html"
    session = FakeSession(
        active_dod_path=str(dod_path),
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `../styles.css` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{temp_dir / 'guide' / 'styles.css'}`; otherwise remove or replace `../styles.css`.\n"
                    "- Do not reread unrelated reference materials or restart discovery while this concrete repair target is unresolved.\n"
                ),
            )
        ],
    )
    hook = ActiveRepairMutationScopeHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="bash-1",
                name="bash",
                arguments={"command": f"mkdir -p {temp_dir / 'guide' / 'assets'}"},
            ),
            tool=registry.get("bash"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.DENY
    assert result.terminal_state == "blocked"
    assert result.message is not None
    assert "active repair mutation scope" in result.message
    assert str(repair_target) in result.message


@pytest.mark.asyncio
async def test_late_reference_drift_hook_blocks_out_of_scope_reference_reads(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    dod.status = "in_progress"
    plan_path = temp_dir / "implementation.md"
    plan_path.write_text(
        "# File Changes\n"
        "- `guide/index.html`\n"
        "- `guide/chapters/01-getting-started.html`\n"
        "- `guide/chapters/02-installation.html`\n"
        "- `guide/chapters/03-first-website.html`\n"
    )
    dod.implementation_plan = str(plan_path)
    dod_path = dod_store.save(dod)
    guide_dir = temp_dir / "guide" / "chapters"
    guide_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "guide" / "index.html").write_text("index")
    (guide_dir / "01-getting-started.html").write_text("one")
    (guide_dir / "02-installation.html").write_text("two")
    session = FakeSession(active_dod_path=str(dod_path), messages=[])
    hook = LateReferenceDriftHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(temp_dir / "reference" / "index.html")},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.DENY
    assert result.terminal_state == "blocked"
    assert result.message is not None
    assert "late reference drift" in result.message
    assert "03-first-website.html" in result.message


@pytest.mark.asyncio
async def test_late_reference_drift_hook_allows_reads_inside_planned_artifact_set(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    dod.status = "in_progress"
    plan_path = temp_dir / "implementation.md"
    plan_path.write_text(
        "# File Changes\n"
        "- `guide/index.html`\n"
        "- `guide/chapters/01-getting-started.html`\n"
        "- `guide/chapters/02-installation.html`\n"
        "- `guide/chapters/03-first-website.html`\n"
    )
    dod.implementation_plan = str(plan_path)
    dod_path = dod_store.save(dod)
    guide_dir = temp_dir / "guide" / "chapters"
    guide_dir.mkdir(parents=True, exist_ok=True)
    target = guide_dir / "02-installation.html"
    (temp_dir / "guide" / "index.html").write_text("index")
    (guide_dir / "01-getting-started.html").write_text("one")
    target.write_text("two")
    session = FakeSession(active_dod_path=str(dod_path), messages=[])
    hook = LateReferenceDriftHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(target)},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_late_reference_drift_hook_blocks_reference_reads_after_artifacts_exist(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    dod.status = "in_progress"
    plan_path = temp_dir / "implementation.md"
    plan_path.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'guide'}`",
                f"- `{temp_dir / 'guide' / 'chapters'}`",
                f"- `{temp_dir / 'guide' / 'index.html'}`",
                f"- `{temp_dir / 'guide' / 'chapters' / '01-getting-started.html'}`",
                f"- `{temp_dir / 'guide' / 'chapters' / '02-installation.html'}`",
                "",
            ]
        )
    )
    dod.implementation_plan = str(plan_path)
    guide_dir = temp_dir / "guide" / "chapters"
    guide_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "guide" / "index.html").write_text("index")
    (guide_dir / "01-getting-started.html").write_text("one")
    (guide_dir / "02-installation.html").write_text("two")
    dod_path = dod_store.save(dod)
    session = FakeSession(active_dod_path=str(dod_path), messages=[])
    hook = LateReferenceDriftHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(temp_dir / "reference" / "index.html")},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.DENY
    assert result.terminal_state == "blocked"
    assert result.message is not None
    assert "completed artifact set scope" in result.message
    assert str(temp_dir / "guide") in result.message


@pytest.mark.asyncio
async def test_late_reference_drift_hook_blocks_excessive_post_build_self_audits(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    dod.status = "in_progress"
    plan_path = temp_dir / "implementation.md"
    plan_path.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'guide' / 'index.html'}`",
                f"- `{temp_dir / 'guide' / 'chapters' / '01-getting-started.html'}`",
                f"- `{temp_dir / 'guide' / 'chapters' / '02-installation.html'}`",
                "",
            ]
        )
    )
    dod.implementation_plan = str(plan_path)
    guide_dir = temp_dir / "guide" / "chapters"
    guide_dir.mkdir(parents=True, exist_ok=True)
    target = guide_dir / "02-installation.html"
    (temp_dir / "guide" / "index.html").write_text("<h1>Nginx Guide</h1>\n")
    (guide_dir / "01-getting-started.html").write_text("<h1>Getting Started</h1>\n")
    target.write_text("<h1>Installation</h1>\n")
    dod_path = dod_store.save(dod)
    session = FakeSession(active_dod_path=str(dod_path), messages=[])
    hook = LateReferenceDriftHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    def make_context(index: int) -> HookContext:
        return HookContext(
            tool_call=ToolCall(
                id=f"read-{index}",
                name="read",
                arguments={"file_path": str(target)},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )

    for index in range(1, 5):
        context = make_context(index)
        result = await hook.pre_tool_use(context)
        assert result.decision == HookDecision.CONTINUE
        await hook.post_tool_use(context)

    blocked = await hook.pre_tool_use(make_context(5))

    assert blocked.decision == HookDecision.DENY
    assert blocked.terminal_state == "blocked"
    assert blocked.message is not None
    assert "post-build audit loop" in blocked.message


@pytest.mark.asyncio
async def test_late_reference_drift_hook_does_not_treat_empty_output_dir_as_complete_artifact_set(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    dod.status = "in_progress"
    dod.completed_items = ["Create chapter files with appropriate content"]
    plan_path = temp_dir / "implementation.md"
    plan_path.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'guide' / 'index.html'}`",
                f"- `{temp_dir / 'guide' / 'chapters'}/` (directory for chapter files)",
                "",
                "## Execution Order",
                "- Create chapter files with appropriate content",
            ]
        )
    )
    dod.implementation_plan = str(plan_path)
    guide_dir = temp_dir / "guide" / "chapters"
    guide_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "guide" / "index.html").write_text("index")
    dod_path = dod_store.save(dod)
    session = FakeSession(active_dod_path=str(dod_path), messages=[])
    hook = LateReferenceDriftHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(temp_dir / "reference" / "index.html")},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE


@pytest.mark.asyncio
async def test_late_reference_drift_hook_does_not_block_when_html_outputs_still_link_to_missing_files(
    temp_dir: Path,
) -> None:
    registry = create_default_registry(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
    )
    dod_store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    dod.status = "in_progress"
    dod.completed_items = ["Create chapter files with appropriate content"]
    plan_path = temp_dir / "implementation.md"
    plan_path.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'guide' / 'index.html'}`",
                f"- `{temp_dir / 'guide' / 'chapters'}/` (directory for chapter files)",
                "",
                "## Execution Order",
                "- Create chapter files with appropriate content",
            ]
        )
    )
    dod.implementation_plan = str(plan_path)
    guide_dir = temp_dir / "guide"
    chapters = guide_dir / "chapters"
    chapters.mkdir(parents=True, exist_ok=True)
    index = guide_dir / "index.html"
    index.write_text(
        '<a href="chapters/01-getting-started.html">One</a>\n'
        '<a href="chapters/02-installation.html">Two</a>\n'
    )
    (chapters / "01-getting-started.html").write_text("one")
    dod.touched_files = [str(index), str(chapters / "01-getting-started.html")]
    dod_path = dod_store.save(dod)
    session = FakeSession(active_dod_path=str(dod_path), messages=[])
    hook = LateReferenceDriftHook(
        dod_store=dod_store,
        project_root=temp_dir,
        session=session,
    )

    result = await hook.pre_tool_use(
        HookContext(
            tool_call=ToolCall(
                id="read-1",
                name="read",
                arguments={"file_path": str(temp_dir / "reference" / "index.html")},
            ),
            tool=registry.get("read"),
            registry=registry,
            permission_policy=policy,
            source="native",
        )
    )

    assert result.decision == HookDecision.CONTINUE
