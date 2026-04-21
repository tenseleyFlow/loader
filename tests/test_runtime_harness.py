"""Deterministic runtime parity coverage for the current Loader loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import CompletionResponse, Role, StreamChunk, ToolCall
from loader.runtime.capabilities import resolve_capability_profile
from loader.runtime.permissions import PermissionMode
from tests.helpers.runtime_harness import (
    ScriptedBackend,
    run_explore_scenario,
    run_scenario,
)

SCENARIO_NAMES = [
    "streaming_text",
    "read_file_roundtrip",
    "multi_tool_turn_roundtrip",
    "turn_summary_smoke_for_multi_tool_turn",
    "write_file_allowed",
    "write_file_denied",
    "bash_stdout_roundtrip",
    "bash_confirmation_prompt_approved",
    "bash_confirmation_prompt_denied",
    "read_only_mode_denies_write",
    "read_only_mode_denies_mutating_bash",
    "read_only_mode_allows_safe_bash",
    "workspace_write_denies_write_outside_root",
    "danger_full_access_allows_dangerous_bash",
    "prompt_mode_prompts_destructive_write",
    "allow_mode_skips_prompt_for_destructive_write",
    "deny_rule_blocks_allowed_mode",
    "ask_rule_prompts_even_when_mode_would_allow",
    "raw_json_tool_call_fallback",
    "raw_json_todowrite_tool_call_fallback",
    "raw_json_patch_tool_call_fallback",
    "raw_json_ask_user_question_tool_call_fallback",
    "raw_bracket_ask_user_question_tool_call_fallback",
    "native_and_raw_tool_paths_share_executor_trace",
    "backend_capability_probe_refreshes_native_tool_mode",
    "run_streaming_delegates_to_primary_runtime",
    "definition_of_done_verify_phase",
    "verify_failure_routes_to_fix_loop",
    "verify_retry_budget_exhaustion",
    "ambiguous_prompt_routes_to_clarify",
    "complex_prompt_routes_to_plan",
    "verify_failure_fix_loop_does_not_reroute_workflow",
    "conversational_task_skips_verify_phase",
    "explore_mode_skips_dod_and_router",
    "explore_mode_denies_write",
    "explore_mode_ignores_global_allow_policy",
    "non_mutating_completion_no_longer_forces_continuation",
    "tool_result_contract_regression",
]


def load_manifest() -> list[dict[str, str]]:
    """Load the auditable parity scenario manifest."""

    manifest_path = Path(__file__).parent / "fixtures" / "runtime_parity_manifest.json"
    return json.loads(manifest_path.read_text())


def non_streaming_config(*, completion_check: bool = False) -> AgentConfig:
    """Shared config for deterministic complete() tests."""

    config = AgentConfig(auto_context=False, stream=False, max_iterations=8)
    config.reasoning.completion_check = completion_check
    return config


def native_tool_response(
    *tool_calls: ToolCall,
    content: str = "Using tools.",
) -> CompletionResponse:
    """Build a completion that includes native tool calls."""

    return CompletionResponse(content=content, tool_calls=list(tool_calls))


def final_response(content: str) -> CompletionResponse:
    """Build a completion with no further tool calls."""

    return CompletionResponse(content=content)


def tool_event_names(run) -> list[str]:
    """Return emitted tool event names in order."""

    return [
        event.tool_name
        for event in run.events
        if event.type == "tool_call" and event.tool_name and event.phase != "verification"
    ]


def tool_result_messages(run) -> list[str]:
    """Return emitted tool result messages in order."""

    return [
        event.content
        for event in run.events
        if event.type == "tool_result" and event.phase != "verification"
    ]


def verification_commands(run) -> list[str]:
    """Return verification-phase bash commands."""

    return [
        str((event.tool_args or {}).get("command", ""))
        for event in run.events
        if event.type == "tool_call" and event.phase == "verification"
    ]


def trace_event_names(run) -> list[str]:
    """Return recorded runtime trace event names."""

    summary = run.agent.last_turn_summary
    assert summary is not None
    return [event.name for event in summary.trace]


def dod_statuses(run) -> list[str]:
    """Return DoD statuses emitted during a run."""

    return [
        event.dod_status
        for event in run.events
        if event.type == "dod_status" and event.dod_status
    ]


def workflow_modes(run) -> list[str]:
    """Return emitted workflow modes in order."""

    return [
        event.workflow_mode
        for event in run.events
        if event.type == "workflow_mode" and event.workflow_mode
    ]


def artifact_kinds(run) -> list[str]:
    """Return emitted artifact kinds in order."""

    return [
        event.artifact_kind
        for event in run.events
        if event.type == "artifact" and event.artifact_kind
    ]


@pytest.mark.asyncio
async def test_runtime_parity_manifest_matches_implemented_cases() -> None:
    manifest_names = [entry["name"] for entry in load_manifest()]
    assert manifest_names == SCENARIO_NAMES


@pytest.mark.asyncio
async def test_streaming_text_scenario() -> None:
    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(content="Mock streaming ", is_done=False),
                StreamChunk(
                    content="says hello from Loader.",
                    full_content="Mock streaming says hello from Loader.",
                    is_done=True,
                ),
            ]
        ]
    )

    run = await run_scenario("hello there", backend, config=AgentConfig(auto_context=False))

    assert run.response == "Mock streaming says hello from Loader."
    assert [call.mode for call in run.invocations] == ["stream"]
    assert not tool_event_names(run)


@pytest.mark.asyncio
async def test_read_file_roundtrip(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\nbeta line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                content="I'll inspect that file.",
            ),
            final_response("The file contains alpha parity line."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file and summarize it.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert "alpha parity line" in run.response
    assert tool_event_names(run) == ["read"]
    assert any("alpha parity line" in message for message in tool_result_messages(run))
    assert len(run.invocations) == 2
    assert any(message.role == Role.TOOL for message in run.invocations[1].messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("alias_key", ["file", "filepath"])
async def test_read_file_alias_roundtrip(temp_dir: Path, alias_key: str) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\nbeta line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={alias_key: str(fixture)}),
                content="I'll inspect that file.",
            ),
            final_response("The file contains alpha parity line."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file and summarize it.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert "alpha parity line" in run.response
    assert tool_event_names(run) == ["read"]
    assert any("alpha parity line" in message for message in tool_result_messages(run))


@pytest.mark.asyncio
async def test_multi_tool_turn_roundtrip(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\nbeta line\ngamma parity line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                ToolCall(
                    id="grep-1",
                    name="grep",
                    arguments={"pattern": "parity", "path": str(fixture)},
                ),
                content="I'll inspect the file and count parity matches.",
            ),
            final_response("The file has two parity lines, including alpha parity line."),
        ]
    )

    run = await run_scenario(
        "Inspect the fixture and find parity lines.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read", "grep"]
    assert len(tool_result_messages(run)) == 2
    assert "two parity lines" in run.response


@pytest.mark.asyncio
async def test_turn_summary_smoke_for_multi_tool_turn(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\nbeta line\ngamma parity line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                ToolCall(
                    id="grep-1",
                    name="grep",
                    arguments={"pattern": "parity", "path": str(fixture)},
                ),
                content="I'll inspect the file and count parity matches.",
            ),
            final_response("The file has two parity lines, including alpha parity line."),
        ]
    )

    run = await run_scenario(
        "Inspect the fixture and find parity lines.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    summary = run.agent.last_turn_summary
    assert summary is not None
    assert summary.final_response == run.response
    assert summary.iterations == 2
    assert len(summary.assistant_messages) == 2
    assert len(summary.tool_result_messages) == 2
    assert "assistant.tool_batch" in trace_event_names(run)


@pytest.mark.asyncio
async def test_write_file_allowed(temp_dir: Path) -> None:
    target = temp_dir / "allowed.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "hello from loader\n"},
                ),
                content="I'll create the file now.",
            ),
            final_response("Successfully created the file."),
        ]
    )

    run = await run_scenario(
        "Create allowed.txt with a greeting.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert target.read_text() == "hello from loader\n"
    assert "Successfully created the file." in run.response
    assert tool_event_names(run) == ["write"]


@pytest.mark.asyncio
async def test_write_file_denied(temp_dir: Path) -> None:
    target = temp_dir / "denied.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.PROMPT
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "should not exist\n"},
                ),
                content="I'll create the file if you approve it.",
            ),
            final_response("I skipped the write as requested."),
        ]
    )

    async def deny_confirmation(tool_name: str, message: str, details: str) -> bool:
        assert tool_name == "write"
        assert "approval" in message.lower()
        assert "active_mode=prompt" in details
        return False

    run = await run_scenario(
        "Create denied.txt with a greeting.",
        backend,
        config=config,
        project_root=temp_dir,
        on_confirmation=deny_confirmation,
    )

    assert not target.exists()
    assert "skipped the write" in run.response.lower()
    assert any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_bash_stdout_roundtrip(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(temp_dir)
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": "pwd"}),
                content="I'll check the current directory.",
            ),
            final_response("Confirmed the working directory."),
        ]
    )

    run = await run_scenario(
        "Tell me the current directory.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert str(temp_dir) in tool_result_messages(run)[0]
    assert "Confirmed the working directory." in run.response


@pytest.mark.asyncio
async def test_bash_confirmation_prompt_approved(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "approved.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.PROMPT
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": "touch approved.txt"}),
                content="I'll create the file after approval.",
            ),
            final_response("The shell command completed."),
        ]
    )

    async def approve_confirmation(tool_name: str, message: str, details: str) -> bool:
        assert tool_name == "bash"
        assert "approval" in message.lower()
        assert "touch approved.txt" in details
        return True

    run = await run_scenario(
        "Create approved.txt using bash.",
        backend,
        config=config,
        project_root=temp_dir,
        on_confirmation=approve_confirmation,
    )

    assert target.exists()
    assert "shell command completed" in run.response.lower()
    assert any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_bash_confirmation_prompt_denied(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "denied-bash.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.PROMPT
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": "touch denied-bash.txt"}),
                content="I'll create the file if you allow it.",
            ),
            final_response("I left the shell command undone."),
        ]
    )

    async def deny_confirmation(tool_name: str, message: str, details: str) -> bool:
        assert tool_name == "bash"
        assert "touch denied-bash.txt" in details
        return False

    run = await run_scenario(
        "Create denied-bash.txt using bash.",
        backend,
        config=config,
        project_root=temp_dir,
        on_confirmation=deny_confirmation,
    )

    assert not target.exists()
    assert "left the shell command undone" in run.response.lower()
    assert any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_read_only_mode_denies_write(temp_dir: Path) -> None:
    config = non_streaming_config()
    config.permission_mode = PermissionMode.READ_ONLY
    config.auto_recover = False
    target = temp_dir / "blocked-by-policy.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "denied\n"},
                ),
                content="I'll create the file.",
            ),
            final_response("The write was blocked."),
        ]
    )

    run = await run_scenario(
        "Create blocked-by-policy.txt.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert not target.exists()
    assert any("requires workspace-write" in message for message in tool_result_messages(run))


@pytest.mark.asyncio
async def test_read_only_mode_denies_mutating_bash(temp_dir: Path) -> None:
    config = non_streaming_config()
    config.permission_mode = PermissionMode.READ_ONLY
    config.auto_recover = False
    target = temp_dir / "bash-blocked.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="bash-1",
                    name="bash",
                    arguments={"command": f"touch {target}"},
                ),
                content="I'll create the file with bash.",
            ),
            final_response("The bash command was blocked."),
        ]
    )

    run = await run_scenario(
        "Create bash-blocked.txt using bash.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert not target.exists()
    assert any("requires workspace-write" in message for message in tool_result_messages(run))


@pytest.mark.asyncio
async def test_read_only_mode_allows_safe_bash(temp_dir: Path) -> None:
    config = non_streaming_config()
    config.permission_mode = PermissionMode.READ_ONLY
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": "pwd"}),
                content="I'll inspect the current directory.",
            ),
            final_response("Inspected the current directory."),
        ]
    )

    run = await run_scenario(
        "Show the current directory.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["bash"]
    assert not any("requires" in message for message in tool_result_messages(run))


@pytest.mark.asyncio
async def test_workspace_write_denies_write_outside_root(temp_dir: Path) -> None:
    config = non_streaming_config()
    config.auto_recover = False
    outside = temp_dir.parent / "outside-root.txt"
    if outside.exists():
        outside.unlink()

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(outside), "content": "outside\n"},
                ),
                content="I'll write outside the workspace.",
            ),
            final_response("The write was blocked."),
        ]
    )

    async def decline_confirmation(_name: str, _msg: str, _details: str) -> bool:
        return False

    run = await run_scenario(
        "Write a file outside the workspace.",
        backend,
        config=config,
        project_root=temp_dir,
        on_confirmation=decline_confirmation,
    )

    assert not outside.exists()
    assert any(
        "declined" in message.lower() or "outside workspace" in message.lower()
        for message in tool_result_messages(run)
    )


@pytest.mark.asyncio
async def test_danger_full_access_allows_dangerous_bash(temp_dir: Path) -> None:
    target = temp_dir / "mode.txt"
    target.write_text("hello\n")
    config = non_streaming_config()
    config.permission_mode = PermissionMode.DANGER_FULL_ACCESS
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="bash-1",
                    name="bash",
                    arguments={"command": f"chmod 600 {target}"},
                ),
                content="I'll change the file permissions.",
            ),
            final_response("Updated the file permissions."),
        ]
    )

    run = await run_scenario(
        "Lock down mode.txt permissions.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["bash"]
    assert not any("requires" in message for message in tool_result_messages(run))
    assert not any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_prompt_mode_prompts_destructive_write(temp_dir: Path) -> None:
    target = temp_dir / "prompted.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.PROMPT
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "prompted\n"},
                ),
                content="I'll create the file after approval.",
            ),
            final_response("The file was created."),
        ]
    )
    prompts: list[str] = []

    async def approve_confirmation(tool_name: str, message: str, details: str) -> bool:
        assert tool_name == "write"
        prompts.append(details)
        return True

    run = await run_scenario(
        "Create prompted.txt after approval.",
        backend,
        config=config,
        project_root=temp_dir,
        on_confirmation=approve_confirmation,
    )

    assert target.read_text() == "prompted\n"
    assert prompts and "active_mode=prompt" in prompts[0]
    assert any(event.type == "confirmation" for event in run.events)


@pytest.mark.asyncio
async def test_allow_mode_skips_prompt_for_destructive_write(temp_dir: Path) -> None:
    target = temp_dir / "allow-mode.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.ALLOW
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "allow mode\n"},
                ),
                content="I'll create the file directly.",
            ),
            final_response("The file was created."),
        ]
    )
    prompts: list[str] = []

    async def unexpected_confirmation(tool_name: str, message: str, details: str) -> bool:
        prompts.append(tool_name)
        return False

    run = await run_scenario(
        "Create allow-mode.txt directly.",
        backend,
        config=config,
        project_root=temp_dir,
        on_confirmation=unexpected_confirmation,
    )

    assert target.read_text() == "allow mode\n"
    assert prompts == []
    assert not any(event.type == "confirmation" for event in run.events)
    assert "The file was created." in run.response


@pytest.mark.asyncio
async def test_deny_rule_blocks_allowed_mode(temp_dir: Path) -> None:
    loader_root = temp_dir / ".loader"
    loader_root.mkdir()
    (loader_root / "permission-rules.json").write_text(
        '{"deny": [{"tool": "write", "path_contains": "secrets"}]}\n'
    )
    target = temp_dir / "secrets.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.ALLOW
    config.auto_recover = False
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "denied\n"},
                ),
                content="I'll write the secret file.",
            ),
            final_response("The write was blocked by policy."),
        ]
    )

    run = await run_scenario(
        "Create secrets.txt.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert not target.exists()
    assert any("denied by rule" in message for message in tool_result_messages(run))
    assert "tool.permission_denied" in trace_event_names(run)


@pytest.mark.asyncio
async def test_ask_rule_prompts_even_when_mode_would_allow(temp_dir: Path) -> None:
    loader_root = temp_dir / ".loader"
    loader_root.mkdir()
    (loader_root / "permission-rules.json").write_text(
        '{"ask": [{"tool": "write", "path_contains": "README"}]}\n'
    )
    target = temp_dir / "README.md"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.ALLOW
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "ask rule\n"},
                ),
                content="I'll update the README if you approve it.",
            ),
            final_response("The write was declined."),
        ]
    )
    prompts: list[str] = []

    async def deny_confirmation(tool_name: str, message: str, details: str) -> bool:
        prompts.append(details)
        return False

    run = await run_scenario(
        "Update README.md.",
        backend,
        config=config,
        project_root=temp_dir,
        on_confirmation=deny_confirmation,
    )

    assert not target.exists()
    assert prompts and "matched_ask_rule=tool=write, path_contains=README" in prompts[0]
    assert any(event.type == "confirmation" for event in run.events)
    assert "declined" in run.response.lower()


@pytest.mark.asyncio
async def test_raw_json_tool_call_fallback(temp_dir: Path) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("alpha parity line\n")
    raw_json = f'{{"name": "read", "arguments": {{"file_path": "{fixture}"}}}}'

    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(content=raw_json[:25], is_done=False),
                StreamChunk(content=raw_json[25:], full_content=raw_json, is_done=True),
            ],
            [
                StreamChunk(
                    content="Recovered the raw JSON tool call and read the file.",
                    full_content="Recovered the raw JSON tool call and read the file.",
                    is_done=True,
                )
            ],
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=AgentConfig(auto_context=False, max_iterations=8),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read"]
    assert any("alpha parity line" in message for message in tool_result_messages(run))
    assert "Recovered the raw JSON tool call" in run.response


@pytest.mark.asyncio
async def test_raw_json_todowrite_tool_call_fallback(temp_dir: Path) -> None:
    raw_json = json.dumps(
        {
            "name": "TodoWrite",
            "arguments": {
                "todos": [
                    {
                        "content": "Run tests",
                        "active_form": "Running tests",
                        "status": "completed",
                    }
                ]
            },
        }
    )
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=raw_json),
            final_response("Tracked the current todo list."),
        ]
    )

    run = await run_scenario(
        "Track the current work items.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    todo_store = temp_dir / ".loader" / "todos" / "active.json"
    assert tool_event_names(run) == ["TodoWrite"]
    assert json.loads(todo_store.read_text()) == []
    assert "Tracked the current todo list." in run.response


@pytest.mark.asyncio
async def test_raw_json_patch_tool_call_fallback(temp_dir: Path) -> None:
    target = temp_dir / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    raw_json = json.dumps(
        {
            "name": "patch",
            "arguments": {
                "file_path": str(target),
                "hunks": [
                    {
                        "old_start": 2,
                        "old_lines": 1,
                        "new_start": 2,
                        "new_lines": 1,
                        "lines": ["-beta", "+beta updated"],
                    }
                ],
            },
        }
    )
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=raw_json),
            final_response("Patched sample.txt."),
        ]
    )

    run = await run_scenario(
        "Update sample.txt.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["patch"]
    assert target.read_text() == "alpha\nbeta updated\ngamma\n"
    assert "Patched sample.txt." in run.response


@pytest.mark.asyncio
async def test_native_patch_tool_accepts_unified_diff_string(temp_dir: Path) -> None:
    target = temp_dir / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="patch-1",
                    name="patch",
                    arguments={
                        "file_path": str(target),
                        "patch": (
                            "--- a/sample.txt\n"
                            "+++ b/sample.txt\n"
                            "@@ -2,1 +2,1 @@\n"
                            "-beta\n"
                            "+beta updated\n"
                        ),
                    },
                ),
                content="I'll patch the file directly.",
            ),
            final_response("Patched sample.txt."),
        ]
    )

    run = await run_scenario(
        "Update sample.txt.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["patch"]
    assert target.read_text() == "alpha\nbeta updated\ngamma\n"
    assert "Patched sample.txt." in run.response


@pytest.mark.asyncio
async def test_raw_json_ask_user_question_tool_call_fallback(temp_dir: Path) -> None:
    raw_json = json.dumps(
        {
            "name": "AskUserQuestion",
            "arguments": {
                "title": "Path Choice",
                "context": "Choose the safer Loader cleanup path.",
                "question": "Which path should we take?",
                "options": [
                    {
                        "label": "Plan first",
                        "description": "Keep the next move documented.",
                    },
                    {
                        "label": "Execute now",
                        "description": "Start changing code immediately.",
                    },
                ],
            },
        }
    )
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=raw_json),
            final_response("We'll execute now."),
        ]
    )

    async def answer(question: str, options: list[str] | None) -> str:
        assert "Which path should we take?" in question
        assert options == [
            "Plan first - Keep the next move documented.",
            "Execute now - Start changing code immediately.",
        ]
        return "2"

    run = await run_scenario(
        "Decide the next path before changing code.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
        on_user_question=answer,
    )

    assert tool_event_names(run) == ["AskUserQuestion"]
    assert any("Execute now" in message for message in tool_result_messages(run))
    assert "We'll execute now." in run.response


@pytest.mark.asyncio
async def test_raw_bracket_ask_user_question_tool_call_fallback(temp_dir: Path) -> None:
    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(
                    content='[calls askuserquestion tool with: question="Which path should we take?"]',
                    full_content='[calls askuserquestion tool with: question="Which path should we take?"]',
                    is_done=True,
                )
            ],
            [
                StreamChunk(
                    content="We'll plan first.",
                    full_content="We'll plan first.",
                    is_done=True,
                )
            ],
        ]
    )

    async def answer(question: str, options: list[str] | None) -> str:
        assert "Which path should we take?" in question
        assert options is None
        return "Plan first"

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=AgentConfig(auto_context=False, max_iterations=8),
        project_root=temp_dir,
        on_user_question=answer,
    )

    assert tool_event_names(run) == ["AskUserQuestion"]
    assert any('"answer": "Plan first"' in message for message in tool_result_messages(run))
    assert "We'll plan first." in run.response


@pytest.mark.asyncio
async def test_non_streaming_bracket_ask_user_question_tool_call_fallback(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content='[calls askuserquestion tool with: question="Which path should we take?"]'
            ),
            final_response("We'll plan first."),
        ]
    )

    async def answer(question: str, options: list[str] | None) -> str:
        assert "Which path should we take?" in question
        assert options is None
        return "Plan first"

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
        on_user_question=answer,
    )

    assert tool_event_names(run) == ["AskUserQuestion"]
    assert any('"answer": "Plan first"' in message for message in tool_result_messages(run))
    assert "We'll plan first." in run.response


@pytest.mark.asyncio
async def test_native_and_raw_tool_paths_share_executor_trace(temp_dir: Path) -> None:
    native_fixture = temp_dir / "native.txt"
    native_fixture.write_text("native parity line\n")
    native_backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(native_fixture)}),
                content="I'll inspect the native tool result.",
            ),
            final_response("Native read complete."),
        ]
    )
    native_run = await run_scenario(
        "Read native.txt.",
        native_backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    raw_fixture = temp_dir / "raw.txt"
    raw_fixture.write_text("raw parity line\n")
    raw_json = f'{{"name": "read", "arguments": {{"file_path": "{raw_fixture}"}}}}'
    raw_backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(content=raw_json[:20], is_done=False),
                StreamChunk(content=raw_json[20:], full_content=raw_json, is_done=True),
            ],
            [
                StreamChunk(
                    content="Raw read complete.",
                    full_content="Raw read complete.",
                    is_done=True,
                )
            ],
        ]
    )
    raw_run = await run_scenario(
        "Read raw.txt.",
        raw_backend,
        config=AgentConfig(auto_context=False, max_iterations=8),
        project_root=temp_dir,
    )

    for run in (native_run, raw_run):
        names = trace_event_names(run)
        assert "assistant.tool_batch" in names
        assert "tool.received" in names
        assert "tool.executed" in names

    native_summary = native_run.agent.last_turn_summary
    raw_summary = raw_run.agent.last_turn_summary
    assert native_summary is not None
    assert raw_summary is not None
    assert any(
        event.name == "tool.received" and event.data["source"] == "native"
        for event in native_summary.trace
    )
    assert any(
        event.name == "tool.received" and event.data["source"] == "raw_text"
        for event in raw_summary.trace
    )


@pytest.mark.asyncio
async def test_backend_capability_probe_refreshes_native_tool_mode(
    temp_dir: Path,
) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("capability probe line\n")

    class LazyCapabilityBackend(ScriptedBackend):
        def __init__(self, completions: list[CompletionResponse]) -> None:
            super().__init__(completions=completions, supports_native_tools=False)
            self.model = "custom-qwen-build"
            self._described = False

        async def describe_model(self) -> dict[str, dict[str, list[str]]]:
            self._described = True
            return {"details": {"families": ["qwen2.5"]}}

        def capability_profile(self):
            model_details = (
                {"details": {"families": ["qwen2.5"]}} if self._described else None
            )
            return resolve_capability_profile(
                self.model,
                model_details=model_details,
            )

    backend = LazyCapabilityBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                content="I'll inspect that file after probing capabilities.",
            ),
            final_response("Capability probing enabled the native read."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file after checking model capabilities.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert backend._described
    assert not run.agent.use_react
    assert run.invocations[0].tools is not None
    assert tool_event_names(run) == ["read"]
    assert "Capability probing enabled the native read." in run.response


@pytest.mark.asyncio
async def test_run_streaming_delegates_to_primary_runtime(temp_dir: Path) -> None:
    fixture = temp_dir / "streaming.txt"
    fixture.write_text("streamed runtime line\n")
    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(
                    content="I'll inspect the file now.",
                    full_content="I'll inspect the file now.",
                    tool_calls=[
                        ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)})
                    ],
                    is_done=True,
                )
            ],
            [
                StreamChunk(
                    content="Finished reading the streamed fixture.",
                    full_content="Finished reading the streamed fixture.",
                    is_done=True,
                )
            ],
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False, max_iterations=8),
        project_root=temp_dir,
    )

    events = [event async for event in agent.run_streaming("Read the streamed fixture file.")]

    assert any(event.type == "tool_call" and event.tool_name == "read" for event in events)
    assert any(
        event.type == "tool_result" and "streamed runtime line" in event.content
        for event in events
    )
    assert agent.last_turn_summary is not None
    assert agent.last_turn_summary.final_response.startswith(
        "Finished reading the streamed fixture."
    )


@pytest.mark.asyncio
async def test_definition_of_done_verify_phase(temp_dir: Path) -> None:
    target = temp_dir / "verified.txt"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "verified\n"},
                ),
                content="I'll create the file now.",
            ),
            final_response("Created verified.txt."),
        ]
    )

    run = await run_scenario(
        "Create verified.txt with a line of text.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert verification_commands(run) == [f"test -f {target}"]
    assert dod_statuses(run) == ["draft", "verifying", "done"]
    assert "Verification:" in run.response
    assert run.agent.last_turn_summary is not None
    assert run.agent.last_turn_summary.verification_status == "passed"
    assert run.agent.last_turn_summary.definition_of_done is not None


@pytest.mark.asyncio
async def test_verify_failure_routes_to_fix_loop(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "broken.py"
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "print(\n"},
                ),
                content="I'll create the script.",
            ),
            final_response("Created broken.py."),
            native_tool_response(
                ToolCall(
                    id="write-2",
                    name="write",
                    arguments={
                        "file_path": str(target),
                        "content": "print('fixed from verify loop')\n",
                    },
                ),
                content="I'll fix the verification failure.",
            ),
            final_response("Fixed broken.py."),
        ]
    )

    run = await run_scenario(
        "Create broken.py and make sure it runs.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert target.read_text() == "print('fixed from verify loop')\n"
    assert verification_commands(run) == ["python broken.py", "python broken.py"]
    assert "fixing" in dod_statuses(run)
    assert "Verification:" in run.response
    assert run.agent.last_turn_summary is not None
    assert run.agent.last_turn_summary.verification_status == "passed"


@pytest.mark.asyncio
async def test_verify_retry_budget_exhaustion(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "still-broken.py"
    config = non_streaming_config()
    config.verification_retry_budget = 1
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "print(\n"},
                ),
                content="I'll create the script.",
            ),
            final_response("Created still-broken.py."),
            native_tool_response(
                ToolCall(
                    id="write-2",
                    name="write",
                    arguments={"file_path": str(target), "content": "print(\n"},
                ),
                content="I'll try one more fix.",
            ),
            final_response("Tried to fix still-broken.py."),
        ]
    )

    run = await run_scenario(
        "Create still-broken.py and make sure it runs.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert "couldn't verify" in run.response.lower()
    assert dod_statuses(run)[-1] == "failed"
    assert run.agent.last_turn_summary is not None
    assert run.agent.last_turn_summary.verification_status == "failed"


@pytest.mark.asyncio
async def test_ambiguous_prompt_routes_to_clarify(temp_dir: Path) -> None:
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="ask-1",
                    name="AskUserQuestion",
                    arguments={
                        "question": (
                            "What outcome matters most, and what should stay out of scope?"
                        )
                    },
                ),
                content="I need one clarification first.",
            ),
            final_response(
                "\n".join(
                    [
                        "## Task Statement",
                        "Improve Loader so it feels more like claw-code.",
                        "",
                        "## Desired Outcome",
                        "- Make Loader more reliable without broad redesign.",
                        "",
                        "## In Scope",
                        "- Tighten the runtime workflow around the user-facing goal.",
                        "",
                        "## Non Goals",
                        "- Rebuild unrelated subsystems.",
                        "",
                        "## Decision Boundaries",
                        "- Escalate before changing unrelated UX patterns.",
                        "",
                        "## Constraints",
                        "- Stay inside the current repository.",
                        "",
                        "## Likely Touchpoints",
                        "- Runtime entry points and prompt behavior.",
                        "",
                        "## Assumptions",
                        "- The user wants a narrow runtime-quality improvement.",
                        "",
                        "## Acceptance Criteria",
                        "- The improvement stays focused on runtime behavior.",
                    ]
                )
            ),
            final_response("I have the brief and can move forward."),
        ]
    )

    async def answer(question: str, options: list[str] | None) -> str:
        assert "outcome matters most" in question.lower()
        assert options is None
        return "Do not redesign the whole interface."

    run = await run_scenario(
        "Improve Loader so it feels more like claw-code.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
        on_user_question=answer,
    )

    dod = run.agent.last_turn_summary.definition_of_done
    assert dod is not None
    assert workflow_modes(run)[:2] == ["clarify", "execute"]
    assert artifact_kinds(run) == ["clarify_brief"]
    assert dod.clarify_brief is not None
    assert Path(dod.clarify_brief).exists()


@pytest.mark.asyncio
async def test_complex_prompt_routes_to_plan(temp_dir: Path) -> None:
    target = temp_dir / "planned.txt"
    backend = ScriptedBackend(
        completions=[
            final_response(
                "\n".join(
                    [
                        "# Implementation Plan",
                        "",
                        "## File Changes",
                        f"- Create {target.name} in the workspace root.",
                        "",
                        "## Execution Order",
                        f"1. Write {target.name}.",
                        "2. Confirm the file exists.",
                        "",
                        "## Risks",
                        "- Writing the wrong file path.",
                        "",
                        "<<<VERIFICATION>>>",
                        "",
                        "# Verification Plan",
                        "",
                        "## Acceptance Criteria",
                        f"- {target.name} exists in the workspace root.",
                        "",
                        "## Verification Commands",
                        f"- `test -f {target}`",
                        "",
                        "## Notes",
                        "- Use a deterministic file existence check.",
                    ]
                )
            ),
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "planned output\n"},
                ),
                content="I'll create the file now.",
            ),
            final_response("The file is in place."),
        ]
    )

    run = await run_scenario(
        "Implement a persistent workflow mode router with clarify artifacts, "
        "planning artifacts, and verification-plan wiring in the runtime.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    dod = run.agent.last_turn_summary.definition_of_done
    assert dod is not None
    assert workflow_modes(run)[:3] == ["plan", "execute", "verify"]
    assert artifact_kinds(run) == ["implementation_plan", "verification_plan"]
    assert not any(event.type == "decomposition" for event in run.events)
    assert not any(event.type == "subtask" for event in run.events)
    assert dod.verification_commands == [f"test -f {target}"]
    assert verification_commands(run) == [f"test -f {target}"]


@pytest.mark.asyncio
async def test_verify_failure_fix_loop_does_not_reroute_workflow(temp_dir: Path) -> None:
    target = temp_dir / "retry.txt"
    backend = ScriptedBackend(
        completions=[
            final_response(
                "\n".join(
                    [
                        "# Implementation Plan",
                        "",
                        "## File Changes",
                        f"- Create {target.name}.",
                        "",
                        "## Execution Order",
                        f"1. Write {target.name}.",
                        "2. Fix it if verification fails.",
                        "",
                        "## Risks",
                        "- Initial content may be wrong.",
                        "",
                        "<<<VERIFICATION>>>",
                        "",
                        "# Verification Plan",
                        "",
                        "## Acceptance Criteria",
                        "- The file contains the word fixed.",
                        "",
                        "## Verification Commands",
                        f"- `grep -q fixed {target}`",
                        "",
                        "## Notes",
                        "- Retry if the first write misses the target string.",
                    ]
                )
            ),
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={"file_path": str(target), "content": "draft output\n"},
                ),
                content="I'll write the first draft.",
            ),
            final_response("First draft is written."),
            native_tool_response(
                ToolCall(
                    id="write-2",
                    name="write",
                    arguments={"file_path": str(target), "content": "fixed output\n"},
                ),
                content="I'll correct the file.",
            ),
            final_response("The file now contains the fixed output."),
        ]
    )

    run = await run_scenario(
        "Implement a persistent workflow mode router with clarify artifacts, "
        "planning artifacts, and verification-plan wiring in the runtime.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    modes = workflow_modes(run)
    assert modes.count("plan") == 1
    assert modes.count("clarify") == 0
    assert modes.count("execute") >= 2
    assert modes.count("verify") >= 2


@pytest.mark.asyncio
async def test_conversational_task_skips_verify_phase() -> None:
    backend = ScriptedBackend(
        streams=[
            [
                StreamChunk(content="Hello there.", full_content="Hello there.", is_done=True),
            ]
        ]
    )

    run = await run_scenario("hello there", backend, config=AgentConfig(auto_context=False))

    assert run.response == "Hello there."
    assert not dod_statuses(run)
    assert run.agent.last_turn_summary is None


@pytest.mark.asyncio
async def test_explore_mode_skips_dod_and_router(temp_dir: Path) -> None:
    target = temp_dir / "feature.py"
    target.write_text("def important_helper():\n    return 1\n")
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="grep-1",
                    name="grep",
                    arguments={
                        "pattern": "important_helper",
                        "path": str(temp_dir),
                        "include": "*.py",
                    },
                ),
                content="I'll search for that helper.",
            ),
            final_response("important_helper is defined in feature.py."),
        ]
    )

    run = await run_explore_scenario(
        "Where is important_helper defined?",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert "feature.py" in run.response
    assert tool_event_names(run) == ["grep"]
    assert not dod_statuses(run)
    assert not workflow_modes(run)
    assert run.agent.last_turn_summary is not None
    assert run.agent.last_turn_summary.definition_of_done is None
    assert run.agent.last_turn_summary.workflow_mode == "explore"
    assert "explore.completed" in trace_event_names(run)
    assert not (temp_dir / ".loader" / "dod").exists()
    assert run.invocations[0].tools is not None
    assert "write" not in {tool["name"] for tool in run.invocations[0].tools or []}


@pytest.mark.asyncio
async def test_explore_mode_denies_write(temp_dir: Path) -> None:
    target = temp_dir / "new.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.WORKSPACE_WRITE
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={
                        "file_path": str(target),
                        "content": "not allowed\n",
                    },
                ),
                content="I'll write a file.",
            ),
            final_response("Explore mode is read-only, so I cannot make that change here."),
        ]
    )

    run = await run_explore_scenario(
        "Create a new file anyway.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert not target.exists()
    assert tool_event_names(run) == ["write"]
    assert any("read-only" in message.lower() for message in tool_result_messages(run))
    assert "cannot make that change" in run.response.lower()
    assert "tool.permission_denied" in trace_event_names(run)
    assert not dod_statuses(run)
    assert not workflow_modes(run)
    assert not (temp_dir / ".loader" / "dod").exists()


@pytest.mark.asyncio
async def test_explore_mode_ignores_global_allow_policy(temp_dir: Path) -> None:
    loader_root = temp_dir / ".loader"
    loader_root.mkdir()
    (loader_root / "permission-rules.json").write_text(
        '{"allow": [{"tool": "write", "path_contains": "new.txt"}]}\n'
    )
    target = temp_dir / "new.txt"
    config = non_streaming_config()
    config.permission_mode = PermissionMode.ALLOW
    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="write-1",
                    name="write",
                    arguments={
                        "file_path": str(target),
                        "content": "still denied\n",
                    },
                ),
                content="I'll write a file.",
            ),
            final_response("Explore mode is read-only, so I cannot make that change here."),
        ]
    )

    run = await run_explore_scenario(
        "Create a new file anyway.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert not target.exists()
    assert any("read-only" in message.lower() for message in tool_result_messages(run))
    assert "tool.permission_denied" in trace_event_names(run)
    assert not dod_statuses(run)
    assert not workflow_modes(run)


@pytest.mark.asyncio
async def test_informational_completion_allows_explicit_done_without_continuation(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "hello.py"
    backend = ScriptedBackend(
        completions=[
            final_response("Done."),
        ]
    )
    config = non_streaming_config(completion_check=True)

    run = await run_scenario(
        "Explain how a hello.py file would work.",
        backend,
        config=config,
        project_root=temp_dir,
    )

    assert not target.exists()
    assert not any(event.type == "completion_check" for event in run.events)
    assert tool_event_names(run) == []
    assert run.response == "Done."


@pytest.mark.asyncio
async def test_tool_result_contract_regression() -> None:
    errors: list[str] = []
    duplicate_path = "/tmp/already-created.txt"

    duplicate_backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="dup-1",
                    name="write",
                    arguments={"file_path": duplicate_path, "content": "already there\n"},
                ),
                content="I'll create the file again.",
            ),
            final_response("Skipped the duplicate write."),
        ]
    )
    duplicate_agent = Agent(duplicate_backend, config=non_streaming_config())
    duplicate_agent.safeguards.record_action(
        "write",
        {"file_path": duplicate_path, "content": "already there\n"},
    )

    try:
        await duplicate_agent.run("Create /tmp/already-created.txt again.")
    except TypeError as exc:
        errors.append(f"duplicate branch raised {exc}")

    validation_backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="invalid-1", name="bash", arguments={"command": ""}),
                content="I'll run that command.",
            ),
            final_response("Blocked the invalid command."),
        ]
    )
    validation_agent = Agent(validation_backend, config=non_streaming_config())

    try:
        await validation_agent.run("Run an empty command.")
    except TypeError as exc:
        errors.append(f"validation branch raised {exc}")

    assert not errors, "\n".join(errors)


@pytest.mark.asyncio
async def test_duplicate_read_is_skipped_without_intervening_mutation(
    temp_dir: Path,
) -> None:
    fixture = temp_dir / "index.html"
    fixture.write_text("alpha parity line\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="read-1", name="read", arguments={"file_path": str(fixture)}),
                content="I'll inspect the file.",
            ),
            native_tool_response(
                ToolCall(id="read-2", name="read", arguments={"file_path": str(fixture)}),
                content="I'll reread the same file.",
            ),
            final_response("I'll use the existing file contents instead of rereading."),
        ]
    )

    run = await run_scenario(
        "Inspect index.html and keep moving.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read", "read"]
    messages = tool_result_messages(run)
    assert any("alpha parity line" in message for message in messages)
    assert any(
        "Skipped - duplicate action" in message and "Already read" in message
        for message in messages
    )
    assert "existing file contents" in run.response


@pytest.mark.asyncio
async def test_duplicate_observation_queues_steering_to_reuse_prior_evidence(
    temp_dir: Path,
) -> None:
    chapters = temp_dir / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text("<h1>Chapter 1: Introduction to Fortran</h1>\n")
    (chapters / "02-setup.html").write_text("<h1>Chapter 2: Setting Up Fortran</h1>\n")
    index_file = temp_dir / "index.html"
    index_file.write_text("broken table of contents\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="glob-1",
                    name="glob",
                    arguments={"path": str(chapters), "pattern": "*.html"},
                ),
                content="I'll inspect the chapter inventory first.",
            ),
            native_tool_response(
                ToolCall(
                    id="read-1",
                    name="read",
                    arguments={"file_path": str(index_file)},
                ),
                content="I'll inspect the index next.",
            ),
            native_tool_response(
                ToolCall(
                    id="read-2",
                    name="read",
                    arguments={"file_path": str(index_file)},
                ),
                content="I'll reopen the index.",
            ),
            final_response("I'll reuse the earlier evidence and patch the index next."),
        ]
    )

    run = await run_scenario(
        "Update index.html so the table of contents links are correct.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    messages = tool_result_messages(run)
    steering_messages = [
        event.content
        for event in run.events
        if event.type == "steering" and event.content
    ]

    assert any("reuse the earlier read result instead of rereading" in message for message in messages)
    assert any("Reuse the earlier observation instead of repeating it." in message for message in steering_messages)
    assert any("index.html" in message for message in steering_messages)


@pytest.mark.asyncio
async def test_interleaved_reread_is_allowed_once_without_intervening_mutation(
    temp_dir: Path,
) -> None:
    index_file = temp_dir / "index.html"
    chapter_file = temp_dir / "chapter-1.html"
    index_file.write_text("table of contents\n")
    chapter_file.write_text("chapter body\n")

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(
                    id="read-1",
                    name="read",
                    arguments={"file_path": str(index_file)},
                ),
                content="I'll inspect the index first.",
            ),
            native_tool_response(
                ToolCall(
                    id="read-2",
                    name="read",
                    arguments={"file_path": str(chapter_file)},
                ),
                content="I'll inspect the chapter next.",
            ),
            native_tool_response(
                ToolCall(
                    id="read-3",
                    name="read",
                    arguments={"file_path": str(index_file)},
                ),
                content="I'll reopen the index to reconcile the findings.",
            ),
            final_response("I re-opened the index after checking the chapter."),
        ]
    )

    run = await run_scenario(
        "Inspect the index, inspect a chapter, then return to the index.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read", "read", "read"]
    messages = tool_result_messages(run)
    assert not any("Skipped - duplicate action" in message for message in messages)
    assert sum("table of contents" in message for message in messages) == 2
    assert any("chapter body" in message for message in messages)


@pytest.mark.asyncio
async def test_repeated_bash_probe_is_allowed_after_mutation(
    temp_dir: Path,
) -> None:
    target = temp_dir / "notes.txt"
    target.write_text("old value\n")
    list_command = f"ls -1 {temp_dir}"

    backend = ScriptedBackend(
        completions=[
            native_tool_response(
                ToolCall(id="bash-1", name="bash", arguments={"command": list_command}),
                content="I'll inspect the directory first.",
            ),
            native_tool_response(
                ToolCall(
                    id="edit-1",
                    name="edit",
                    arguments={
                        "file_path": str(target),
                        "old_string": "old value",
                        "new_string": "new value",
                    },
                ),
                content="I'll update the file.",
            ),
            native_tool_response(
                ToolCall(id="bash-2", name="bash", arguments={"command": list_command}),
                content="I'll list the directory again after the edit.",
            ),
            final_response("I re-ran ls after the edit without hitting duplicate rejection."),
        ]
    )

    run = await run_scenario(
        "Inspect the directory, edit the file, then inspect again.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["bash", "edit", "bash"]
    messages = tool_result_messages(run)
    assert not any("Skipped - duplicate action" in message for message in messages)
    assert sum("notes.txt" in message for message in messages) >= 2
    assert target.read_text() == "new value\n"
