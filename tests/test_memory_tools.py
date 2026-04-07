"""Tests for project memory and notepad tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, ToolCall
from loader.runtime.executor import ToolExecutionState, ToolExecutor
from loader.runtime.hooks import HookManager, MemoryLifecycleHook
from loader.runtime.permissions import PermissionMode, build_permission_policy
from loader.runtime.tracing import RuntimeTracer
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend, run_scenario


@pytest.mark.asyncio
async def test_project_memory_tools_round_trip(temp_dir: Path) -> None:
    registry = create_default_registry(temp_dir)

    write_result = await registry.execute(
        "project_memory_write",
        memory={"techStack": "Python", "build": "uv run pytest -q"},
        merge=True,
    )
    read_result = await registry.execute("project_memory_read", section="all")
    directive_result = await registry.execute(
        "project_memory_add_directive",
        directive="Use uv, never pip.",
        priority="high",
    )

    assert not write_result.is_error
    assert not read_result.is_error
    assert not directive_result.is_error
    memory = json.loads(read_result.output)
    assert memory["techStack"] == "Python"
    assert registry.get("project_memory_read").required_permission == PermissionMode.READ_ONLY
    assert registry.get("project_memory_write").required_permission == PermissionMode.WORKSPACE_WRITE


@pytest.mark.asyncio
async def test_notepad_tools_round_trip(temp_dir: Path) -> None:
    registry = create_default_registry(temp_dir)

    await registry.execute("notepad_write_priority", content="Keep Loader focused on runtime quality.")
    await registry.execute("notepad_write_working", content="Audit session resume behavior.")
    await registry.execute("notepad_write_manual", content="Never delete refs without asking.")
    read_result = await registry.execute("notepad_read", section="all")

    assert not read_result.is_error
    assert "Keep Loader focused on runtime quality." in read_result.output
    assert "Audit session resume behavior." in read_result.output
    assert "Never delete refs without asking." in read_result.output


@pytest.mark.asyncio
async def test_memory_lifecycle_hook_mirrors_directives_into_notepad(temp_dir: Path) -> None:
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
        hooks=HookManager([MemoryLifecycleHook()]),
    )

    outcome = await executor.execute_tool_call(
        ToolCall(
            id="directive-1",
            name="project_memory_add_directive",
            arguments={
                "directive": "Use uv, never pip.",
                "priority": "high",
            },
        ),
        source="native",
        skip_confirmation=True,
    )

    assert outcome.state == ToolExecutionState.EXECUTED
    notepad = (temp_dir / ".loader" / "notepad.md").read_text()
    assert "Remembered directive [high]: Use uv, never pip." in notepad


@pytest.mark.asyncio
async def test_definition_of_done_summary_is_captured_in_project_memory(
    temp_dir: Path,
) -> None:
    target = temp_dir / "memory-proof.txt"
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll create the file.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={"file_path": str(target), "content": "memory proof\n"},
                    )
                ],
            ),
            CompletionResponse(content="The file is in place."),
        ]
    )

    await run_scenario(
        "Create memory-proof.txt in the workspace root.",
        backend,
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )

    memory = json.loads((temp_dir / ".loader" / "project-memory.json").read_text())
    notes = memory.get("notes", [])
    assert any(
        note.get("category") == "definition_of_done"
        and "Verification:" in note.get("content", "")
        for note in notes
    )
