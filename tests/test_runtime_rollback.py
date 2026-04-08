"""Tests for runtime-owned rollback planning."""

from __future__ import annotations

import pytest

from loader.runtime.rollback import (
    RollbackPlan,
    RollbackType,
    create_rollback_plan_for_action,
    execute_rollback,
    get_undo_command,
    is_destructive_tool,
)


def test_rollback_plan_formats_reverse_steps() -> None:
    plan = RollbackPlan()
    plan.add_file_creation("new.txt")
    plan.add_file_modification("config.json", '{"a":1}')

    assert plan.get_rollback_steps() == [
        "Restore: config.json",
        "Delete: new.txt",
    ]
    assert plan.to_prompt() == (
        "Rollback plan:\n"
        "  1. Restore: config.json\n"
        "  2. Delete: new.txt"
    )


def test_get_undo_command_handles_common_install_commands() -> None:
    assert get_undo_command("mkdir docs") == "rmdir docs"
    assert get_undo_command("npm install react") == "npm uninstall react"
    assert get_undo_command("pip install pytest") == "pip uninstall -y pytest"


def test_is_destructive_tool_covers_patch_and_bash_patterns() -> None:
    assert is_destructive_tool("patch", {"file_path": "notes.txt"}) is True
    assert is_destructive_tool("bash", {"command": "git checkout -- README.md"}) is True
    assert is_destructive_tool("bash", {"command": "ls -la"}) is False


@pytest.mark.asyncio
async def test_create_rollback_plan_for_new_write_returns_delete_action(tmp_path) -> None:
    target = tmp_path / "notes.txt"

    async def read_file(_path: str) -> str:
        raise AssertionError("new files should not be read for rollback")

    action = await create_rollback_plan_for_action(
        "write",
        {"file_path": str(target), "content": "alpha\n"},
        read_file,
    )

    assert action is not None
    assert action.type == RollbackType.FILE_DELETE
    assert action.file_path == str(target)


@pytest.mark.asyncio
async def test_execute_rollback_restores_file_and_marks_action(tmp_path) -> None:
    target = tmp_path / "notes.txt"
    plan = RollbackPlan()
    plan.add_file_modification(str(target), "restored\n")

    writes: list[tuple[str, str]] = []
    commands: list[str] = []

    async def write_file(path: str, content: str) -> None:
        writes.append((path, content))

    async def run_command(command: str) -> None:
        commands.append(command)

    results = await execute_rollback(plan, write_file, run_command)

    assert results == [f"✓ Restored: {target}"]
    assert writes == [(str(target), "restored\n")]
    assert commands == []
    assert plan.actions[0].executed is True
