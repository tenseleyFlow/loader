"""Tests for Sprint 06 tool-surface expansion."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from loader.tools.base import create_default_registry
from loader.tools.file_tools import PatchTool
from loader.tools.git_tools import GitTool
from loader.tools.workflow_tools import AskUserQuestionTool


@pytest.mark.asyncio
async def test_patch_tool_applies_structured_hunks(temp_dir: Path) -> None:
    target = temp_dir / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    tool = PatchTool(workspace_root=temp_dir)

    result = await tool.execute(
        file_path=str(target),
        hunks=[
            {
                "old_start": 2,
                "old_lines": 1,
                "new_start": 2,
                "new_lines": 1,
                "lines": ["-beta", "+beta updated"],
            }
        ],
    )

    assert result.is_error is False
    assert target.read_text() == "alpha\nbeta updated\ngamma\n"
    assert result.metadata["structured_patch"]


@pytest.mark.asyncio
async def test_patch_tool_rejects_context_mismatch(temp_dir: Path) -> None:
    target = temp_dir / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n")
    tool = PatchTool(workspace_root=temp_dir)

    result = await tool.execute(
        file_path=str(target),
        hunks=[
            {
                "old_start": 2,
                "old_lines": 1,
                "new_start": 2,
                "new_lines": 1,
                "lines": ["-wrong line", "+beta updated"],
            }
        ],
    )

    assert result.is_error is True
    assert "context mismatch" in result.output


@pytest.mark.asyncio
async def test_patch_tool_accepts_replacement_block_hunks(temp_dir: Path) -> None:
    target = temp_dir / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\ndelta\n")
    tool = PatchTool(workspace_root=temp_dir)

    result = await tool.execute(
        file_path=str(target),
        hunks=[
            {
                "old_start": 2,
                "old_end": 3,
                "new_lines": [
                    "beta updated",
                    "gamma updated",
                    "inserted line",
                ],
            }
        ],
    )

    assert result.is_error is False
    assert target.read_text() == "alpha\nbeta updated\ngamma updated\ninserted line\ndelta\n"
    assert result.metadata["structured_patch"]


@pytest.mark.asyncio
async def test_git_tool_inspects_read_only_repo_state(temp_dir: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=temp_dir, check=True)
    (temp_dir / "README.md").write_text("loader\n")

    tool = GitTool(workspace_root=temp_dir)
    status = await tool.execute(action="status", args=["--short"], cwd=".")
    branch = await tool.execute(action="branch", args=["--show-current"], cwd=".")

    assert status.is_error is False
    assert "README.md" in status.output
    assert branch.is_error is False


@pytest.mark.asyncio
async def test_notepad_append_alias_updates_selected_section(temp_dir: Path) -> None:
    registry = create_default_registry(temp_dir)

    working = await registry.execute(
        "notepad_append",
        content="Investigate explore runtime.",
        section="working",
    )
    manual = await registry.execute(
        "notepad_append",
        content="Keep read-only mode strict.",
        section="manual",
    )
    notepad = await registry.execute("notepad_read", section="all")

    assert working.is_error is False
    assert manual.is_error is False
    assert "Investigate explore runtime." in notepad.output
    assert "Keep read-only mode strict." in notepad.output


@pytest.mark.asyncio
async def test_richer_ask_user_question_formats_context_and_options() -> None:
    tool = AskUserQuestionTool()

    async def answer(question: str, options: list[str] | None) -> str:
        assert "Fix Strategy" in question
        assert "Pick the safer repair path." in question
        assert "Which fix should Loader apply?" in question
        assert options == [
            "Minimal patch - touch the parser only",
            "Broader refactor - clean up parser and tests",
        ]
        return "1"

    result = await tool.execute(
        title="Fix Strategy",
        context="Pick the safer repair path.",
        question="Which fix should Loader apply?",
        options=[
            {"label": "Minimal patch", "description": "touch the parser only"},
            {"label": "Broader refactor", "description": "clean up parser and tests"},
        ],
        user_response_handler=answer,
    )

    payload = json.loads(result.output)
    assert result.is_error is False
    assert payload["answer"] == "Minimal patch - touch the parser only"
