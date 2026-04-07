"""Tests for file and shell hardening introduced in Sprint 03."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.runtime.permissions import PermissionMode
from loader.tools.file_tools import ReadTool, WriteTool
from loader.tools.fs_safety import MAX_WRITE_SIZE
from loader.tools.shell_tools import BashTool


@pytest.mark.asyncio
async def test_read_tool_blocks_binary_file(temp_dir: Path) -> None:
    tool = ReadTool(workspace_root=temp_dir)
    binary_path = temp_dir / "binary.bin"
    binary_path.write_bytes(b"\x00\x01\x02")

    result = await tool.execute(file_path=str(binary_path))

    assert result.is_error
    assert "binary" in result.output.lower()


@pytest.mark.asyncio
async def test_read_tool_blocks_symlink_escape(temp_dir: Path) -> None:
    outside = temp_dir.parent / "outside.txt"
    outside.write_text("outside\n")
    inside_link = temp_dir / "escape.txt"
    inside_link.symlink_to(outside)
    tool = ReadTool(workspace_root=temp_dir)

    result = await tool.execute(file_path=str(inside_link))

    assert result.is_error
    assert "workspace boundary" in result.output.lower()


@pytest.mark.asyncio
async def test_write_tool_blocks_workspace_escape(temp_dir: Path) -> None:
    outside = temp_dir.parent / "outside-write.txt"
    tool = WriteTool(workspace_root=temp_dir)

    result = await tool.execute(file_path=str(outside), content="outside\n")

    assert result.is_error
    assert "workspace boundary" in result.output.lower()


@pytest.mark.asyncio
async def test_write_tool_blocks_oversize_content(temp_dir: Path) -> None:
    tool = WriteTool(workspace_root=temp_dir)
    target = temp_dir / "too-large.txt"

    result = await tool.execute(file_path=str(target), content="a" * (MAX_WRITE_SIZE + 1))

    assert result.is_error
    assert "too large" in result.output.lower()


@pytest.mark.asyncio
async def test_write_tool_returns_structured_patch_metadata(temp_dir: Path) -> None:
    tool = WriteTool(workspace_root=temp_dir)
    target = temp_dir / "patch.txt"

    result = await tool.execute(file_path=str(target), content="line one\nline two\n")

    assert not result.is_error
    assert result.metadata["file_path"] == str(target.resolve())
    assert result.metadata["kind"] == "create"
    assert result.metadata["structured_patch"]


@pytest.mark.asyncio
async def test_bash_tool_reports_truncation_metadata() -> None:
    tool = BashTool()

    result = await tool.execute(command='python -c "print(\'a\' * 60000)"')

    assert not result.is_error
    assert result.metadata["truncated"] is True
    assert result.metadata["output_limit"] == tool.OUTPUT_LIMIT
    assert "output truncated" in result.output.lower()


def test_bash_tool_classifies_permissions() -> None:
    tool = BashTool()

    assert tool.classify_command_permission("ls -la") == PermissionMode.READ_ONLY
    assert tool.classify_command_permission("touch notes.txt") == PermissionMode.WORKSPACE_WRITE
    assert tool.classify_command_permission("sudo rm -rf /tmp/test") == PermissionMode.DANGER_FULL_ACCESS
