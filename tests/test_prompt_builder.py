"""Tests for Loader's typed system-prompt builder."""

from __future__ import annotations

from pathlib import Path

from loader.context.project import ProjectContext
from loader.runtime.prompting import (
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    build_system_prompt_result,
)


def _tool_schema(name: str = "read") -> dict[str, object]:
    return {
        "name": name,
        "description": "Inspect repository files.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to inspect.",
                }
            },
            "required": ["file_path"],
        },
    }


def test_prompt_builder_renders_dynamic_sections_with_metadata(temp_dir: Path) -> None:
    context = ProjectContext(
        root=temp_dir,
        project_type="python",
        package_manager="uv",
        test_framework="pytest",
        structure=["src/", "tests/"],
        test_command="uv run pytest -q",
    )

    result = build_system_prompt_result(
        tools=[_tool_schema()],
        use_react=False,
        project_context=context,
        workflow_mode="clarify",
        permission_mode="prompt",
        cwd=temp_dir,
        current_task="Clarify the acceptance criteria for Loader.",
    )

    assert result.prompt_format == "native"
    assert result.dynamic_section_names == [
        "Runtime Config",
        "Workflow Context",
        "Mode Guidance",
        "Project Context",
        "Project Tips",
    ]
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in result.content
    assert "## Clarify Mode" in result.content
    assert "Permission mode: `prompt`" in result.content
    assert "Current task: Clarify the acceptance criteria for Loader." in result.content
    assert "Package manager: uv" in result.content


def test_prompt_builder_keeps_sections_stable_across_formats(temp_dir: Path) -> None:
    native = build_system_prompt_result(
        tools=[_tool_schema("bash")],
        use_react=False,
        workflow_mode="execute",
        permission_mode="workspace-write",
        cwd=temp_dir,
    )
    react = build_system_prompt_result(
        tools=[_tool_schema("bash")],
        use_react=True,
        workflow_mode="execute",
        permission_mode="workspace-write",
        cwd=temp_dir,
    )

    assert native.section_names == react.section_names
    assert native.dynamic_section_names == react.dynamic_section_names
    assert "`native`" in native.content
    assert "`react`" in react.content
    assert "<tool_call>" in react.content
    assert "[bash: command=" in native.content
