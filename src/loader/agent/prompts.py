"""Compatibility wrappers for Loader prompt construction."""

from __future__ import annotations

from ..runtime.prompting import (
    MODE_GUIDANCE,
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    SystemPromptBuildResult,
    build_system_prompt,
    build_system_prompt_result,
    format_tool_descriptions,
    get_project_specific_tips,
)

__all__ = [
    "MODE_GUIDANCE",
    "SYSTEM_PROMPT_DYNAMIC_BOUNDARY",
    "SystemPromptBuildResult",
    "build_system_prompt",
    "build_system_prompt_result",
    "format_tool_descriptions",
    "get_project_specific_tips",
]
