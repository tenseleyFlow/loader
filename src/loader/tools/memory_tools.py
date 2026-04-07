"""Native Loader tools for project memory and working notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runtime.memory import MemoryStore
from ..runtime.permissions import PermissionMode
from .base import Tool, ToolResult


class MemoryTool(Tool):
    """Shared base class for `.loader/` memory tools."""

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

    def store(self) -> MemoryStore:
        return MemoryStore(self.workspace_root or Path.cwd())


class ProjectMemoryReadTool(MemoryTool):
    """Read project memory."""

    required_permission = PermissionMode.READ_ONLY

    @property
    def name(self) -> str:
        return "project_memory_read"

    @property
    def description(self) -> str:
        return "Read project memory from .loader/project-memory.json."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": [
                        "all",
                        "techStack",
                        "build",
                        "conventions",
                        "structure",
                        "notes",
                        "directives",
                    ],
                    "description": "Optional project-memory section to read.",
                }
            },
        }

    async def execute(self, section: str = "all", **kwargs: Any) -> ToolResult:
        payload = self.store().read_project_memory(section=section or "all")
        return ToolResult(
            output=json.dumps(payload, indent=2, sort_keys=True),
            metadata={"section": section or "all", "payload": payload},
        )


class ProjectMemoryWriteTool(MemoryTool):
    """Write project memory."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    @property
    def name(self) -> str:
        return "project_memory_write"

    @property
    def description(self) -> str:
        return "Write or merge project memory in .loader/project-memory.json."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "object",
                    "description": "Memory object to write.",
                },
                "merge": {
                    "type": "boolean",
                    "description": "Merge with existing memory when true.",
                },
            },
            "required": ["memory"],
        }

    async def execute(
        self,
        memory: dict[str, Any],
        merge: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        payload = self.store().write_project_memory(memory, merge=merge)
        return ToolResult(
            output=json.dumps(payload, indent=2, sort_keys=True),
            metadata={"memory": payload, "merge": merge},
        )


class ProjectMemoryAddNoteTool(MemoryTool):
    """Append a categorized note to project memory."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    @property
    def name(self) -> str:
        return "project_memory_add_note"

    @property
    def description(self) -> str:
        return "Add a categorized note to project memory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["category", "content"],
        }

    async def execute(self, category: str, content: str, **kwargs: Any) -> ToolResult:
        payload = self.store().add_project_note(category, content)
        return ToolResult(
            output=json.dumps(payload, indent=2, sort_keys=True),
            metadata=payload,
        )


class ProjectMemoryAddDirectiveTool(MemoryTool):
    """Append a persistent directive to project memory."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    @property
    def name(self) -> str:
        return "project_memory_add_directive"

    @property
    def description(self) -> str:
        return "Add a durable directive to project memory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directive": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["high", "normal"],
                },
                "context": {"type": "string"},
            },
            "required": ["directive"],
        }

    async def execute(
        self,
        directive: str,
        priority: str = "normal",
        context: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        payload = self.store().add_project_directive(
            directive,
            priority=priority,
            context=context,
        )
        return ToolResult(
            output=json.dumps(payload, indent=2, sort_keys=True),
            metadata=payload,
        )


class NotepadReadTool(MemoryTool):
    """Read Loader's durable working notepad."""

    required_permission = PermissionMode.READ_ONLY

    @property
    def name(self) -> str:
        return "notepad_read"

    @property
    def description(self) -> str:
        return "Read the Loader notepad from .loader/notepad.md."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": ["all", "priority", "working", "manual"],
                }
            },
        }

    async def execute(self, section: str = "all", **kwargs: Any) -> ToolResult:
        payload = self.store().read_notepad(section=section or "all")
        return ToolResult(
            output=payload,
            metadata={"section": section or "all", "payload": payload},
        )


class NotepadWritePriorityTool(MemoryTool):
    """Replace the priority context section in the notepad."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    @property
    def name(self) -> str:
        return "notepad_write_priority"

    @property
    def description(self) -> str:
        return "Replace the priority-context section in .loader/notepad.md."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        }

    async def execute(self, content: str, **kwargs: Any) -> ToolResult:
        payload = self.store().write_notepad_priority(content)
        return ToolResult(output=payload, metadata={"content": payload})


class NotepadWriteWorkingTool(MemoryTool):
    """Append one timestamped working-memory note."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    @property
    def name(self) -> str:
        return "notepad_write_working"

    @property
    def description(self) -> str:
        return "Append a timestamped entry to the working-memory section."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        }

    async def execute(self, content: str, **kwargs: Any) -> ToolResult:
        payload = self.store().append_notepad_working(content)
        return ToolResult(output=payload, metadata={"content": payload})


class NotepadWriteManualTool(MemoryTool):
    """Append one manual note."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    @property
    def name(self) -> str:
        return "notepad_write_manual"

    @property
    def description(self) -> str:
        return "Append a manual note to .loader/notepad.md."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        }

    async def execute(self, content: str, **kwargs: Any) -> ToolResult:
        payload = self.store().append_notepad_manual(content)
        return ToolResult(output=payload, metadata={"content": payload})
