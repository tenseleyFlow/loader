"""File operation tools."""

import asyncio
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult, ConfirmationRequired


class ReadTool(Tool):
    """Read file contents."""

    @property
    def name(self) -> str:
        return "read"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Returns the file content with line numbers."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed)",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "default": 500,
                },
            },
            "required": ["file_path"],
        }

    async def execute(
        self,
        file_path: str,
        offset: int = 1,
        limit: int = 500,
        **kwargs: Any,
    ) -> ToolResult:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)

        if not path.is_file():
            return ToolResult(f"Not a file: {file_path}", is_error=True)

        try:
            content = await asyncio.to_thread(path.read_text)
            lines = content.splitlines()

            # Apply offset and limit
            start_idx = max(0, offset - 1)
            end_idx = start_idx + limit
            selected_lines = lines[start_idx:end_idx]

            # Add line numbers
            numbered = []
            for i, line in enumerate(selected_lines, start=offset):
                numbered.append(f"{i:4d}\t{line}")

            output = "\n".join(numbered)

            if end_idx < len(lines):
                output += f"\n\n... ({len(lines) - end_idx} more lines)"

            return ToolResult(output)
        except Exception as e:
            return ToolResult(f"Error reading file: {e}", is_error=True)


class WriteTool(Tool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "write"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates the file if it doesn't exist."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    @property
    def is_destructive(self) -> bool:
        return True

    def check_confirmation(self, skip_confirmation: bool = False, **kwargs: Any) -> None:
        if skip_confirmation:
            return
        file_path = kwargs.get("file_path", "")
        content = kwargs.get("content", "")
        raise ConfirmationRequired(
            tool_name=self.name,
            message=f"Write to file: {file_path}",
            details=f"{len(content)} bytes",
        )

    async def execute(
        self,
        file_path: str,
        content: str,
        **kwargs: Any,
    ) -> ToolResult:
        path = Path(file_path).expanduser().resolve()

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(path.write_text, content)

            return ToolResult(f"Successfully wrote {len(content)} bytes to {file_path}")
        except Exception as e:
            return ToolResult(f"Error writing file: {e}", is_error=True)


class EditTool(Tool):
    """Edit a file by replacing text."""

    @property
    def name(self) -> str:
        return "edit"

    @property
    def description(self) -> str:
        return "Edit a file by replacing old_string with new_string. The old_string must match exactly."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement string",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    @property
    def is_destructive(self) -> bool:
        return True

    def check_confirmation(self, skip_confirmation: bool = False, **kwargs: Any) -> None:
        if skip_confirmation:
            return
        file_path = kwargs.get("file_path", "")
        raise ConfirmationRequired(
            tool_name=self.name,
            message=f"Edit file: {file_path}",
            details="replace text",
        )

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        **kwargs: Any,
    ) -> ToolResult:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)

        try:
            content = await asyncio.to_thread(path.read_text)

            if old_string not in content:
                return ToolResult(
                    f"old_string not found in file. Make sure it matches exactly.",
                    is_error=True,
                )

            # Count occurrences
            count = content.count(old_string)
            if count > 1:
                return ToolResult(
                    f"old_string appears {count} times. Please provide more context to make it unique.",
                    is_error=True,
                )

            new_content = content.replace(old_string, new_string, 1)
            await asyncio.to_thread(path.write_text, new_content)

            return ToolResult(f"Successfully edited {file_path}")
        except Exception as e:
            return ToolResult(f"Error editing file: {e}", is_error=True)


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return "Find files matching a glob pattern (e.g., '**/*.py', 'src/*.ts')."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current directory)",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        **kwargs: Any,
    ) -> ToolResult:
        base_path = Path(path).expanduser().resolve()

        if not base_path.exists():
            return ToolResult(f"Directory not found: {path}", is_error=True)

        try:
            matches = list(base_path.glob(pattern))
            # Sort by modification time (newest first)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Limit results
            if len(matches) > 100:
                matches = matches[:100]
                output = "\n".join(str(p) for p in matches)
                output += f"\n\n... (showing first 100 of {len(matches)} matches)"
            else:
                output = "\n".join(str(p) for p in matches)

            if not matches:
                output = f"No files matching pattern: {pattern}"

            return ToolResult(output)
        except Exception as e:
            return ToolResult(f"Error searching files: {e}", is_error=True)
