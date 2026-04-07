"""Code search tools."""

import asyncio
import re
from pathlib import Path
from typing import Any

from ..runtime.permissions import PermissionMode
from .base import Tool, ToolResult
from .fs_safety import detect_binary_file, resolve_workspace_path


class GrepTool(Tool):
    """Search for patterns in files."""

    required_permission = PermissionMode.READ_ONLY

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )

    @property
    def name(self) -> str:
        return "grep"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

    @property
    def description(self) -> str:
        return "Search for a regex pattern in files. Returns matching lines with file paths and line numbers."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: current directory)",
                    "default": ".",
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern for files to include (e.g., '*.py')",
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines before and after match",
                    "default": 0,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 50,
                },
            },
            "required": ["pattern"],
        }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
        context: int = 0,
        max_results: int = 50,
        **kwargs: Any,
    ) -> ToolResult:
        try:
            base_path = resolve_workspace_path(
                path,
                workspace_root=self.workspace_root,
            )
        except FileNotFoundError:
            return ToolResult(f"Path not found: {path}", is_error=True)
        except PermissionError as exc:
            return ToolResult(f"Permission denied: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(f"Error resolving search path: {exc}", is_error=True)

        if not base_path.exists():
            return ToolResult(f"Path not found: {path}", is_error=True)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(f"Invalid regex pattern: {e}", is_error=True)

        # Collect files to search
        files: list[Path] = []
        if base_path.is_file():
            files = [base_path]
        else:
            glob_pattern = include or "**/*"
            for f in base_path.glob(glob_pattern):
                if f.is_file():
                    # Skip binary files, hidden files, common non-code directories
                    if f.name.startswith("."):
                        continue
                    if any(part.startswith(".") for part in f.parts):
                        continue
                    if any(part in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                           for part in f.parts):
                        continue
                    try:
                        if detect_binary_file(f):
                            continue
                    except OSError:
                        continue
                    files.append(f)

        results: list[str] = []
        total_matches = 0

        for file_path in files:
            if total_matches >= max_results:
                break

            try:
                content = await asyncio.to_thread(file_path.read_text, errors="ignore")
                lines = content.splitlines()

                for i, line in enumerate(lines, 1):
                    if total_matches >= max_results:
                        break

                    if regex.search(line):
                        total_matches += 1

                        # Build result with context
                        result_lines = []

                        # Context before
                        for ctx_i in range(max(1, i - context), i):
                            if ctx_i > 0:
                                result_lines.append(f"  {ctx_i}: {lines[ctx_i - 1]}")

                        # Match line (highlighted)
                        result_lines.append(f"> {i}: {line}")

                        # Context after
                        for ctx_i in range(i + 1, min(len(lines) + 1, i + context + 1)):
                            result_lines.append(f"  {ctx_i}: {lines[ctx_i - 1]}")

                        results.append(f"{file_path}:\n" + "\n".join(result_lines))

            except Exception:
                # Skip files we can't read
                continue

        if not results:
            return ToolResult(f"No matches found for pattern: {pattern}")

        output = "\n\n".join(results)
        if total_matches >= max_results:
            output += f"\n\n... (showing first {max_results} matches)"

        return ToolResult(
            output,
            metadata={
                "path": str(base_path),
                "num_files": len(files),
                "num_matches": total_matches,
                "max_results": max_results,
            },
        )
