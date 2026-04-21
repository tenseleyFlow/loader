"""File operation tools."""

import asyncio
from pathlib import Path
from typing import Any

from ..runtime.permissions import PermissionMode
from ..utils.file_mutations import build_file_mutation_preview_dict
from .base import ConfirmationRequired, Tool, ToolResult
from .fs_safety import (
    StructuredPatchHunk,
    apply_structured_patch,
    ensure_safe_to_read,
    ensure_safe_to_write,
    make_structured_patch,
    parse_unified_diff_patch,
    resolve_workspace_path,
)

_GLOB_MAGIC_CHARS = "*?["


def _has_glob_magic(segment: str) -> bool:
    """Return whether one path segment contains glob syntax."""

    return any(char in segment for char in _GLOB_MAGIC_CHARS)


def _resolve_glob_base_and_pattern(
    pattern: str,
    path: str,
) -> tuple[Path, str]:
    """Resolve glob inputs, including `~`/absolute patterns outside the cwd."""

    expanded_pattern = Path(pattern).expanduser()
    pattern_is_explicit_path = pattern.startswith("~") or expanded_pattern.is_absolute()

    if not pattern_is_explicit_path:
        base_path = resolve_workspace_path(path, workspace_root=None)
        return base_path, pattern

    base_parts: list[str] = []
    pattern_parts: list[str] = []
    saw_glob = False
    for part in expanded_pattern.parts:
        if saw_glob or _has_glob_magic(part):
            saw_glob = True
            pattern_parts.append(part)
        else:
            base_parts.append(part)

    if not pattern_parts:
        if expanded_pattern.name:
            pattern_parts = [expanded_pattern.name]
            base_parts = list(expanded_pattern.parent.parts)
        else:
            pattern_parts = ["*"]

    raw_base = str(Path(*base_parts)) if base_parts else expanded_pattern.anchor or "."
    base_path = resolve_workspace_path(raw_base, workspace_root=None)
    return base_path, "/".join(pattern_parts)


class ReadTool(Tool):
    """Read file contents."""

    required_permission = PermissionMode.READ_ONLY

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )

    @property
    def name(self) -> str:
        return "read"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

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
        try:
            # Reads are safe — don't enforce workspace boundary
            path = resolve_workspace_path(
                file_path,
                workspace_root=None,
            )
        except FileNotFoundError:
            return ToolResult(f"File not found: {file_path}", is_error=True)
        except Exception as exc:
            return ToolResult(f"Error resolving file path: {exc}", is_error=True)

        if not path.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)

        if not path.is_file():
            return ToolResult(f"Not a file: {file_path}", is_error=True)

        try:
            ensure_safe_to_read(path)
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

            return ToolResult(
                output,
                metadata={
                    "file_path": str(path),
                    "line_count": len(lines),
                    "offset": offset,
                    "limit": limit,
                },
            )
        except Exception as e:
            return ToolResult(f"Error reading file: {e}", is_error=True)


class WriteTool(Tool):
    """Write content to a file."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )
        self._pending_escape_approvals: set[str] = set()

    @property
    def name(self) -> str:
        return "write"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

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
            preview=build_file_mutation_preview_dict(self.name, tool_args=kwargs),
        )

    async def execute(
        self,
        file_path: str,
        content: str,
        **kwargs: Any,
    ) -> ToolResult:
        kwargs.pop("_skip_confirmation", None)
        try:
            ensure_safe_to_write(content)
            path = resolve_workspace_path(
                file_path,
                workspace_root=self.workspace_root,
                allow_missing=True,
            )
        except PermissionError:
            resolved = Path(file_path).expanduser().resolve()
            key = str(resolved)
            if key in self._pending_escape_approvals:
                self._pending_escape_approvals.discard(key)
                path = resolved
            else:
                self._pending_escape_approvals.add(key)
                raise ConfirmationRequired(
                    tool_name=self.name,
                    message=f"Write outside workspace: {file_path}",
                    details=f"Target is outside the workspace root ({self.workspace_root})",
                    preview=build_file_mutation_preview_dict(
                        self.name,
                        tool_args={"file_path": file_path, "content": content},
                    ),
                )
        except Exception as exc:
            return ToolResult(f"Error writing file: {exc}", is_error=True)

        try:
            original_content = ""
            if path.exists():
                ensure_safe_to_read(path)
                original_content = await asyncio.to_thread(path.read_text)

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(path.write_text, content)

            structured_patch = [
                hunk.to_dict()
                for hunk in make_structured_patch(original_content, content)
            ]
            metadata = {
                "kind": "update" if original_content else "create",
                "file_path": str(path),
                "content": content,
                "original_file": original_content or None,
                "structured_patch": structured_patch,
                "bytes_written": len(content.encode("utf-8")),
            }
            return ToolResult(
                f"Successfully wrote {len(content)} bytes to {path}",
                metadata=metadata,
            )
        except Exception as e:
            return ToolResult(f"Error writing file: {e}", is_error=True)


class EditTool(Tool):
    """Edit a file by replacing text."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )
        self._pending_escape_approvals: set[str] = set()

    @property
    def name(self) -> str:
        return "edit"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_string with new_string. "
            "The old_string must match exactly."
        )

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
            preview=build_file_mutation_preview_dict(self.name, tool_args=kwargs),
        )

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        **kwargs: Any,
    ) -> ToolResult:
        kwargs.pop("_skip_confirmation", None)
        try:
            path = resolve_workspace_path(
                file_path,
                workspace_root=self.workspace_root,
            )
        except FileNotFoundError:
            return ToolResult(f"File not found: {file_path}", is_error=True)
        except PermissionError:
            resolved = Path(file_path).expanduser().resolve()
            key = str(resolved)
            if key in self._pending_escape_approvals:
                self._pending_escape_approvals.discard(key)
                path = resolved
            else:
                self._pending_escape_approvals.add(key)
                raise ConfirmationRequired(
                    tool_name=self.name,
                    message=f"Edit outside workspace: {file_path}",
                    details=f"Target is outside the workspace root ({self.workspace_root})",
                    preview=build_file_mutation_preview_dict(
                        self.name,
                        tool_args={
                            "file_path": file_path,
                            "old_string": old_string,
                            "new_string": new_string,
                        },
                    ),
                )
        except Exception as exc:
            return ToolResult(f"Error resolving file path: {exc}", is_error=True)

        if not path.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)

        try:
            ensure_safe_to_read(path)
            content = await asyncio.to_thread(path.read_text)

            if old_string not in content:
                return ToolResult(
                    "old_string not found in file. Make sure it matches exactly.",
                    is_error=True,
                )

            # Count occurrences
            count = content.count(old_string)
            if count > 1:
                return ToolResult(
                    "old_string appears "
                    f"{count} times. Please provide more context to make it unique.",
                    is_error=True,
                )

            new_content = content.replace(old_string, new_string, 1)
            ensure_safe_to_write(new_content)
            await asyncio.to_thread(path.write_text, new_content)

            structured_patch = [
                hunk.to_dict()
                for hunk in make_structured_patch(content, new_content)
            ]
            return ToolResult(
                f"Successfully edited {path}",
                metadata={
                    "file_path": str(path),
                    "old_string": old_string,
                    "new_string": new_string,
                    "original_file": content,
                    "structured_patch": structured_patch,
                },
            )
        except Exception as e:
            return ToolResult(f"Error editing file: {e}", is_error=True)


class PatchTool(Tool):
    """Edit a file by applying structured patch hunks."""

    required_permission = PermissionMode.WORKSPACE_WRITE

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )
        self._pending_escape_approvals: set[str] = set()

    @property
    def name(self) -> str:
        return "patch"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

    @property
    def description(self) -> str:
        return (
            "Apply structured patch hunks to a file. Prefer this for larger "
            "or multi-line edits where exact old/new string replacement is brittle. "
            "A raw unified diff string is also accepted via `patch`."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to patch",
                },
                "hunks": {
                    "type": "array",
                    "description": "Structured patch hunks to apply in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_start": {"type": "integer"},
                            "old_lines": {"type": "integer"},
                            "new_start": {"type": "integer"},
                            "new_lines": {"type": "integer"},
                            "lines": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "old_start",
                            "old_lines",
                            "new_start",
                            "new_lines",
                            "lines",
                        ],
                    },
                },
                "patch": {
                    "type": "string",
                    "description": (
                        "Optional unified diff patch string. Loader will parse this "
                        "into structured hunks when possible."
                    ),
                },
            },
            "required": ["file_path"],
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
            message=f"Patch file: {file_path}",
            details="apply structured patch hunks",
            preview=build_file_mutation_preview_dict(self.name, tool_args=kwargs),
        )

    async def execute(
        self,
        file_path: str,
        hunks: list[dict[str, Any]] | None = None,
        patch: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        kwargs.pop("_skip_confirmation", None)
        try:
            path = resolve_workspace_path(
                file_path,
                workspace_root=self.workspace_root,
            )
        except FileNotFoundError:
            return ToolResult(f"File not found: {file_path}", is_error=True)
        except PermissionError:
            resolved = Path(file_path).expanduser().resolve()
            key = str(resolved)
            if key in self._pending_escape_approvals:
                self._pending_escape_approvals.discard(key)
                path = resolved
            else:
                self._pending_escape_approvals.add(key)
                raise ConfirmationRequired(
                    tool_name=self.name,
                    message=f"Patch outside workspace: {file_path}",
                    details=f"Target is outside the workspace root ({self.workspace_root})",
                    preview=build_file_mutation_preview_dict(
                        self.name,
                        tool_args={"file_path": file_path, "hunks": hunks},
                    ),
                )

        except Exception as exc:
            return ToolResult(f"Error resolving file path: {exc}", is_error=True)

        if not path.exists():
            return ToolResult(f"File not found: {file_path}", is_error=True)

        try:
            ensure_safe_to_read(path)
            original_content = await asyncio.to_thread(path.read_text)
            original_lines = original_content.splitlines()
            raw_patch = patch or kwargs.get("diff") or kwargs.get("patch_text")
            parsed_hunks: list[StructuredPatchHunk]
            if hunks:
                parsed_hunks = [
                    StructuredPatchHunk.from_dict_with_original(
                        hunk,
                        original_lines=original_lines,
                    )
                    for hunk in hunks
                ]
            elif isinstance(raw_patch, str) and raw_patch.strip():
                parsed_hunks = parse_unified_diff_patch(raw_patch)
            else:
                parsed_hunks = []
            if not parsed_hunks:
                raise ValueError("hunks must not be empty")
            updated_content = apply_structured_patch(original_content, parsed_hunks)
            ensure_safe_to_write(updated_content)
            await asyncio.to_thread(path.write_text, updated_content)
            structured_patch = [hunk.to_dict() for hunk in parsed_hunks]
            return ToolResult(
                f"Successfully patched {path}",
                metadata={
                    "file_path": str(path),
                    "original_file": original_content,
                    "content": updated_content,
                    "structured_patch": structured_patch,
                    "bytes_written": len(updated_content.encode("utf-8")),
                },
            )
        except Exception as exc:
            return ToolResult(f"Error patching file: {exc}", is_error=True)


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    required_permission = PermissionMode.READ_ONLY

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )

    @property
    def name(self) -> str:
        return "glob"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

    @property
    def description(self) -> str:
        return (
            "Find files matching a glob pattern (e.g., '**/*.py', 'src/*.ts'). "
            "For external directories, prefer path='~/Loader/animals' with "
            "pattern='*.html'; absolute or '~'-prefixed patterns are also accepted."
        )

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
        try:
            # Glob is read-only — don't enforce workspace boundary
            base_path, effective_pattern = _resolve_glob_base_and_pattern(pattern, path)
        except FileNotFoundError:
            return ToolResult(f"Directory not found: {path}", is_error=True)
        except Exception as exc:
            return ToolResult(f"Error resolving directory: {exc}", is_error=True)

        if not base_path.exists():
            missing_target = path if path != "." else str(base_path)
            return ToolResult(f"Directory not found: {missing_target}", is_error=True)

        try:
            matches = list(base_path.glob(effective_pattern))
            # Sort by modification time (newest first)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Limit results
            total_matches = len(matches)
            truncated = total_matches > 100
            if truncated:
                matches = matches[:100]
                output = "\n".join(str(p) for p in matches)
                output += f"\n\n... (showing first 100 of {total_matches} matches)"
            else:
                output = "\n".join(str(p) for p in matches)

            if not matches:
                output = f"No files matching pattern: {pattern}"

            return ToolResult(
                output,
                metadata={
                    "base_path": str(base_path),
                    "effective_pattern": effective_pattern,
                    "requested_pattern": pattern,
                    "num_files": len(matches),
                    "truncated": truncated if "truncated" in locals() else False,
                },
            )
        except Exception as e:
            return ToolResult(f"Error searching files: {e}", is_error=True)
