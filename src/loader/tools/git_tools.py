"""Read-only git inspection tool for explore and main runtime use."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ..runtime.permissions import PermissionMode
from .base import Tool, ToolResult
from .fs_safety import resolve_workspace_path

READ_ONLY_ACTIONS = {
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "rev-parse",
}
DISALLOWED_GIT_TOKENS = {"|", "&", ";", ">", "<", "`", "$(", ")"}


class GitTool(Tool):
    """Inspect git state through a narrow read-only interface."""

    required_permission = PermissionMode.READ_ONLY
    OUTPUT_LIMIT = 50_000

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "git"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

    @property
    def description(self) -> str:
        return (
            "Inspect git state with read-only subcommands such as status, log, diff, "
            "show, branch, and rev-parse."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(READ_ONLY_ACTIONS),
                    "description": "Read-only git action to run.",
                },
                "args": {
                    "type": "array",
                    "description": "Additional git arguments for the selected action.",
                    "items": {"type": "string"},
                },
                "cwd": {
                    "type": "string",
                    "description": "Directory inside the workspace where git should run.",
                    "default": ".",
                },
                "timeout": {
                    "type": "number",
                    "description": f"Timeout in seconds (default: {self.timeout})",
                    "default": self.timeout,
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        args: list[str] | None = None,
        cwd: str = ".",
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        normalized_action = action.strip().lower()
        if normalized_action not in READ_ONLY_ACTIONS:
            return ToolResult(
                f"Unsupported read-only git action: {action}",
                is_error=True,
            )

        try:
            resolved_cwd = resolve_workspace_path(
                cwd or ".",
                workspace_root=self.workspace_root,
            )
        except FileNotFoundError:
            return ToolResult(f"Path not found: {cwd}", is_error=True)
        except PermissionError as exc:
            return ToolResult(f"Permission denied: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(f"Error resolving cwd: {exc}", is_error=True)

        if not resolved_cwd.is_dir():
            return ToolResult(f"Not a directory: {resolved_cwd}", is_error=True)

        extra_args = [str(value) for value in (args or [])]
        if any(_contains_disallowed_token(value) for value in extra_args):
            return ToolResult(
                "git args contain disallowed shell syntax; pass plain git arguments only",
                is_error=True,
            )

        effective_timeout = timeout or self.timeout
        command = ["git", normalized_action, *extra_args]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(resolved_cwd),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    f"git {normalized_action} timed out after {effective_timeout}s",
                    is_error=True,
                )
        except Exception as exc:
            return ToolResult(
                f"Error running git {normalized_action}: {exc}",
                is_error=True,
            )

        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
        output = stdout_text.strip() or stderr_text.strip() or "(no output)"

        truncated = len(output) > self.OUTPUT_LIMIT
        if truncated:
            output = output[: self.OUTPUT_LIMIT] + "\n\n... (output truncated)"

        metadata = {
            "action": normalized_action,
            "args": extra_args,
            "cwd": str(resolved_cwd),
            "command": " ".join(command),
            "exit_code": process.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "truncated": truncated,
            "output_limit": self.OUTPUT_LIMIT,
        }
        if process.returncode != 0:
            return ToolResult(
                f"git {normalized_action} exited with {process.returncode}\n{output}",
                is_error=True,
                metadata=metadata,
            )

        return ToolResult(output, metadata=metadata)


def _contains_disallowed_token(value: str) -> bool:
    return any(token in value for token in DISALLOWED_GIT_TOKENS)
