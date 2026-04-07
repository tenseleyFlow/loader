"""Shell command execution tools."""

import asyncio
import shlex
from pathlib import Path
from typing import Any

from ..runtime.permissions import PermissionMode
from .base import ConfirmationRequired, Tool, ToolResult


class BashTool(Tool):
    """Execute bash commands."""

    required_permission = PermissionMode.DANGER_FULL_ACCESS
    OUTPUT_LIMIT = 50_000

    # Commands that are generally safe (read-only operations)
    SAFE_COMMANDS = {
        "ls", "cat", "head", "tail", "grep", "find", "pwd", "whoami", "date",
        "wc", "sort", "uniq", "diff", "file", "stat", "du", "df",
        "git status", "git log", "git diff", "git branch", "git show",
        "python --version", "node --version", "npm --version",
        "uv --version", "pip list", "pip show",
    }

    def __init__(self, timeout: float = 120.0, allowed_commands: list[str] | None = None):
        self.timeout = timeout
        self.allowed_commands = allowed_commands  # None means all allowed

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Execute a bash command and return the output. Use for git, npm, build tools, etc."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory to run the command in (default: current directory)",
                },
                "timeout": {
                    "type": "number",
                    "description": f"Timeout in seconds (default: {self.timeout})",
                    "default": self.timeout,
                },
            },
            "required": ["command"],
        }

    @property
    def is_destructive(self) -> bool:
        return True

    def _is_safe_command(self, command: str) -> bool:
        """Check if command is a known safe (read-only) command."""
        cmd = command.strip().lower()
        # Check exact matches and prefix matches
        for safe in self.SAFE_COMMANDS:
            if cmd == safe or cmd.startswith(safe + " "):
                return True
        return False

    def get_required_permission(self, **kwargs: Any) -> PermissionMode:
        """Classify one shell invocation by its mutability."""
        command = str(kwargs.get("command", ""))
        return self.classify_command_permission(command)

    def classify_command_permission(self, command: str) -> PermissionMode:
        """Classify a shell command into a runtime permission mode."""
        normalized = command.strip().lower()
        if not normalized:
            return PermissionMode.DANGER_FULL_ACCESS
        if self._is_safe_command(normalized):
            return PermissionMode.READ_ONLY

        danger_signals = (
            "sudo ",
            "chmod ",
            "chown ",
            "rm -",
            "mkfs",
            "dd ",
            "/etc/",
            "/usr/",
            "/bin/",
            "/sbin/",
            "/boot/",
            "/proc/",
            "/sys/",
        )
        if any(signal in normalized for signal in danger_signals):
            return PermissionMode.DANGER_FULL_ACCESS

        return PermissionMode.WORKSPACE_WRITE

    def check_confirmation(self, skip_confirmation: bool = False, **kwargs: Any) -> None:
        if skip_confirmation:
            return
        command = kwargs.get("command", "")
        # Safe commands don't need confirmation
        if self._is_safe_command(command):
            return
        raise ConfirmationRequired(
            tool_name=self.name,
            message=f"Run command: {command[:50]}{'...' if len(command) > 50 else ''}",
            details=command,
        )

    def _is_command_allowed(self, command: str) -> bool:
        """Check if command is in the allowed list."""
        if self.allowed_commands is None:
            return True

        # Extract the base command
        try:
            parts = shlex.split(command)
            if not parts:
                return False
            base_cmd = parts[0]
            return base_cmd in self.allowed_commands
        except ValueError:
            return False

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not self._is_command_allowed(command):
            return ToolResult(
                f"Command not allowed. Allowed commands: {self.allowed_commands}",
                is_error=True,
            )

        timeout = timeout or self.timeout
        resolved_cwd = None
        if cwd:
            resolved_cwd = str(Path(cwd).expanduser().resolve())

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=resolved_cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    f"Command timed out after {timeout}s",
                    is_error=True,
                )

            output_parts = []

            if stdout:
                stdout_text = stdout.decode("utf-8", errors="replace")
                output_parts.append(stdout_text)

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"[stderr]\n{stderr_text}")

            output = "\n".join(output_parts) if output_parts else "(no output)"

            # Truncate if too long
            truncated = len(output) > self.OUTPUT_LIMIT
            if truncated:
                output = output[: self.OUTPUT_LIMIT] + "\n\n... (output truncated)"
            else:
                truncated = False

            metadata = {
                "command": command,
                "cwd": resolved_cwd,
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
                "stderr": stderr.decode("utf-8", errors="replace") if stderr else "",
                "mutability": self.classify_command_permission(command).as_str(),
                "truncated": truncated,
                "output_limit": self.OUTPUT_LIMIT,
            }

            if process.returncode != 0:
                return ToolResult(
                    f"Exit code {process.returncode}\n{output}",
                    is_error=True,
                    metadata=metadata,
                )

            return ToolResult(output, metadata=metadata)

        except Exception as e:
            return ToolResult(
                f"Error executing command: {e}",
                is_error=True,
                metadata={"command": command, "cwd": resolved_cwd},
            )
