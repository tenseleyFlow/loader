"""Shell command execution tools."""

import asyncio
import shlex
from typing import Any

from .base import Tool, ToolResult, ConfirmationRequired


class BashTool(Tool):
    """Execute bash commands."""

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
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not self._is_command_allowed(command):
            return ToolResult(
                f"Command not allowed. Allowed commands: {self.allowed_commands}",
                is_error=True,
            )

        timeout = timeout or self.timeout

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
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
            if len(output) > 50000:
                output = output[:50000] + "\n\n... (output truncated)"

            if process.returncode != 0:
                return ToolResult(
                    f"Exit code {process.returncode}\n{output}",
                    is_error=True,
                )

            return ToolResult(output)

        except Exception as e:
            return ToolResult(f"Error executing command: {e}", is_error=True)
