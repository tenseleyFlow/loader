"""Runtime-owned rollback planning services."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto


class RollbackType(Enum):
    """Types of rollback actions."""

    FILE_RESTORE = auto()
    FILE_DELETE = auto()
    GIT_CHECKOUT = auto()
    GIT_STASH_POP = auto()
    COMMAND_UNDO = auto()
    NO_ROLLBACK = auto()


@dataclass
class RollbackAction:
    """A single rollback action."""

    type: RollbackType
    description: str
    file_path: str = ""
    original_content: str = ""
    undo_command: str = ""
    executed: bool = False


@dataclass
class RollbackPlan:
    """Plan for rolling back a series of actions."""

    actions: list[RollbackAction] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    modified_files: dict[str, str] = field(default_factory=dict)
    git_stashed: bool = False
    can_rollback: bool = True

    def add_file_creation(self, file_path: str) -> None:
        """Track a file that was created (can be deleted to rollback)."""

        self.created_files.append(file_path)
        self.actions.append(
            RollbackAction(
                type=RollbackType.FILE_DELETE,
                description=f"Delete created file: {file_path}",
                file_path=file_path,
            )
        )

    def add_file_modification(self, file_path: str, original_content: str) -> None:
        """Track a file modification (can restore original content)."""

        if file_path not in self.modified_files:
            self.modified_files[file_path] = original_content
            self.actions.append(
                RollbackAction(
                    type=RollbackType.FILE_RESTORE,
                    description=f"Restore original: {file_path}",
                    file_path=file_path,
                    original_content=original_content,
                )
            )

    def add_git_stash(self) -> None:
        """Track that we stashed git changes."""

        if not self.git_stashed:
            self.git_stashed = True
            self.actions.append(
                RollbackAction(
                    type=RollbackType.GIT_STASH_POP,
                    description="Restore stashed changes: git stash pop",
                )
            )

    def add_command_undo(self, description: str, undo_command: str) -> None:
        """Track a command that can be undone."""

        self.actions.append(
            RollbackAction(
                type=RollbackType.COMMAND_UNDO,
                description=description,
                undo_command=undo_command,
            )
        )

    def add_no_rollback(self, description: str) -> None:
        """Track an action that cannot be rolled back."""

        self.can_rollback = False
        self.actions.append(
            RollbackAction(
                type=RollbackType.NO_ROLLBACK,
                description=f"Cannot undo: {description}",
            )
        )

    def get_rollback_steps(self) -> list[str]:
        """Get human-readable rollback steps in reverse order."""

        steps = []
        for action in reversed(self.actions):
            if action.type == RollbackType.FILE_DELETE:
                steps.append(f"Delete: {action.file_path}")
            elif action.type == RollbackType.FILE_RESTORE:
                steps.append(f"Restore: {action.file_path}")
            elif action.type == RollbackType.GIT_CHECKOUT:
                steps.append(f"Git restore: {action.file_path}")
            elif action.type == RollbackType.GIT_STASH_POP:
                steps.append("Run: git stash pop")
            elif action.type == RollbackType.COMMAND_UNDO:
                steps.append(f"Run: {action.undo_command}")
            elif action.type == RollbackType.NO_ROLLBACK:
                steps.append(f"⚠ {action.description}")
        return steps

    def to_prompt(self) -> str:
        """Format rollback plan for display."""

        if not self.actions:
            return "No rollback actions recorded."

        lines = ["Rollback plan:"]
        for index, step in enumerate(self.get_rollback_steps(), 1):
            lines.append(f"  {index}. {step}")

        if not self.can_rollback:
            lines.append("\n⚠ Warning: Some actions cannot be undone!")

        return "\n".join(lines)


def is_destructive_tool(tool_name: str, tool_args: dict) -> bool:
    """Check if a tool call is potentially destructive."""

    if tool_name in {"write", "edit", "patch"}:
        return True

    if tool_name == "bash":
        command = tool_args.get("command", "").lower()
        destructive_patterns = [
            "rm ",
            "rm -",
            "rmdir",
            "mv ",
            "rename",
            "> ",
            ">>",
            "chmod",
            "chown",
            "git reset",
            "git checkout",
            "git clean",
            "git stash",
            "npm uninstall",
            "pip uninstall",
            "drop ",
            "delete ",
            "truncate",
        ]
        return any(pattern in command for pattern in destructive_patterns)

    return False


def get_undo_command(command: str) -> str | None:
    """Get the undo command for a bash command, if possible."""

    command_lower = command.lower().strip()

    if command_lower.startswith("mkdir "):
        dir_path = command.split("mkdir", 1)[1].strip().split()[0]
        return f"rmdir {dir_path}"

    if "git stash" in command_lower and "pop" not in command_lower:
        return "git stash pop"

    if "npm install " in command_lower or "npm i " in command_lower:
        parts = command.split()
        for index, part in enumerate(parts):
            if part in ("install", "i") and index + 1 < len(parts):
                package = parts[index + 1]
                if not package.startswith("-"):
                    return f"npm uninstall {package}"

    if "pip install " in command_lower or "pip3 install " in command_lower:
        parts = command.split()
        for index, part in enumerate(parts):
            if part == "install" and index + 1 < len(parts):
                package = parts[index + 1]
                if not package.startswith("-"):
                    return f"pip uninstall -y {package}"

    return None


async def create_rollback_plan_for_action(
    tool_name: str,
    tool_args: dict,
    read_file_func,
) -> RollbackAction | None:
    """Create a rollback action for a tool call."""

    if tool_name == "write":
        file_path = tool_args.get("file_path", "")
        if not file_path:
            return None

        if os.path.exists(file_path):
            try:
                original = await read_file_func(file_path)
                return RollbackAction(
                    type=RollbackType.FILE_RESTORE,
                    description=f"Restore original: {file_path}",
                    file_path=file_path,
                    original_content=original,
                )
            except Exception:
                return RollbackAction(
                    type=RollbackType.NO_ROLLBACK,
                    description=f"Could not backup: {file_path}",
                    file_path=file_path,
                )

        return RollbackAction(
            type=RollbackType.FILE_DELETE,
            description=f"Delete created file: {file_path}",
            file_path=file_path,
        )

    if tool_name == "edit":
        file_path = tool_args.get("file_path", "")
        if not file_path:
            return None

        try:
            original = await read_file_func(file_path)
            return RollbackAction(
                type=RollbackType.FILE_RESTORE,
                description=f"Restore original: {file_path}",
                file_path=file_path,
                original_content=original,
            )
        except Exception:
            return RollbackAction(
                type=RollbackType.NO_ROLLBACK,
                description=f"Could not backup: {file_path}",
                file_path=file_path,
            )

    if tool_name == "patch":
        file_path = tool_args.get("file_path", "")
        if not file_path:
            return None

        try:
            original = await read_file_func(file_path)
            return RollbackAction(
                type=RollbackType.FILE_RESTORE,
                description=f"Restore original: {file_path}",
                file_path=file_path,
                original_content=original,
            )
        except Exception:
            return RollbackAction(
                type=RollbackType.NO_ROLLBACK,
                description=f"Could not backup: {file_path}",
                file_path=file_path,
            )

    if tool_name == "bash":
        command = tool_args.get("command", "")
        undo = get_undo_command(command)
        if undo:
            return RollbackAction(
                type=RollbackType.COMMAND_UNDO,
                description=f"Undo with: {undo}",
                undo_command=undo,
            )
        if is_destructive_tool(tool_name, tool_args):
            return RollbackAction(
                type=RollbackType.NO_ROLLBACK,
                description=f"Cannot undo: {command[:50]}...",
            )

    return None


async def execute_rollback(plan: RollbackPlan, write_file_func, run_command_func) -> list[str]:
    """Execute a rollback plan."""

    results = []

    for action in reversed(plan.actions):
        if action.executed:
            continue

        try:
            if action.type == RollbackType.FILE_DELETE:
                if os.path.exists(action.file_path):
                    os.remove(action.file_path)
                    results.append(f"✓ Deleted: {action.file_path}")
                    action.executed = True

            elif action.type == RollbackType.FILE_RESTORE:
                await write_file_func(action.file_path, action.original_content)
                results.append(f"✓ Restored: {action.file_path}")
                action.executed = True

            elif action.type == RollbackType.COMMAND_UNDO:
                await run_command_func(action.undo_command)
                results.append(f"✓ Ran: {action.undo_command}")
                action.executed = True

            elif action.type == RollbackType.GIT_STASH_POP:
                await run_command_func("git stash pop")
                results.append("✓ Restored git stash")
                action.executed = True

            elif action.type == RollbackType.NO_ROLLBACK:
                results.append(f"⚠ Skipped (no rollback): {action.description}")

        except Exception as exc:
            results.append(f"✗ Failed {action.description}: {exc}")

    return results
