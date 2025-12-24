"""Error recovery system for the agent.

Provides intelligent retry logic with adaptation when tools fail.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ErrorCategory(Enum):
    """Categories of errors for recovery strategies."""
    FILE_NOT_FOUND = auto()
    PERMISSION_DENIED = auto()
    SYNTAX_ERROR = auto()
    COMMAND_NOT_FOUND = auto()
    TIMEOUT = auto()
    INVALID_ARGUMENTS = auto()
    NETWORK_ERROR = auto()
    UNKNOWN = auto()


@dataclass
class ToolAttempt:
    """Record of a single tool execution attempt."""
    tool_name: str
    arguments: dict[str, Any]
    error: str
    category: ErrorCategory


@dataclass
class RecoveryContext:
    """Tracks recovery state for a tool execution."""
    original_tool: str
    original_args: dict[str, Any]
    attempts: list[ToolAttempt] = field(default_factory=list)
    max_retries: int = 3

    def add_attempt(self, tool_name: str, args: dict[str, Any], error: str) -> None:
        """Record an attempted tool execution."""
        category = categorize_error(error)
        self.attempts.append(ToolAttempt(
            tool_name=tool_name,
            arguments=args,
            error=error,
            category=category,
        ))

    def can_retry(self) -> bool:
        """Check if more retries are allowed."""
        return len(self.attempts) < self.max_retries

    def attempts_summary(self) -> str:
        """Summarize what's been tried for the LLM."""
        if not self.attempts:
            return ""

        lines = ["Previous attempts:"]
        for i, attempt in enumerate(self.attempts, 1):
            args_str = ", ".join(f"{k}={v!r}" for k, v in attempt.arguments.items())
            lines.append(f"{i}. {attempt.tool_name}({args_str})")
            lines.append(f"   Error: {attempt.error}")
        return "\n".join(lines)

    def was_tried(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if this exact tool+args combination was already tried."""
        for attempt in self.attempts:
            if attempt.tool_name == tool_name and attempt.arguments == args:
                return True
        return False


def categorize_error(error_message: str) -> ErrorCategory:
    """Categorize an error message for recovery strategy selection."""
    error_lower = error_message.lower()

    if any(x in error_lower for x in ["no such file", "file not found", "does not exist"]):
        return ErrorCategory.FILE_NOT_FOUND

    if any(x in error_lower for x in ["permission denied", "access denied", "not permitted"]):
        return ErrorCategory.PERMISSION_DENIED

    if any(x in error_lower for x in ["syntax error", "invalid syntax", "parse error"]):
        return ErrorCategory.SYNTAX_ERROR

    if any(x in error_lower for x in ["command not found", "not recognized", "no such command"]):
        return ErrorCategory.COMMAND_NOT_FOUND

    if any(x in error_lower for x in ["timeout", "timed out"]):
        return ErrorCategory.TIMEOUT

    if any(x in error_lower for x in ["invalid argument", "missing required", "expected"]):
        return ErrorCategory.INVALID_ARGUMENTS

    if any(x in error_lower for x in ["network", "connection", "unreachable"]):
        return ErrorCategory.NETWORK_ERROR

    return ErrorCategory.UNKNOWN


def get_recovery_hints(category: ErrorCategory, tool_name: str) -> str:
    """Get hints for recovering from a specific error category."""
    hints = {
        ErrorCategory.FILE_NOT_FOUND: [
            "Check if the file path is correct (use glob to search for it)",
            "The file might be in a different directory",
            "Check for typos in the filename",
            "List the directory contents first",
        ],
        ErrorCategory.PERMISSION_DENIED: [
            "The file might be read-only or owned by another user",
            "Try reading the file instead of modifying it",
            "Check if the directory exists and is writable",
        ],
        ErrorCategory.SYNTAX_ERROR: [
            "Review the edit content for syntax errors",
            "Check that the old_string matches exactly (including whitespace)",
            "Read the file first to verify current contents",
        ],
        ErrorCategory.COMMAND_NOT_FOUND: [
            "Check if the command is installed",
            "Try using the full path to the command",
            "Use a different command that achieves the same goal",
        ],
        ErrorCategory.TIMEOUT: [
            "The operation is taking too long",
            "Try a simpler or more targeted operation",
            "Break the task into smaller steps",
        ],
        ErrorCategory.INVALID_ARGUMENTS: [
            "Review the tool parameters",
            "Check that required arguments are provided",
            "Verify argument types and formats",
        ],
        ErrorCategory.NETWORK_ERROR: [
            "Network operations may be unavailable",
            "Try an offline alternative if possible",
        ],
        ErrorCategory.UNKNOWN: [
            "Try a different approach",
            "Read related files for more context",
            "Break the task into smaller steps",
        ],
    }

    category_hints = hints.get(category, hints[ErrorCategory.UNKNOWN])

    # Add tool-specific hints
    if tool_name == "edit" and category == ErrorCategory.FILE_NOT_FOUND:
        category_hints.insert(0, "Use 'write' instead of 'edit' to create a new file")

    if tool_name == "bash" and category == ErrorCategory.COMMAND_NOT_FOUND:
        category_hints.insert(0, "Check available commands with 'which' or 'type'")

    return "\n".join(f"- {hint}" for hint in category_hints)


RECOVERY_PROMPT = """The tool execution failed. Analyze the error and try an alternative approach.

Tool: {tool_name}
Arguments: {args}
Error: {error}
Category: {category}

{attempts_summary}

Recovery hints:
{hints}

IMPORTANT:
- Do NOT repeat the exact same tool call that just failed
- Try a different approach or gather more information first
- If you've tried {attempt_count}/{max_retries} times, consider explaining the issue to the user

What would you like to try instead?"""


def format_recovery_prompt(
    context: RecoveryContext,
    tool_name: str,
    args: dict[str, Any],
    error: str,
) -> str:
    """Format a prompt asking the LLM to recover from an error."""
    category = categorize_error(error)
    hints = get_recovery_hints(category, tool_name)
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())

    return RECOVERY_PROMPT.format(
        tool_name=tool_name,
        args=args_str,
        error=error,
        category=category.name.replace("_", " ").title(),
        attempts_summary=context.attempts_summary(),
        hints=hints,
        attempt_count=len(context.attempts),
        max_retries=context.max_retries,
    )


def format_failure_message(context: RecoveryContext) -> str:
    """Format a message when all retries are exhausted."""
    lines = [
        f"Failed to complete the operation after {len(context.attempts)} attempts.",
        "",
        "What was tried:",
    ]

    for i, attempt in enumerate(context.attempts, 1):
        args_str = ", ".join(f"{k}={v!r}" for k, v in attempt.arguments.items())
        lines.append(f"{i}. {attempt.tool_name}({args_str})")
        lines.append(f"   Error: {attempt.error}")

    lines.extend([
        "",
        "You may need to:",
        "- Manually check the file/directory structure",
        "- Verify permissions",
        "- Try a completely different approach",
    ])

    return "\n".join(lines)
