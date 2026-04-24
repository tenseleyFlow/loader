"""Runtime-owned recovery state and retry guidance services."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .safeguard_services import extract_shell_text_rewrite_target


class ErrorCategory(Enum):
    """Categories of errors for recovery strategies."""

    FILE_NOT_FOUND = auto()
    PERMISSION_DENIED = auto()
    DISK_FULL = auto()
    PATH_TOO_LONG = auto()

    SYNTAX_ERROR = auto()
    TYPE_ERROR = auto()
    IMPORT_ERROR = auto()

    COMMAND_NOT_FOUND = auto()
    SCRIPT_NOT_FOUND = auto()

    MISSING_DEPENDENCY = auto()
    VERSION_MISMATCH = auto()

    BUILD_ERROR = auto()
    LINT_ERROR = auto()
    TEST_FAILURE = auto()

    TIMEOUT = auto()
    OUT_OF_MEMORY = auto()
    PORT_IN_USE = auto()
    PROCESS_ERROR = auto()

    NETWORK_ERROR = auto()
    CONNECTION_REFUSED = auto()
    AUTH_ERROR = auto()

    GIT_CONFLICT = auto()
    GIT_NOT_REPO = auto()
    GIT_DIRTY = auto()

    CONFIG_ERROR = auto()
    INVALID_ARGUMENTS = auto()

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
    successful_steps: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    max_retries: int = 3

    def add_attempt(self, tool_name: str, args: dict[str, Any], error: str) -> None:
        """Record an attempted tool execution."""

        category = categorize_error(error)
        self.attempts.append(
            ToolAttempt(
                tool_name=tool_name,
                arguments=args,
                error=error,
                category=category,
            )
        )

    def can_retry(self) -> bool:
        """Check if more retries are allowed."""

        return len(self.attempts) < self.max_retries

    def note_success(self, tool_name: str, args: dict[str, Any]) -> None:
        """Track successful diagnostic steps taken during recovery."""

        self.successful_steps.append((tool_name, dict(args)))

    def attempts_summary(self) -> str:
        """Summarize what's been tried for the LLM."""

        if not self.attempts:
            return ""

        lines = ["Previous attempts:"]
        for index, attempt in enumerate(self.attempts, 1):
            args_str = ", ".join(
                f"{key}={value!r}" for key, value in attempt.arguments.items()
            )
            lines.append(f"{index}. {attempt.tool_name}({args_str})")
            lines.append(f"   Error: {attempt.error}")
        return "\n".join(lines)

    def was_tried(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if this exact tool and args combination was already tried."""

        for attempt in self.attempts:
            if attempt.tool_name == tool_name and attempt.arguments == args:
                return True
        return False

    def is_similar_attempt(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if this attempt is effectively the same as a previous one."""

        if tool_name != "bash":
            return self.was_tried(tool_name, args)

        new_cmd = args.get("command", "")
        new_cmd_normalized = self._normalize_command(new_cmd)

        for attempt in self.attempts:
            if attempt.tool_name != "bash":
                continue

            old_cmd = attempt.arguments.get("command", "")
            old_cmd_normalized = self._normalize_command(old_cmd)

            if new_cmd_normalized == old_cmd_normalized:
                return True

        return False

    def is_related_failure(self, tool_name: str, args: dict[str, Any], error: str) -> bool:
        """Decide whether a new failure belongs to the current recovery episode."""

        if not self.attempts:
            return tool_name == self.original_tool

        new_category = categorize_error(error)
        root_category = self.attempts[0].category
        if new_category != root_category:
            return False

        current_path = self._extract_primary_path(args)
        original_path = self._extract_primary_path(self.original_args)
        if current_path and original_path:
            if self._same_parent_directory(current_path, original_path):
                return True
            if current_path == original_path:
                return True
            return False

        if tool_name == "bash" and self.original_tool == "bash":
            return any(
                attempt.tool_name == "bash"
                and self._normalize_command(attempt.arguments.get("command", ""))
                == self._normalize_command(args.get("command", ""))
                for attempt in self.attempts
            )

        return tool_name == self.original_tool

    def should_clear_after_success(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Decide when a successful tool execution resolves the recovery episode."""

        if tool_name in {"write", "edit", "patch"}:
            return True

        if tool_name == "bash":
            command = str(args.get("command", ""))
            if extract_shell_text_rewrite_target(command) is not None:
                return True
            mutating_tokens = (
                "git commit",
                "git add",
                "mv ",
                "cp ",
                "rm ",
                "mkdir ",
                "touch ",
                "sed -i",
                "perl -pi",
                "python -c",
                "python3 -c",
            )
            return any(token in command for token in mutating_tokens)

        return False

    @staticmethod
    def _extract_primary_path(args: dict[str, Any]) -> str | None:
        for key in ("file_path", "path", "filepath", "file", "filename", "directory"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _same_parent_directory(left: str, right: str) -> bool:
        try:
            return Path(left).parent == Path(right).parent
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_command(cmd: str) -> str:
        """Normalize a shell command for comparison."""

        import re

        cmd = re.sub(r'^cd\s+[^\s]+\s*&&\s*', '', cmd)
        cmd = re.sub(r'\bnpm run start\b', 'npm start', cmd)
        cmd = re.sub(r'\bnpm run serve\b', 'npm serve', cmd)
        cmd = re.sub(r'\bnpm run dev\b', 'npm dev', cmd)
        cmd = re.sub(r'\bpython3\b', 'python', cmd)
        cmd = ' '.join(cmd.split())
        return cmd.strip()


def categorize_error(error_message: str) -> ErrorCategory:
    """Categorize an error message for recovery strategy selection."""

    error_lower = error_message.lower()

    if any(
        token in error_lower
        for token in [
            "merge conflict",
            "conflict",
            "unmerged paths",
            "fix conflicts",
            "both modified",
        ]
    ):
        return ErrorCategory.GIT_CONFLICT

    if any(
        token in error_lower
        for token in [
            "not a git repository",
            "fatal: not a git",
            "not in a git directory",
        ]
    ):
        return ErrorCategory.GIT_NOT_REPO

    if any(
        token in error_lower
        for token in [
            "uncommitted changes",
            "working tree not clean",
            "please commit or stash",
            "local changes would be overwritten",
            "changes not staged",
        ]
    ):
        return ErrorCategory.GIT_DIRTY

    if any(
        token in error_lower
        for token in [
            "missing script",
            "npm err! missing script",
            "script not found",
            "no script",
            "error: script",
            "yarn run error",
            "make: *** no rule to make target",
            "no such task",
            "task not found",
        ]
    ):
        return ErrorCategory.SCRIPT_NOT_FOUND

    if any(
        token in error_lower
        for token in [
            "address already in use",
            "port already in use",
            "eaddrinuse",
            "bind: address already in use",
            "port is already allocated",
        ]
    ):
        return ErrorCategory.PORT_IN_USE

    if any(
        token in error_lower
        for token in [
            "connection refused",
            "econnrefused",
            "could not connect",
            "unable to connect",
            "service unavailable",
            "no such host",
        ]
    ):
        return ErrorCategory.CONNECTION_REFUSED

    if any(
        token in error_lower
        for token in [
            "unauthorized",
            "403 forbidden",
            "401 unauthorized",
            "authentication failed",
            "invalid credentials",
            "invalid token",
            "token expired",
        ]
    ):
        return ErrorCategory.AUTH_ERROR

    if any(
        token in error_lower
        for token in [
            "version mismatch",
            "incompatible version",
            "requires node",
            "requires python",
            "unsupported version",
            "engine requirements",
            "peer dep",
            "peer dependency",
        ]
    ):
        return ErrorCategory.VERSION_MISMATCH

    if any(
        token in error_lower
        for token in [
            "test failed",
            "tests failed",
            "failing tests",
            "assertion error",
            "assertionerror",
            "expected",
            "to equal",
            "to be",
            "test suite failed",
        ]
    ):
        return ErrorCategory.TEST_FAILURE

    if any(
        token in error_lower
        for token in [
            "eslint",
            "pylint",
            "flake8",
            "ruff",
            "lint error",
            "linting failed",
            "style violation",
            "formatting error",
        ]
    ):
        return ErrorCategory.LINT_ERROR

    if any(
        token in error_lower
        for token in [
            "typeerror",
            "type error",
            "is not a function",
            "is not defined",
            "undefined is not",
            "null is not",
            "cannot read propert",
            "has no attribute",
        ]
    ):
        return ErrorCategory.TYPE_ERROR

    if any(
        token in error_lower
        for token in [
            "importerror",
            "import error",
            "cannot import",
            "failed to import",
            "no module named",
            "module not found",
        ]
    ):
        return ErrorCategory.IMPORT_ERROR

    if any(
        token in error_lower
        for token in [
            "cannot find module",
            "module not found",
            "package not found",
            "not installed",
            "modulenotfounderror",
            "could not resolve",
            "missing dependency",
            "unmet dependency",
        ]
    ):
        return ErrorCategory.MISSING_DEPENDENCY

    if any(
        token in error_lower
        for token in [
            "build failed",
            "compilation error",
            "compile error",
            "tsc error",
            "failed to compile",
            "build error",
            "bundler error",
            "webpack error",
            "vite error",
        ]
    ):
        return ErrorCategory.BUILD_ERROR

    if any(
        token in error_lower
        for token in [
            "invalid configuration",
            "config error",
            "missing environment",
            "env var",
            "environment variable",
            "configuration file",
            "config file not found",
            ".env",
            "dotenv",
        ]
    ):
        return ErrorCategory.CONFIG_ERROR

    if any(
        token in error_lower
        for token in [
            "out of memory",
            "memory error",
            "heap out of memory",
            "javascript heap",
            "killed",
            "oom",
            "cannot allocate memory",
            "memoryerror",
        ]
    ):
        return ErrorCategory.OUT_OF_MEMORY

    if any(
        token in error_lower
        for token in [
            "segmentation fault",
            "segfault",
            "sigsegv",
            "bus error",
            "sigbus",
            "core dumped",
            "aborted",
            "sigabrt",
        ]
    ):
        return ErrorCategory.PROCESS_ERROR

    if any(
        token in error_lower
        for token in [
            "no space left",
            "disk full",
            "enospc",
            "not enough space",
            "disk quota exceeded",
        ]
    ):
        return ErrorCategory.DISK_FULL

    if any(
        token in error_lower
        for token in ["no such file", "file not found", "does not exist", "enoent"]
    ):
        return ErrorCategory.FILE_NOT_FOUND

    if any(
        token in error_lower
        for token in ["permission denied", "access denied", "not permitted", "eacces"]
    ):
        return ErrorCategory.PERMISSION_DENIED

    if any(
        token in error_lower
        for token in ["syntax error", "invalid syntax", "parse error", "unexpected token"]
    ):
        return ErrorCategory.SYNTAX_ERROR

    if any(
        token in error_lower
        for token in [
            "command not found",
            "not recognized",
            "no such command",
            "not found in path",
        ]
    ):
        return ErrorCategory.COMMAND_NOT_FOUND

    if any(
        token in error_lower
        for token in ["timeout", "timed out", "etimedout", "deadline exceeded"]
    ):
        return ErrorCategory.TIMEOUT

    if any(
        token in error_lower
        for token in ["invalid argument", "missing required", "bad argument"]
    ):
        return ErrorCategory.INVALID_ARGUMENTS

    if any(
        token in error_lower
        for token in [
            "required positional argument",
            "missing 1 required",
            "missing required positional",
            "empty content",
        ]
    ):
        return ErrorCategory.INVALID_ARGUMENTS

    if any(token in error_lower for token in ["network", "unreachable", "dns", "getaddrinfo"]):
        return ErrorCategory.NETWORK_ERROR

    return ErrorCategory.UNKNOWN


def detect_missing_mutation_payload(
    tool_name: str,
    args: dict[str, Any] | None,
    error: str,
) -> dict[str, Any] | None:
    """Detect invalid mutation calls missing their real payload or target path."""

    arguments = dict(args or {})
    error_lower = error.lower()
    if error and not any(
        token in error_lower
        for token in [
            "required positional argument",
            "missing 1 required",
            "missing required",
            "empty content",
            "validation warning",
            "empty file path",
            "valid file path",
            "missing file path",
        ]
    ):
        return None

    file_path = str(arguments.get("file_path") or arguments.get("path") or "").strip()
    missing_target = tool_name in {"write", "edit", "patch"} and (
        "empty file path" in error_lower
        or "valid file path" in error_lower
        or "missing file path" in error_lower
        or (
            any(
                token in error_lower
                for token in [
                    "required positional argument",
                    "missing 1 required",
                    "missing required",
                ]
            )
            and "file_path" in error_lower
        )
    )

    if missing_target:
        return {
            "kind": "missing_target",
            "required_fields": ["file_path"],
            "invalid_fields": [],
            "file_path": file_path,
        }

    if tool_name == "write":
        invalid_fields = [
            field for field in ("content_chars", "content_lines") if field in arguments
        ]
        if "content" not in arguments and invalid_fields:
            return {
                "kind": "missing_payload",
                "required_fields": ["content"],
                "invalid_fields": invalid_fields,
                "file_path": file_path,
            }

    if tool_name == "edit":
        missing_fields = [
            field for field in ("old_string", "new_string") if field not in arguments
        ]
        invalid_fields = [
            field
            for field in (
                "old_string_chars",
                "old_string_lines",
                "new_string_chars",
                "new_string_lines",
            )
            if field in arguments
        ]
        if missing_fields and invalid_fields:
            return {
                "kind": "missing_payload",
                "required_fields": missing_fields,
                "invalid_fields": invalid_fields,
                "file_path": file_path,
            }

    if tool_name == "patch":
        invalid_fields = [field for field in ("hunk_count",) if field in arguments]
        if "patch" not in arguments and "hunks" not in arguments and invalid_fields:
            return {
                "kind": "missing_payload",
                "required_fields": ["patch or hunks"],
                "invalid_fields": invalid_fields,
                "file_path": file_path,
            }

    return None


def get_recovery_hints(
    category: ErrorCategory,
    tool_name: str,
    args: dict[str, Any] | None = None,
) -> str:
    """Get hints for recovering from a specific error category."""

    hints = {
        ErrorCategory.FILE_NOT_FOUND: [
            "Use glob to search for the file: glob(pattern='**/<filename>')",
            "List the directory to see what exists: bash(ls -la <dir>)",
            "Check for typos in the filename",
            "The file might be in a different directory",
        ],
        ErrorCategory.PERMISSION_DENIED: [
            "Check file permissions: bash(ls -la <file>)",
            "The file might be read-only or owned by another user",
            "Try a different location that is writable",
        ],
        ErrorCategory.DISK_FULL: [
            "Check disk space: bash(df -h)",
            "Clean up temporary files or free space",
            "The operation cannot proceed until space is available",
        ],
        ErrorCategory.SYNTAX_ERROR: [
            "Read the file to see the current content: read(file_path=...)",
            "Check the exact line number mentioned in the error",
            "Verify brackets, quotes, and indentation are correct",
        ],
        ErrorCategory.TYPE_ERROR: [
            "Read the error carefully - it tells you what type was expected vs received",
            "Check if a variable is undefined or null before using it",
            "Verify function arguments match the expected types",
        ],
        ErrorCategory.IMPORT_ERROR: [
            "Check if the module is installed: pip show <pkg> or npm list <pkg>",
            "Verify the import path is correct",
            "The module name might be different from the package name",
        ],
        ErrorCategory.COMMAND_NOT_FOUND: [
            "Check if the tool is installed: bash(which <command>)",
            "Install the missing tool if needed",
            "Use an alternative command that achieves the same goal",
        ],
        ErrorCategory.SCRIPT_NOT_FOUND: [
            "FIRST: Read package.json/Makefile to see available scripts",
            "Run `npm run` or `make help` to list available targets",
            "Do NOT guess script names - inspect what exists",
            "Look for README or docs that explain the project workflow",
        ],
        ErrorCategory.MISSING_DEPENDENCY: [
            "Install dependencies: npm install, pip install -r requirements.txt, etc.",
            "Check if you're in the correct project directory",
            "The package name might be different: check package.json or requirements.txt",
        ],
        ErrorCategory.VERSION_MISMATCH: [
            "Check the required version in package.json or similar",
            "Update the tool: nvm use, pyenv, etc.",
            "Consider using a version manager",
        ],
        ErrorCategory.BUILD_ERROR: [
            "Read the error output - it usually points to a specific file:line",
            "Read the problematic file to understand the issue",
            "Try running a linter first to catch obvious issues",
        ],
        ErrorCategory.LINT_ERROR: [
            "Read the linter output for specific issues",
            "Auto-fix if possible: npm run lint --fix, ruff --fix",
            "Read the problematic file and fix the style issues",
        ],
        ErrorCategory.TEST_FAILURE: [
            "Read the test output to see which test failed and why",
            "Read the failing test file to understand what's expected",
            "Check if the code being tested has the expected behavior",
        ],
        ErrorCategory.TIMEOUT: [
            "The operation is taking too long",
            "Try a simpler or more targeted operation",
            "Break the task into smaller steps",
        ],
        ErrorCategory.OUT_OF_MEMORY: [
            "The operation requires too much memory",
            "Try processing in smaller batches",
            "Close other applications if possible",
        ],
        ErrorCategory.PORT_IN_USE: [
            "Find what's using the port: bash(lsof -i :<port>) or bash(netstat -tlnp)",
            "Kill the existing process or use a different port",
            "Check if another instance is already running",
        ],
        ErrorCategory.PROCESS_ERROR: [
            "This is a crash - likely a bug in the code or a native dependency issue",
            "Check if all native dependencies are installed",
            "Try running with debug output",
        ],
        ErrorCategory.NETWORK_ERROR: [
            "Check network connectivity",
            "The service might be down or unreachable",
            "Try an offline alternative if possible",
        ],
        ErrorCategory.CONNECTION_REFUSED: [
            "The service is not running. Start it first.",
            "For databases: start mysql/postgres/redis/etc.",
            "For APIs: check if the server is running on the expected port",
            "Common commands: systemctl start <service>, docker start <container>",
        ],
        ErrorCategory.AUTH_ERROR: [
            "Check if credentials/tokens are correct",
            "The token might be expired - try logging in again",
            "Check if you have permission for this operation",
        ],
        ErrorCategory.GIT_CONFLICT: [
            "Read the conflicted files to see the conflict markers",
            "Resolve conflicts by editing the files",
            "After resolving: git add <files> && git commit",
        ],
        ErrorCategory.GIT_NOT_REPO: [
            "This directory is not a git repository",
            "Either cd to the correct directory or run git init",
            "Check if you're in the right project folder",
        ],
        ErrorCategory.GIT_DIRTY: [
            "You have uncommitted changes blocking the operation",
            "Options: git stash, git commit, or git checkout -- <files>",
            "Check what's changed: git status",
        ],
        ErrorCategory.CONFIG_ERROR: [
            "Read the config file to check for issues",
            "Check for missing environment variables",
            "Look for a .env.example file to see required vars",
        ],
        ErrorCategory.INVALID_ARGUMENTS: [
            "Review the tool/command parameters",
            "Check documentation for correct usage",
            "Verify argument types and formats",
        ],
        ErrorCategory.UNKNOWN: [
            "INVESTIGATE: Read relevant files to understand the error",
            "Try a fundamentally different approach",
            "Break the task into smaller diagnostic steps",
        ],
    }

    category_hints = hints.get(category, hints[ErrorCategory.UNKNOWN])

    if tool_name == "edit" and category == ErrorCategory.FILE_NOT_FOUND:
        category_hints = ["Use 'write' tool instead of 'edit' to create a new file"] + category_hints

    if tool_name == "bash" and category == ErrorCategory.COMMAND_NOT_FOUND:
        category_hints = ["Check if installed: bash(which <command>)"] + category_hints

    rewrite_target = extract_shell_text_rewrite_target(str((args or {}).get("command", "")))
    if tool_name == "bash" and rewrite_target is not None:
        category_hints = [
            f"Switch to edit/patch/write for `{rewrite_target}` instead of shell rewriting it",
            "Reuse the evidence you already gathered and apply the file change directly",
            "If the exact replacement span is unclear, read just the target file and then edit it",
        ] + category_hints

    payload_fix = detect_missing_mutation_payload(tool_name, args, "")
    if payload_fix is not None:
        required = ", ".join(payload_fix["required_fields"])
        invalid = ", ".join(payload_fix["invalid_fields"])
        target = payload_fix["file_path"]
        if payload_fix.get("kind") == "missing_target":
            if tool_name == "write":
                category_hints = [
                    (
                        f"Resend the mutation as `write(file_path=..., content='...')` "
                        f"for `{target}` with a real file path"
                        if target
                        else "Resend the mutation as `write(file_path=..., content='...')` with a real file path"
                    ),
                    "Do not leave `file_path` empty or pointed at an unknown target",
                    "Do not reread reference files first unless one specific fact still blocks the write target",
                ]
            elif tool_name == "edit":
                category_hints = [
                    (
                        f"Resend the mutation as `edit(file_path=..., old_string='...', new_string='...')` "
                        f"for `{target}` with a real file path"
                        if target
                        else "Resend the mutation as `edit(file_path=..., old_string='...', new_string='...')` with a real file path"
                    ),
                    "Do not leave `file_path` empty or pointed at an unknown target",
                    "Do not reread reference files first unless one specific exact replacement span is still unknown",
                ]
            elif tool_name == "patch":
                category_hints = [
                    (
                        f"Resend the mutation as `patch(file_path=..., patch='...')` "
                        f"for `{target}` with a real file path"
                        if target
                        else "Resend the mutation as `patch(file_path=..., patch='...')` with a real file path"
                    ),
                    "Do not leave `file_path` empty or pointed at an unknown target",
                    "Do not reread reference files first unless one specific edit span is still unknown",
                ]
        elif tool_name == "write":
            category_hints = [
                (
                    f"Resend the mutation as `write(file_path=..., content='...')` "
                    f"for `{target}` with the real file body"
                    if target
                    else "Resend the mutation as `write(file_path=..., content='...')` with the real file body"
                ),
                (
                    f"`{invalid}` are summary fields, not valid write inputs; provide `{required}` instead"
                ),
                "Do not reread reference files first unless one specific fact still blocks the write",
            ]
        elif tool_name == "edit":
            category_hints = [
                (
                    f"Resend the mutation for `{target}` with the real `{required}` text payload"
                    if target
                    else f"Resend the mutation with the real `{required}` text payload"
                ),
                f"`{invalid}` are summary fields, not valid edit inputs; provide `{required}` instead",
                "Do not reread reference files first unless one specific exact replacement span is still unknown",
            ]
        elif tool_name == "patch":
            category_hints = [
                (
                    f"Resend the mutation for `{target}` with real `patch` text or structured `hunks`"
                    if target
                    else "Resend the mutation with real `patch` text or structured `hunks`"
                ),
                f"`{invalid}` are summary fields, not valid patch inputs; provide `{required}` instead",
                "Do not reread reference files first unless one specific edit span is still unknown",
            ]

    return "\n".join(f"- {hint}" for hint in category_hints)


RECOVERY_PROMPT = """## TOOL FAILURE - INVESTIGATE AND ADAPT

The command failed. You MUST analyze the error and take a DIFFERENT action.

**Failed Command:** {tool_name}({args})
**Error Type:** {category}
**Error Message:** {error}

{attempts_summary}

## REQUIRED: Choose ONE of these recovery actions:

{hints}

## CRITICAL RULES:
1. Start from the error and the state you already know
2. Investigate only if a specific fact is still missing
3. If you already have enough confirmed evidence, apply the fix instead of rereading the same files
4. **DO NOT** just retry the same command with slight variations
5. **DO NOT** try `npm start` then `npm run start` - these are the same thing!
6. **READ THE ERROR** - It usually tells you exactly what's wrong
7. If the error says "missing script: start", read package.json to see what scripts exist

## Current attempt: {attempt_count}/{max_retries}

**Your next action should either gather the missing information OR apply the fix using confirmed findings.**
What will you do?"""


def format_recovery_prompt(
    context: RecoveryContext,
    tool_name: str,
    args: dict[str, Any],
    error: str,
) -> str:
    """Format a prompt asking the LLM to recover from an error."""

    category = categorize_error(error)
    hints = get_recovery_hints(category, tool_name, args)
    args_str = ", ".join(f"{key}={value!r}" for key, value in args.items())

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

    last_category = (
        context.attempts[-1].category if context.attempts else ErrorCategory.UNKNOWN
    )

    lines = [
        f"Failed to complete the operation after {len(context.attempts)} attempts.",
        "",
        "What was tried:",
    ]

    for index, attempt in enumerate(context.attempts, 1):
        args_str = ", ".join(
            f"{key}={value!r}" for key, value in attempt.arguments.items()
        )
        lines.append(f"{index}. {attempt.tool_name}({args_str})")
        error_preview = (
            attempt.error[:200] + "..." if len(attempt.error) > 200 else attempt.error
        )
        lines.append(f"   Error: {error_preview}")

    suggestions = {
        ErrorCategory.SCRIPT_NOT_FOUND: [
            "Check package.json/Makefile to see available scripts",
            "The project might not have a start script - check the README",
            "Try running the main file directly: node index.js or similar",
        ],
        ErrorCategory.MISSING_DEPENDENCY: [
            "Run: npm install, pip install -r requirements.txt, etc.",
            "Check if you're in the correct project directory",
        ],
        ErrorCategory.FILE_NOT_FOUND: [
            "Verify the file path exists",
            "Use 'find' or 'ls' to locate the file",
        ],
        ErrorCategory.PORT_IN_USE: [
            "Find and kill the process using the port",
            "Or use a different port",
        ],
        ErrorCategory.CONNECTION_REFUSED: [
            "Start the required service (database, API server, etc.)",
            "Check if the service is configured correctly",
        ],
        ErrorCategory.GIT_CONFLICT: [
            "Manually resolve the merge conflicts",
            "Look for <<<<<<< markers in the files",
        ],
        ErrorCategory.GIT_DIRTY: [
            "Commit or stash your changes first",
            "Run: git status to see what's changed",
        ],
        ErrorCategory.AUTH_ERROR: [
            "Check your credentials/tokens",
            "You may need to log in again",
        ],
        ErrorCategory.BUILD_ERROR: [
            "Check the build output for specific file:line errors",
            "Fix the syntax/type errors in the mentioned files",
        ],
        ErrorCategory.TEST_FAILURE: [
            "Review the test output to see what failed",
            "The tests may need to be updated for code changes",
        ],
        ErrorCategory.CONFIG_ERROR: [
            "Check your configuration files",
            "Look for missing environment variables",
        ],
        ErrorCategory.OUT_OF_MEMORY: [
            "Try processing less data at once",
            "Close other applications to free memory",
        ],
        ErrorCategory.VERSION_MISMATCH: [
            "Check required versions in package.json/pyproject.toml",
            "Use a version manager (nvm, pyenv) to switch versions",
        ],
    }

    specific_suggestions = suggestions.get(
        last_category,
        [
            "Manually check the file/directory structure",
            "Review the error messages for clues",
            "Try a completely different approach",
        ],
    )

    lines.extend(["", "Suggestions:"])
    for suggestion in specific_suggestions:
        lines.append(f"- {suggestion}")

    return "\n".join(lines)
