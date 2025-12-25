"""Error recovery system for the agent.

Provides intelligent retry logic with adaptation when tools fail.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ErrorCategory(Enum):
    """Categories of errors for recovery strategies."""
    # File system
    FILE_NOT_FOUND = auto()
    PERMISSION_DENIED = auto()
    DISK_FULL = auto()
    PATH_TOO_LONG = auto()

    # Code/syntax
    SYNTAX_ERROR = auto()
    TYPE_ERROR = auto()
    IMPORT_ERROR = auto()

    # Commands/executables
    COMMAND_NOT_FOUND = auto()
    SCRIPT_NOT_FOUND = auto()  # npm/yarn/make script missing

    # Dependencies
    MISSING_DEPENDENCY = auto()  # Module/package not found
    VERSION_MISMATCH = auto()  # Incompatible versions

    # Build/compile
    BUILD_ERROR = auto()
    LINT_ERROR = auto()
    TEST_FAILURE = auto()

    # Runtime
    TIMEOUT = auto()
    OUT_OF_MEMORY = auto()
    PORT_IN_USE = auto()
    PROCESS_ERROR = auto()  # Segfault, killed, etc.

    # Network/external
    NETWORK_ERROR = auto()
    CONNECTION_REFUSED = auto()  # Service not running
    AUTH_ERROR = auto()  # Unauthorized, forbidden

    # Git
    GIT_CONFLICT = auto()
    GIT_NOT_REPO = auto()
    GIT_DIRTY = auto()  # Uncommitted changes blocking operation

    # Config/environment
    CONFIG_ERROR = auto()  # Invalid config, missing env var
    INVALID_ARGUMENTS = auto()

    # Fallback
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

    def is_similar_attempt(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if this is essentially the same as a previous attempt.

        Catches cases like:
        - npm start vs npm run start (same thing)
        - python script.py vs python3 script.py
        - Same command with minor flag variations
        """
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

    @staticmethod
    def _normalize_command(cmd: str) -> str:
        """Normalize a command for comparison.

        Makes these equivalent:
        - npm start == npm run start
        - cd dir && npm start == npm start (in dir context)
        - python == python3
        """
        import re

        # Remove cd prefix (we care about the actual command)
        cmd = re.sub(r'^cd\s+[^\s]+\s*&&\s*', '', cmd)

        # Normalize npm commands
        cmd = re.sub(r'\bnpm run start\b', 'npm start', cmd)
        cmd = re.sub(r'\bnpm run serve\b', 'npm serve', cmd)
        cmd = re.sub(r'\bnpm run dev\b', 'npm dev', cmd)

        # Normalize python
        cmd = re.sub(r'\bpython3\b', 'python', cmd)

        # Normalize whitespace
        cmd = ' '.join(cmd.split())

        return cmd.strip()


def categorize_error(error_message: str) -> ErrorCategory:
    """Categorize an error message for recovery strategy selection."""
    error_lower = error_message.lower()

    # === MOST SPECIFIC FIRST ===

    # Git errors
    if any(x in error_lower for x in [
        "merge conflict", "conflict", "unmerged paths",
        "fix conflicts", "both modified",
    ]):
        return ErrorCategory.GIT_CONFLICT

    if any(x in error_lower for x in [
        "not a git repository", "fatal: not a git",
        "not in a git directory",
    ]):
        return ErrorCategory.GIT_NOT_REPO

    if any(x in error_lower for x in [
        "uncommitted changes", "working tree not clean",
        "please commit or stash", "local changes would be overwritten",
        "changes not staged",
    ]):
        return ErrorCategory.GIT_DIRTY

    # Script/task runner errors
    if any(x in error_lower for x in [
        "missing script", "npm err! missing script",
        "script not found", "no script",
        "error: script", "yarn run error",
        "make: *** no rule to make target",
        "no such task", "task not found",
    ]):
        return ErrorCategory.SCRIPT_NOT_FOUND

    # Port/address errors
    if any(x in error_lower for x in [
        "address already in use", "port already in use",
        "eaddrinuse", "bind: address already in use",
        "port is already allocated",
    ]):
        return ErrorCategory.PORT_IN_USE

    # Connection/service errors
    if any(x in error_lower for x in [
        "connection refused", "econnrefused",
        "could not connect", "unable to connect",
        "service unavailable", "no such host",
    ]):
        return ErrorCategory.CONNECTION_REFUSED

    # Auth errors
    if any(x in error_lower for x in [
        "unauthorized", "403 forbidden", "401 unauthorized",
        "authentication failed", "invalid credentials",
        "access denied", "permission denied to",
        "invalid token", "token expired",
    ]):
        return ErrorCategory.AUTH_ERROR

    # Version/compatibility
    if any(x in error_lower for x in [
        "version mismatch", "incompatible version",
        "requires node", "requires python",
        "unsupported version", "engine requirements",
        "peer dep", "peer dependency",
    ]):
        return ErrorCategory.VERSION_MISMATCH

    # Test failures
    if any(x in error_lower for x in [
        "test failed", "tests failed", "failing tests",
        "assertion error", "assertionerror",
        "expected", "to equal", "to be",
        "test suite failed",
    ]):
        return ErrorCategory.TEST_FAILURE

    # Lint errors
    if any(x in error_lower for x in [
        "eslint", "pylint", "flake8", "ruff",
        "lint error", "linting failed",
        "style violation", "formatting error",
    ]):
        return ErrorCategory.LINT_ERROR

    # Type errors (specific)
    if any(x in error_lower for x in [
        "typeerror", "type error",
        "is not a function", "is not defined",
        "undefined is not", "null is not",
        "cannot read propert", "has no attribute",
    ]):
        return ErrorCategory.TYPE_ERROR

    # Import errors (specific)
    if any(x in error_lower for x in [
        "importerror", "import error",
        "cannot import", "failed to import",
        "no module named", "module not found",
    ]):
        return ErrorCategory.IMPORT_ERROR

    # Missing dependencies/modules
    if any(x in error_lower for x in [
        "cannot find module", "module not found",
        "package not found", "not installed",
        "modulenotfounderror", "could not resolve",
        "missing dependency", "unmet dependency",
    ]):
        return ErrorCategory.MISSING_DEPENDENCY

    # Build/compilation errors
    if any(x in error_lower for x in [
        "build failed", "compilation error", "compile error",
        "tsc error", "failed to compile", "build error",
        "bundler error", "webpack error", "vite error",
    ]):
        return ErrorCategory.BUILD_ERROR

    # Config/environment errors
    if any(x in error_lower for x in [
        "invalid configuration", "config error",
        "missing environment", "env var", "environment variable",
        "configuration file", "config file not found",
        ".env", "dotenv",
    ]):
        return ErrorCategory.CONFIG_ERROR

    # Memory/resource errors
    if any(x in error_lower for x in [
        "out of memory", "memory error", "heap out of memory",
        "javascript heap", "killed", "oom",
        "cannot allocate memory", "memoryerror",
    ]):
        return ErrorCategory.OUT_OF_MEMORY

    # Process errors
    if any(x in error_lower for x in [
        "segmentation fault", "segfault", "sigsegv",
        "bus error", "sigbus", "core dumped",
        "aborted", "sigabrt",
    ]):
        return ErrorCategory.PROCESS_ERROR

    # Disk space
    if any(x in error_lower for x in [
        "no space left", "disk full", "enospc",
        "not enough space", "disk quota exceeded",
    ]):
        return ErrorCategory.DISK_FULL

    # === GENERAL CATEGORIES ===

    if any(x in error_lower for x in ["no such file", "file not found", "does not exist", "enoent"]):
        return ErrorCategory.FILE_NOT_FOUND

    if any(x in error_lower for x in ["permission denied", "access denied", "not permitted", "eacces"]):
        return ErrorCategory.PERMISSION_DENIED

    if any(x in error_lower for x in ["syntax error", "invalid syntax", "parse error", "unexpected token"]):
        return ErrorCategory.SYNTAX_ERROR

    if any(x in error_lower for x in ["command not found", "not recognized", "no such command", "not found in path"]):
        return ErrorCategory.COMMAND_NOT_FOUND

    if any(x in error_lower for x in ["timeout", "timed out", "etimedout", "deadline exceeded"]):
        return ErrorCategory.TIMEOUT

    if any(x in error_lower for x in ["invalid argument", "missing required", "bad argument"]):
        return ErrorCategory.INVALID_ARGUMENTS

    if any(x in error_lower for x in ["network", "unreachable", "dns", "getaddrinfo"]):
        return ErrorCategory.NETWORK_ERROR

    return ErrorCategory.UNKNOWN


def get_recovery_hints(category: ErrorCategory, tool_name: str) -> str:
    """Get hints for recovering from a specific error category."""
    hints = {
        # File system
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

        # Code/syntax
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

        # Commands
        ErrorCategory.COMMAND_NOT_FOUND: [
            "Check if the tool is installed: bash(which <command>)",
            "Install the missing tool if needed",
            "Use an alternative command that achieves the same goal",
        ],
        ErrorCategory.SCRIPT_NOT_FOUND: [
            "FIRST: Read package.json/Makefile to see available scripts",
            "Run `npm run` or `make help` to list available targets",
            "Common alternatives: dev, serve, start:dev, develop",
            "You may need to run the entry point directly: node index.js",
        ],

        # Dependencies
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

        # Build
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

        # Runtime
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

        # Network
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

        # Git
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

        # Config
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

        # Fallback
        ErrorCategory.UNKNOWN: [
            "INVESTIGATE: Read relevant files to understand the error",
            "Try a fundamentally different approach",
            "Break the task into smaller diagnostic steps",
        ],
    }

    category_hints = hints.get(category, hints[ErrorCategory.UNKNOWN])

    # Add tool-specific hints
    if tool_name == "edit" and category == ErrorCategory.FILE_NOT_FOUND:
        category_hints = ["Use 'write' tool instead of 'edit' to create a new file"] + category_hints

    if tool_name == "bash" and category == ErrorCategory.COMMAND_NOT_FOUND:
        category_hints = ["Check if installed: bash(which <command>)"] + category_hints

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
1. **INVESTIGATE FIRST** - Read config files, list directories, check what exists
2. **DO NOT** just retry the same command with slight variations
3. **DO NOT** try `npm start` then `npm run start` - these are the same thing!
4. **READ THE ERROR** - It usually tells you exactly what's wrong
5. If the error says "missing script: start", read package.json to see what scripts exist

## Current attempt: {attempt_count}/{max_retries}

**Your next action should gather information OR try a fundamentally different approach.**
What will you do?"""


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
    # Get the category from the last attempt to provide specific guidance
    last_category = context.attempts[-1].category if context.attempts else ErrorCategory.UNKNOWN

    lines = [
        f"Failed to complete the operation after {len(context.attempts)} attempts.",
        "",
        "What was tried:",
    ]

    for i, attempt in enumerate(context.attempts, 1):
        args_str = ", ".join(f"{k}={v!r}" for k, v in attempt.arguments.items())
        lines.append(f"{i}. {attempt.tool_name}({args_str})")
        # Truncate long error messages
        error_preview = attempt.error[:200] + "..." if len(attempt.error) > 200 else attempt.error
        lines.append(f"   Error: {error_preview}")

    # Category-specific suggestions for user action
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

    specific_suggestions = suggestions.get(last_category, [
        "Manually check the file/directory structure",
        "Review the error messages for clues",
        "Try a completely different approach",
    ])

    lines.extend(["", "Suggestions:"])
    for suggestion in specific_suggestions:
        lines.append(f"- {suggestion}")

    return "\n".join(lines)
