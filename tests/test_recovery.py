"""Tests for the error recovery system."""

from loader.agent.recovery import (
    ErrorCategory,
    RecoveryContext,
    categorize_error,
    format_failure_message,
    format_recovery_prompt,
    get_recovery_hints,
)


class TestCategorizeError:
    """Tests for error categorization."""

    def test_file_not_found(self):
        assert categorize_error("No such file or directory") == ErrorCategory.FILE_NOT_FOUND
        assert categorize_error("file not found: test.py") == ErrorCategory.FILE_NOT_FOUND
        assert categorize_error("Path does not exist") == ErrorCategory.FILE_NOT_FOUND

    def test_permission_denied(self):
        assert categorize_error("Permission denied") == ErrorCategory.PERMISSION_DENIED
        assert categorize_error("Access denied to file") == ErrorCategory.PERMISSION_DENIED
        assert categorize_error("Operation not permitted") == ErrorCategory.PERMISSION_DENIED

    def test_syntax_error(self):
        assert categorize_error("SyntaxError: invalid syntax") == ErrorCategory.SYNTAX_ERROR
        assert categorize_error("Parse error at line 5") == ErrorCategory.SYNTAX_ERROR

    def test_command_not_found(self):
        assert categorize_error("command not found: foo") == ErrorCategory.COMMAND_NOT_FOUND
        assert categorize_error("'bar' is not recognized") == ErrorCategory.COMMAND_NOT_FOUND

    def test_timeout(self):
        assert categorize_error("Operation timed out") == ErrorCategory.TIMEOUT
        assert categorize_error("Connection timeout") == ErrorCategory.TIMEOUT

    def test_invalid_arguments(self):
        assert categorize_error("Invalid argument: path") == ErrorCategory.INVALID_ARGUMENTS
        assert categorize_error("Missing required parameter") == ErrorCategory.INVALID_ARGUMENTS

    def test_network_error(self):
        assert categorize_error("Network unreachable") == ErrorCategory.NETWORK_ERROR
        assert categorize_error("Connection refused") == ErrorCategory.CONNECTION_REFUSED

    def test_unknown(self):
        assert categorize_error("Something weird happened") == ErrorCategory.UNKNOWN
        assert categorize_error("") == ErrorCategory.UNKNOWN


class TestRecoveryContext:
    """Tests for RecoveryContext tracking."""

    def test_add_attempt(self):
        ctx = RecoveryContext(
            original_tool="read",
            original_args={"path": "test.py"},
        )
        assert len(ctx.attempts) == 0

        ctx.add_attempt("read", {"path": "test.py"}, "File not found")
        assert len(ctx.attempts) == 1
        assert ctx.attempts[0].tool_name == "read"
        assert ctx.attempts[0].category == ErrorCategory.FILE_NOT_FOUND

    def test_can_retry(self):
        ctx = RecoveryContext(
            original_tool="read",
            original_args={"path": "test.py"},
            max_retries=3,
        )
        assert ctx.can_retry()

        ctx.add_attempt("read", {"path": "test.py"}, "Error 1")
        assert ctx.can_retry()

        ctx.add_attempt("read", {"path": "test2.py"}, "Error 2")
        assert ctx.can_retry()

        ctx.add_attempt("read", {"path": "test3.py"}, "Error 3")
        assert not ctx.can_retry()

    def test_was_tried(self):
        ctx = RecoveryContext(
            original_tool="read",
            original_args={"path": "test.py"},
        )
        assert not ctx.was_tried("read", {"path": "test.py"})

        ctx.add_attempt("read", {"path": "test.py"}, "Error")
        assert ctx.was_tried("read", {"path": "test.py"})
        assert not ctx.was_tried("read", {"path": "other.py"})
        assert not ctx.was_tried("write", {"path": "test.py"})

    def test_attempts_summary(self):
        ctx = RecoveryContext(
            original_tool="read",
            original_args={"path": "test.py"},
        )
        assert ctx.attempts_summary() == ""

        ctx.add_attempt("read", {"path": "test.py"}, "File not found")
        summary = ctx.attempts_summary()
        assert "Previous attempts:" in summary
        assert "read" in summary
        assert "File not found" in summary


class TestGetRecoveryHints:
    """Tests for recovery hints."""

    def test_file_not_found_hints(self):
        hints = get_recovery_hints(ErrorCategory.FILE_NOT_FOUND, "read")
        assert "glob" in hints.lower()
        assert "directory" in hints.lower()

    def test_edit_file_not_found_special_hint(self):
        hints = get_recovery_hints(ErrorCategory.FILE_NOT_FOUND, "edit")
        assert "write" in hints.lower()

    def test_bash_command_not_found_special_hint(self):
        hints = get_recovery_hints(ErrorCategory.COMMAND_NOT_FOUND, "bash")
        assert "which" in hints.lower()


class TestFormatRecoveryPrompt:
    """Tests for recovery prompt formatting."""

    def test_format_recovery_prompt(self):
        ctx = RecoveryContext(
            original_tool="read",
            original_args={"path": "test.py"},
        )
        ctx.add_attempt("read", {"path": "test.py"}, "No such file")

        prompt = format_recovery_prompt(ctx, "read", {"path": "test.py"}, "No such file")
        assert "Failed Command" in prompt
        assert "read(path='test.py')" in prompt
        assert "No such file" in prompt
        assert "1/3" in prompt
        assert "retry the same command with slight variations" in prompt


class TestFormatFailureMessage:
    """Tests for failure message formatting."""

    def test_format_failure_message(self):
        ctx = RecoveryContext(
            original_tool="read",
            original_args={"path": "test.py"},
        )
        ctx.add_attempt("read", {"path": "test.py"}, "Error 1")
        ctx.add_attempt("glob", {"pattern": "*.py"}, "Error 2")
        ctx.add_attempt("read", {"path": "src/test.py"}, "Error 3")

        msg = format_failure_message(ctx)
        assert "3 attempts" in msg
        assert "read" in msg
        assert "glob" in msg
        assert "Error 1" in msg
        assert "Error 2" in msg
        assert "Error 3" in msg
