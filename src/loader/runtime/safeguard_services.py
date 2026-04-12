"""Runtime-owned safeguard services shared by hooks and agent adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class ActionTracker:
    """Tracks completed actions to prevent duplicates and detect loops."""

    MAX_SEQUENCE_LENGTH = 20
    LOOP_PATTERN_MIN = 2
    LOOP_REPEAT_THRESHOLD = 2
    MAX_RESPONSE_HISTORY = 5

    def __init__(self) -> None:
        self._file_writes: dict[str, list[str]] = {}
        self._files_edited: dict[str, list[str]] = {}
        self._commands_run: set[str] = set()
        self._dirs_created: set[str] = set()
        self._action_sequence: list[str] = []
        self._response_history: list[str] = []

    def reset(self) -> None:
        self._file_writes.clear()
        self._files_edited.clear()
        self._commands_run.clear()
        self._dirs_created.clear()
        self._action_sequence.clear()
        self._response_history.clear()

    def _normalize_path(self, path: str) -> str:
        expanded = Path(path).expanduser()
        try:
            return str(expanded.resolve())
        except Exception:
            return str(expanded)

    @staticmethod
    def _make_edit_signature(old_string: str, new_string: str) -> str:
        return f"{hash(old_string)}:{hash(new_string)}"

    @staticmethod
    def _make_write_signature(content: str) -> str:
        return str(hash(content))

    def would_duplicate_file_create(self, file_path: str, content: str) -> bool:
        norm_path = self._normalize_path(file_path)
        sig = self._make_write_signature(content)
        return sig in self._file_writes.get(norm_path, [])

    def would_duplicate_edit(self, file_path: str, old_string: str, new_string: str) -> bool:
        norm_path = self._normalize_path(file_path)
        sig = self._make_edit_signature(old_string, new_string)
        return sig in self._files_edited.get(norm_path, [])

    def would_duplicate_patch(self, file_path: str, hunks: list[dict]) -> bool:
        norm_path = self._normalize_path(file_path)
        sig = str(hash(str(hunks)))
        return sig in self._files_edited.get(norm_path, [])

    def would_duplicate_command(self, command: str) -> bool:
        norm_cmd = " ".join(command.split())
        return norm_cmd in self._commands_run

    def would_duplicate_mkdir(self, dir_path: str) -> bool:
        norm_path = self._normalize_path(dir_path)
        return norm_path in self._dirs_created

    def record_file_create(self, file_path: str, content: str) -> None:
        norm_path = self._normalize_path(file_path)
        sig = self._make_write_signature(content)
        self._file_writes.setdefault(norm_path, []).append(sig)

    def record_edit(self, file_path: str, old_string: str, new_string: str) -> None:
        norm_path = self._normalize_path(file_path)
        sig = self._make_edit_signature(old_string, new_string)
        self._files_edited.setdefault(norm_path, []).append(sig)

    def record_command(self, command: str) -> None:
        norm_cmd = " ".join(command.split())
        self._commands_run.add(norm_cmd)

        mkdir_match = re.match(r'mkdir\s+(-p\s+)?(.+)', norm_cmd)
        if mkdir_match:
            dir_path = mkdir_match.group(2).strip().strip('"\'')
            self._dirs_created.add(self._normalize_path(dir_path))

    def record_mkdir(self, dir_path: str) -> None:
        self._dirs_created.add(self._normalize_path(dir_path))

    def check_tool_call(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        if tool_name == "write":
            file_path = arguments.get("file_path", "")
            content = arguments.get("content", "")
            if self.would_duplicate_file_create(file_path, content):
                return True, f"Same file content already written: {file_path}"

        elif tool_name == "edit":
            file_path = arguments.get("file_path", "")
            old_string = arguments.get("old_string", "")
            new_string = arguments.get("new_string", "")
            if self.would_duplicate_edit(file_path, old_string, new_string):
                return True, f"Same edit already applied to: {file_path}"

        elif tool_name == "patch":
            file_path = arguments.get("file_path", "")
            hunks = arguments.get("hunks", [])
            if isinstance(hunks, list) and self.would_duplicate_patch(file_path, hunks):
                return True, f"Same patch already applied to: {file_path}"

        elif tool_name == "bash":
            command = arguments.get("command", "")
            if self.would_duplicate_command(command):
                return True, f"Command already executed: {command[:50]}..."

        return False, ""

    def record_tool_call(self, tool_name: str, arguments: dict) -> None:
        self._action_sequence.append(tool_name)
        if len(self._action_sequence) > self.MAX_SEQUENCE_LENGTH:
            self._action_sequence.pop(0)

        if tool_name == "write":
            file_path = arguments.get("file_path", "")
            content = arguments.get("content", "")
            if file_path:
                self.record_file_create(file_path, content)

        elif tool_name == "edit":
            file_path = arguments.get("file_path", "")
            old_string = arguments.get("old_string", "")
            new_string = arguments.get("new_string", "")
            if file_path:
                self.record_edit(file_path, old_string, new_string)

        elif tool_name == "patch":
            file_path = arguments.get("file_path", "")
            hunks = arguments.get("hunks", [])
            if file_path:
                self.record_edit(file_path, str(hunks), "structured_patch")

        elif tool_name == "bash":
            command = arguments.get("command", "")
            if command:
                self.record_command(command)

    def detect_loop(self) -> tuple[bool, str]:
        seq = self._action_sequence
        if len(seq) < self.LOOP_PATTERN_MIN * self.LOOP_REPEAT_THRESHOLD:
            return False, ""

        for pattern_len in range(self.LOOP_PATTERN_MIN, min(6, len(seq) // 2 + 1)):
            pattern = seq[-pattern_len:]
            repeats = 1
            for i in range(len(seq) - pattern_len * 2, -1, -pattern_len):
                if seq[i:i + pattern_len] == pattern:
                    repeats += 1
                else:
                    break

            if repeats >= self.LOOP_REPEAT_THRESHOLD:
                pattern_str = " → ".join(pattern)
                return True, f"Repeating pattern detected ({repeats}x): {pattern_str}"

        return False, ""

    @staticmethod
    def _normalize_response(response: str) -> str:
        normalized = response.strip().lower()[:200]
        normalized = re.sub(r'/[\w/.-]+', '<PATH>', normalized)
        normalized = re.sub(r'\d+', '<NUM>', normalized)
        return normalized

    def record_response(self, response: str) -> None:
        normalized = self._normalize_response(response)
        self._response_history.append(normalized)
        if len(self._response_history) > self.MAX_RESPONSE_HISTORY:
            self._response_history.pop(0)

    def detect_text_loop(self, response: str) -> tuple[bool, str]:
        if len(self._response_history) < 2:
            return False, ""

        normalized = self._normalize_response(response)
        exact_matches = sum(1 for r in self._response_history if r == normalized)
        if exact_matches >= 2:
            return True, f"Agent repeated the same response {exact_matches + 1} times"

        repetitive_phrases = [
            "apologies for any confusion",
            "let me proceed",
            "i will now use the",
        ]
        response_lower = response.lower()
        for phrase in repetitive_phrases:
            if phrase in response_lower:
                phrase_count = sum(1 for r in self._response_history if phrase in r)
                if phrase_count >= 2:
                    return True, f"Agent is stuck repeating '{phrase}'"

        current_words = set(normalized.split())
        similarity_matches = 0
        for prev in self._response_history[-3:]:
            prev_words = set(prev.split())
            if len(current_words) > 10 and len(prev_words) > 10:
                overlap = len(current_words & prev_words)
                similarity = overlap / max(len(current_words), len(prev_words))
                if similarity > 0.85:
                    similarity_matches += 1

        if similarity_matches >= 2:
            return True, "Agent responses are highly repetitive"

        return False, ""

    def reset_response_history(self) -> None:
        """Clear response history between turns to prevent cross-turn false positives."""
        self._response_history.clear()


@dataclass
class ValidationResult:
    """Result of pre-action validation."""

    valid: bool
    reason: str = ""
    suggestion: str = ""
    severity: str = "warning"


class PreActionValidator:
    """Validates tool calls before execution to catch problematic actions."""

    DANGEROUS_PATTERNS = [
        (r'rm\s+(-[rf]+\s+)?/', "Dangerous: removing from root directory"),
        (r'rm\s+-rf\s+~', "Dangerous: removing home directory"),
        (r'>\s*/dev/sd[a-z]', "Dangerous: writing directly to disk device"),
        (r'mkfs\.', "Dangerous: formatting filesystem"),
        (r'dd\s+.*of=/dev/', "Dangerous: dd to device"),
        (r'chmod\s+-R\s+777\s+/', "Dangerous: making everything world-writable"),
        (r':\(\)\s*\{\s*:\|:\s*&\s*\}\s*;', "Dangerous: fork bomb"),
    ]

    SUSPICIOUS_PATTERNS = [
        (r'rm\s+-rf\s+', "Warning: recursive force delete"),
        (r'>\s*/etc/', "Warning: overwriting system config"),
        (r'curl\s+.*\|\s*sh', "Warning: piping curl to shell"),
        (r'wget\s+.*\|\s*sh', "Warning: piping wget to shell"),
        (r'eval\s+', "Warning: using eval"),
        (r'sudo\s+', "Warning: using sudo"),
    ]

    def validate(self, tool_name: str, arguments: dict) -> ValidationResult:
        if tool_name == "bash":
            return self._validate_bash(arguments)
        if tool_name == "write":
            return self._validate_write(arguments)
        if tool_name == "edit":
            return self._validate_edit(arguments)
        if tool_name == "patch":
            return self._validate_patch(arguments)
        if tool_name == "read":
            return self._validate_read(arguments)
        if tool_name in ("glob", "grep"):
            return self._validate_search(tool_name, arguments)
        return ValidationResult(valid=True)

    def _validate_bash(self, arguments: dict) -> ValidationResult:
        command = arguments.get("command", "")

        if not command or not command.strip():
            return ValidationResult(
                valid=False,
                reason="Empty command",
                suggestion="Provide a valid command to execute",
                severity="error",
            )

        for pattern, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return ValidationResult(
                    valid=False,
                    reason=reason,
                    suggestion="This command is too dangerous to execute",
                    severity="block",
                )

        for pattern, reason in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, command):
                return ValidationResult(valid=True, reason=reason, severity="warning")

        interactive_patterns = [
            (r'\bnano\b', "nano requires interactive terminal"),
            (r'\bvim?\b', "vim requires interactive terminal"),
            (r'\bemacs\b', "emacs requires interactive terminal"),
            (r'\bless\b', "less requires interactive terminal"),
            (r'\bmore\b', "more requires interactive terminal"),
            (r'\btop\b', "top requires interactive terminal"),
            (r'\bhtop\b', "htop requires interactive terminal"),
        ]
        for pattern, reason in interactive_patterns:
            if re.search(pattern, command):
                return ValidationResult(
                    valid=False,
                    reason=reason,
                    suggestion=(
                        "Use non-interactive alternatives (cat, head, tail for viewing; "
                        "sed for editing)"
                    ),
                    severity="error",
                )

        return ValidationResult(valid=True)

    def _validate_write(self, arguments: dict) -> ValidationResult:
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not file_path or not file_path.strip():
            return ValidationResult(
                valid=False,
                reason="Empty file path",
                suggestion="Provide a valid file path",
                severity="error",
            )

        path_result = self._validate_path(file_path)
        if not path_result.valid:
            return path_result

        if content is None or (isinstance(content, str) and not content.strip()):
            return ValidationResult(
                valid=True,
                reason="Writing empty content to file",
                severity="warning",
            )

        sensitive_paths = ['/etc/', '/usr/', '/bin/', '/sbin/', '/boot/', '/sys/', '/proc/']
        for sensitive in sensitive_paths:
            if file_path.startswith(sensitive):
                return ValidationResult(
                    valid=False,
                    reason=f"Cannot write to system directory: {sensitive}",
                    suggestion="Write to a user directory instead",
                    severity="block",
                )

        return ValidationResult(valid=True)

    def _validate_edit(self, arguments: dict) -> ValidationResult:
        file_path = arguments.get("file_path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")

        if not file_path or not file_path.strip():
            return ValidationResult(
                valid=False,
                reason="Empty file path",
                suggestion="Provide a valid file path",
                severity="error",
            )

        path_result = self._validate_path(file_path)
        if not path_result.valid:
            return path_result

        if old_string is None:
            return ValidationResult(
                valid=False,
                reason="old_string is None",
                suggestion="Provide the text to replace (can be empty string for prepend)",
                severity="error",
            )

        if new_string is None:
            return ValidationResult(
                valid=False,
                reason="new_string is None",
                suggestion="Provide the replacement text (can be empty string for deletion)",
                severity="error",
            )

        if old_string == new_string:
            return ValidationResult(
                valid=False,
                reason="old_string and new_string are identical - no change would occur",
                suggestion="Provide different old and new strings",
                severity="error",
            )

        return ValidationResult(valid=True)

    def _validate_patch(self, arguments: dict) -> ValidationResult:
        file_path = arguments.get("file_path", "")
        hunks = arguments.get("hunks", [])

        if not file_path or not str(file_path).strip():
            return ValidationResult(
                valid=False,
                reason="Empty file path",
                suggestion="Provide a valid file path",
                severity="error",
            )

        path_result = self._validate_path(str(file_path))
        if not path_result.valid:
            return path_result

        if not isinstance(hunks, list) or not hunks:
            return ValidationResult(
                valid=False,
                reason="Patch hunks are missing",
                suggestion="Provide one or more structured patch hunks",
                severity="error",
            )

        return ValidationResult(valid=True)

    def _validate_read(self, arguments: dict) -> ValidationResult:
        file_path = arguments.get("file_path", "")

        if not file_path or not file_path.strip():
            return ValidationResult(
                valid=False,
                reason="Empty file path",
                suggestion="Provide a valid file path",
                severity="error",
            )

        return self._validate_path(file_path)

    def _validate_search(self, tool_name: str, arguments: dict) -> ValidationResult:
        pattern = arguments.get("pattern", "")

        if not pattern or not pattern.strip():
            return ValidationResult(
                valid=False,
                reason=f"Empty {tool_name} pattern",
                suggestion="Provide a valid search pattern",
                severity="error",
            )

        return ValidationResult(valid=True)

    def _validate_path(self, file_path: str) -> ValidationResult:
        if '\x00' in file_path:
            return ValidationResult(
                valid=False,
                reason="Path contains null byte",
                suggestion="Remove null bytes from path",
                severity="block",
            )

        if '/../../../' in file_path or file_path.count('..') > 5:
            return ValidationResult(
                valid=False,
                reason="Excessive path traversal",
                suggestion="Use a direct path instead",
                severity="warning",
            )

        return ValidationResult(valid=True)
