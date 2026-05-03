"""Runtime-owned safeguard services shared by hooks and agent adapters."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

TEXT_REWRITE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".css",
        ".csv",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".htm",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".py",
        ".rb",
        ".rs",
        ".sh",
        ".sql",
        ".svg",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)


def _html_target_tokens(target: str) -> set[str]:
    stem = Path(target).stem.lower()
    return {token for token in re.split(r"[^a-z0-9]+", stem) if token}


def _ordered_html_target_number(target: str) -> int | None:
    match = re.match(r"(\d+)[-_]", Path(target).name)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


TEXT_REWRITE_FILENAMES = frozenset(
    {
        "dockerfile",
        "index.html",
        "makefile",
        "package.json",
        "pyproject.toml",
        "readme",
        "readme.md",
    }
)


def _strip_shell_token(token: str) -> str:
    return token.strip().strip("\"'").rstrip(";|&")


def _looks_like_text_rewrite_target(token: str) -> bool:
    candidate = _strip_shell_token(token)
    if not candidate or candidate in {"-", "/dev/null"}:
        return False
    if candidate.startswith("-"):
        return False
    lowered = Path(candidate).name.lower()
    if lowered in TEXT_REWRITE_FILENAMES:
        return True
    return Path(candidate).suffix.lower() in TEXT_REWRITE_SUFFIXES


def _extract_redirect_target(argv: list[str]) -> str | None:
    for index, token in enumerate(argv):
        if token in {">", ">>"} and index + 1 < len(argv):
            candidate = argv[index + 1]
            if _looks_like_text_rewrite_target(candidate):
                return _strip_shell_token(candidate)
        if token == "tee":
            for candidate in argv[index + 1 :]:
                if candidate.startswith("-"):
                    continue
                if _looks_like_text_rewrite_target(candidate):
                    return _strip_shell_token(candidate)
                break
    return None


def extract_shell_text_rewrite_target(command: str) -> str | None:
    """Return the target file when bash is used as a brittle text editor."""

    normalized = " ".join(str(command or "").split())
    if not normalized:
        return None

    try:
        argv = shlex.split(normalized)
    except ValueError:
        argv = []

    if argv:
        for index, token in enumerate(argv):
            if token == "sed" and any(part.startswith("-i") for part in argv[index + 1 :]):
                for candidate in reversed(argv[index + 1 :]):
                    if _looks_like_text_rewrite_target(candidate):
                        return _strip_shell_token(candidate)
            if token == "perl" and any(
                part.startswith("-p") or part.startswith("-0p") for part in argv[index + 1 :]
            ):
                for candidate in reversed(argv[index + 1 :]):
                    if _looks_like_text_rewrite_target(candidate):
                        return _strip_shell_token(candidate)

        redirect_target = _extract_redirect_target(argv)
        if redirect_target is not None:
            return redirect_target

    regex_match = re.search(
        r"(?:sed\s+-i(?:\s+''|\s+\"\"|\s+'[^']*'|\s+\"[^\"]*\")?.*?|perl\s+-[0-9]*p[i0-9-]*.*?)\s+([^\s\"';|&]+(?:\.[A-Za-z0-9]+)?)",
        normalized,
    )
    if regex_match:
        candidate = _strip_shell_token(regex_match.group(1))
        if _looks_like_text_rewrite_target(candidate):
            return candidate

    redirect_match = re.search(r"(?:>>?|tee(?:\s+-a)?)\s+([^\s\"';|&]+)", normalized)
    if redirect_match:
        candidate = _strip_shell_token(redirect_match.group(1))
        if _looks_like_text_rewrite_target(candidate):
            return candidate

    return None


class ActionTracker:
    """Tracks completed actions to prevent duplicates and detect loops."""

    MAX_SEQUENCE_LENGTH = 20
    LOOP_PATTERN_MIN = 2
    LOOP_REPEAT_THRESHOLD = 2
    MAX_RESPONSE_HISTORY = 5
    OBSERVATION_REPEAT_WINDOW = 8
    READ_REPEAT_THRESHOLD = 3
    SEARCH_REPEAT_THRESHOLD = 2
    BASH_OBSERVATION_REPEAT_THRESHOLD = 2
    RECENT_PATH_CONTEXT_LIMIT = 12

    def __init__(self) -> None:
        self._file_writes: dict[str, list[str]] = {}
        self._files_edited: dict[str, list[str]] = {}
        self._commands_run: set[str] = set()
        self._dirs_created: set[str] = set()
        self._action_sequence: list[str] = []
        self._response_history: list[str] = []
        self._action_index = 0
        self._mutation_epoch = 0
        self._recent_reads: dict[str, tuple[int, int, int]] = {}
        self._recent_searches: dict[str, tuple[int, int, int]] = {}
        self._recent_bash_observations: dict[str, tuple[int, int, int]] = {}
        self._recent_path_contexts: list[str] = []

    def reset(self) -> None:
        self._file_writes.clear()
        self._files_edited.clear()
        self._commands_run.clear()
        self._dirs_created.clear()
        self._action_sequence.clear()
        self._response_history.clear()
        self._action_index = 0
        self._mutation_epoch = 0
        self._recent_reads.clear()
        self._recent_searches.clear()
        self._recent_bash_observations.clear()
        self._recent_path_contexts.clear()

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

    def would_duplicate_raw_patch(self, file_path: str, patch_text: str) -> bool:
        norm_path = self._normalize_path(file_path)
        sig = str(hash(patch_text))
        return sig in self._files_edited.get(norm_path, [])

    def would_duplicate_command(self, command: str) -> bool:
        norm_cmd = self._normalize_command(command)
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
        norm_cmd = self._normalize_command(command)
        self._commands_run.add(norm_cmd)

        mkdir_match = re.match(r'mkdir\s+(-p\s+)?(.+)', norm_cmd)
        if mkdir_match:
            dir_path = mkdir_match.group(2).strip().strip('"\'')
            self._dirs_created.add(self._normalize_path(dir_path))

    def record_mkdir(self, dir_path: str) -> None:
        self._dirs_created.add(self._normalize_path(dir_path))

    def recent_path_contexts(self) -> list[str]:
        return list(self._recent_path_contexts)

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
            raw_patch = arguments.get("patch") or arguments.get("diff") or arguments.get("patch_text")
            if isinstance(hunks, list) and hunks and self.would_duplicate_patch(file_path, hunks):
                return True, f"Same patch already applied to: {file_path}"
            if isinstance(raw_patch, str) and raw_patch.strip():
                if self.would_duplicate_raw_patch(file_path, raw_patch):
                    return True, f"Same patch already applied to: {file_path}"

        elif tool_name == "read":
            read_key = self._make_read_key(arguments)
            if read_key:
                duplicate, reason = self._check_recent_observation(
                    self._recent_reads,
                    read_key,
                    (
                        "Already read "
                        f"{str(arguments.get('file_path', '')).strip()} "
                        "recently without any intervening changes; "
                        "reuse the earlier read result instead of rereading"
                    ),
                    repeat_threshold=self.READ_REPEAT_THRESHOLD,
                )
                if duplicate:
                    return True, reason

        elif tool_name in {"glob", "grep"}:
            observation_key = self._make_search_key(tool_name, arguments)
            if observation_key:
                duplicate, reason = self._check_recent_observation(
                    self._recent_searches,
                    observation_key,
                    (
                        "Already ran the same search recently without any intervening "
                        "changes; reuse the earlier search result instead of rerunning it"
                    ),
                    repeat_threshold=self.SEARCH_REPEAT_THRESHOLD,
                )
                if duplicate:
                    return True, reason

        elif tool_name == "bash":
            command = str(arguments.get("command", "")).strip()
            if self._is_observational_bash(command):
                duplicate, reason = self._check_recent_observation(
                    self._recent_bash_observations,
                    self._normalize_command(command),
                    (
                        "Already ran the same read-only shell probe recently without any "
                        "intervening changes; reuse the earlier shell output instead of rerunning it"
                    ),
                    repeat_threshold=self.BASH_OBSERVATION_REPEAT_THRESHOLD,
                )
                if duplicate:
                    return True, reason

        # Bash commands intentionally skip exact-command dedupe here.
        # Re-running the same shell probe after a filesystem change is often valid,
        # and higher-level loop detection is a safer backstop than blocking `ls`.
        return False, ""

    def record_tool_call(self, tool_name: str, arguments: dict) -> None:
        self._action_index += 1
        self._action_sequence.append(tool_name)
        if len(self._action_sequence) > self.MAX_SEQUENCE_LENGTH:
            self._action_sequence.pop(0)

        if tool_name == "write":
            file_path = arguments.get("file_path", "")
            content = arguments.get("content", "")
            if file_path:
                self.record_file_create(file_path, content)
                self._record_path_context(file_path)
                self._note_mutation()

        elif tool_name == "edit":
            file_path = arguments.get("file_path", "")
            old_string = arguments.get("old_string", "")
            new_string = arguments.get("new_string", "")
            if file_path:
                self.record_edit(file_path, old_string, new_string)
                self._record_path_context(file_path)
                self._note_mutation()

        elif tool_name == "patch":
            file_path = arguments.get("file_path", "")
            hunks = arguments.get("hunks", [])
            if file_path:
                raw_patch = arguments.get("patch") or arguments.get("diff") or arguments.get("patch_text")
                if isinstance(hunks, list) and hunks:
                    self.record_edit(file_path, str(hunks), "structured_patch")
                elif isinstance(raw_patch, str) and raw_patch.strip():
                    self.record_edit(file_path, raw_patch, "raw_patch")
                self._record_path_context(file_path)
                self._note_mutation()

        elif tool_name == "read":
            read_key = self._make_read_key(arguments)
            if read_key:
                self._record_observation(
                    self._recent_reads,
                    read_key,
                )
            file_path = str(arguments.get("file_path", "")).strip()
            if file_path:
                self._record_path_context(file_path)

        elif tool_name in {"glob", "grep"}:
            observation_key = self._make_search_key(tool_name, arguments)
            if observation_key:
                self._record_observation(
                    self._recent_searches,
                    observation_key,
                )
            search_path = str(arguments.get("path", "")).strip()
            if search_path:
                self._record_path_context(search_path, is_directory_hint=True)

        elif tool_name == "bash":
            command = arguments.get("command", "")
            if command:
                self.record_command(command)
                if self._is_mutating_bash(command):
                    self._note_mutation()
                elif self._is_observational_bash(command):
                    self._record_observation(
                        self._recent_bash_observations,
                        self._normalize_command(command),
                    )

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

    @staticmethod
    def _normalize_command(command: str) -> str:
        return " ".join(command.split())

    def _note_mutation(self) -> None:
        self._mutation_epoch += 1

    def _check_recent_observation(
        self,
        cache: dict[str, tuple[int, int, int]],
        key: str,
        reason: str,
        *,
        repeat_threshold: int,
    ) -> tuple[bool, str]:
        last_seen = cache.get(key)
        if last_seen is None:
            return False, ""

        last_epoch, last_index, repeat_count = last_seen
        if last_epoch != self._mutation_epoch:
            return False, ""
        gap = self._action_index - last_index
        if gap > self.OBSERVATION_REPEAT_WINDOW:
            return False, ""
        if gap <= 0:
            return True, reason
        if repeat_count >= repeat_threshold:
            return True, reason
        return False, ""

    def _record_observation(
        self,
        cache: dict[str, tuple[int, int, int]],
        key: str,
    ) -> None:
        last_seen = cache.get(key)
        if last_seen is None:
            cache[key] = (self._mutation_epoch, self._action_index, 1)
            return

        last_epoch, last_index, repeat_count = last_seen
        gap = self._action_index - last_index
        if last_epoch != self._mutation_epoch or gap > self.OBSERVATION_REPEAT_WINDOW:
            cache[key] = (self._mutation_epoch, self._action_index, 1)
            return

        cache[key] = (
            self._mutation_epoch,
            self._action_index,
            repeat_count + 1,
        )

    def _make_search_key(self, tool_name: str, arguments: dict) -> str | None:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return None
        path = str(arguments.get("path", "")).strip()
        normalized_path = self._normalize_path(path) if path else ""
        return f"{tool_name}:{normalized_path}:{pattern}"

    def _make_read_key(self, arguments: dict) -> str | None:
        file_path = str(arguments.get("file_path", "")).strip()
        if not file_path:
            return None
        offset = str(arguments.get("offset", "")).strip()
        limit = str(arguments.get("limit", "")).strip()
        return (
            f"{self._normalize_path(file_path)}"
            f":offset={offset or 'full'}"
            f":limit={limit or 'all'}"
        )

    def _is_observational_bash(self, command: str) -> bool:
        norm_cmd = self._normalize_command(command)
        if not norm_cmd:
            return False
        if any(token in norm_cmd for token in ("&&", "||", ";", ">", ">>", "|", "<", "$(", "`")):
            return False
        try:
            argv = shlex.split(norm_cmd)
        except ValueError:
            return False
        if not argv:
            return False
        return argv[0] in {"ls", "pwd", "find", "stat", "cat", "head", "tail", "rg"}

    def _is_mutating_bash(self, command: str) -> bool:
        norm_cmd = self._normalize_command(command)
        if not norm_cmd:
            return False
        if extract_shell_text_rewrite_target(norm_cmd) is not None:
            return True
        mutating_fragments = (
            " >",
            ">>",
            "| tee",
            "touch ",
            "mkdir ",
            "rm ",
            "mv ",
            "cp ",
            "sed -i",
            "perl -pi",
            "git add",
            "git commit",
            "git apply",
        )
        if any(fragment in norm_cmd for fragment in mutating_fragments):
            return True
        try:
            argv = shlex.split(norm_cmd)
        except ValueError:
            return False
        if not argv:
            return False
        return argv[0] in {"touch", "mkdir", "rm", "mv", "cp", "chmod", "chown"}

    def _record_path_context(self, path_value: str, *, is_directory_hint: bool = False) -> None:
        normalized = self._normalize_path(path_value)
        path = Path(normalized)
        primary_dir = path if is_directory_hint or path.is_dir() else path.parent
        candidate_dirs = [primary_dir]
        if primary_dir.parent != primary_dir:
            candidate_dirs.append(primary_dir.parent)

        for candidate_dir in candidate_dirs:
            normalized_dir = self._normalize_path(str(candidate_dir))
            if normalized_dir in self._recent_path_contexts:
                self._recent_path_contexts.remove(normalized_dir)
            self._recent_path_contexts.insert(0, normalized_dir)

        if len(self._recent_path_contexts) > self.RECENT_PATH_CONTEXT_LIMIT:
            del self._recent_path_contexts[self.RECENT_PATH_CONTEXT_LIMIT :]

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

        rewrite_target = extract_shell_text_rewrite_target(str(command))
        if rewrite_target is not None:
            return ValidationResult(
                valid=False,
                reason="Shell-based text rewrites are brittle and bypass Loader's safer file tools",
                suggestion=(
                    f"Use edit/patch/write for `{rewrite_target}` instead of rewriting it with bash"
                ),
                severity="error",
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

        sibling_result = self._validate_numbered_sibling_conflict(str(file_path))
        if not sibling_result.valid:
            return sibling_result

        html_declared_file_result = self._validate_html_declared_file_creation(
            str(file_path),
        )
        if not html_declared_file_result.valid:
            return html_declared_file_result

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

        html_declared_target_result = self._validate_html_declared_target_set(
            str(file_path),
            str(content),
        )
        if not html_declared_target_result.valid:
            return html_declared_target_result

        html_root_coverage_result = self._validate_html_root_link_coverage(
            str(file_path),
            str(content),
        )
        if not html_root_coverage_result.valid:
            return html_root_coverage_result

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

        prospective_content = self._prospective_edit_content(
            str(file_path),
            str(old_string),
            str(new_string),
        )

        html_index_result = self._validate_html_index_links(
            str(file_path),
            prospective_content,
        )
        if not html_index_result.valid:
            return html_index_result

        html_declared_target_result = self._validate_html_declared_target_set(
            str(file_path),
            prospective_content,
        )
        if not html_declared_target_result.valid:
            return html_declared_target_result

        return ValidationResult(valid=True)

    def _validate_patch(self, arguments: dict) -> ValidationResult:
        file_path = arguments.get("file_path", "")
        hunks = arguments.get("hunks", [])
        raw_patch = arguments.get("patch") or arguments.get("diff") or arguments.get("patch_text")

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

        sibling_result = self._validate_numbered_sibling_conflict(str(file_path))
        if not sibling_result.valid:
            return sibling_result

        html_declared_file_result = self._validate_html_declared_file_creation(
            str(file_path),
        )
        if not html_declared_file_result.valid:
            return html_declared_file_result

        has_hunks = isinstance(hunks, list) and bool(hunks)
        has_raw_patch = isinstance(raw_patch, str) and bool(raw_patch.strip())
        if not has_hunks and not has_raw_patch:
            return ValidationResult(
                valid=False,
                reason="Patch hunks are missing",
                suggestion="Provide structured patch hunks or a unified diff patch string",
                severity="error",
            )

        return ValidationResult(valid=True)

    def _validate_numbered_sibling_conflict(self, file_path: str) -> ValidationResult:
        path = Path(file_path).expanduser()
        if path.exists() or not path.suffix or not path.parent.exists():
            return ValidationResult(valid=True)

        prefix_match = re.match(r"^(\d+)[-_]", path.name)
        if prefix_match is None:
            return ValidationResult(valid=True)

        prefix = prefix_match.group(1)
        siblings = sorted(
            candidate
            for candidate in path.parent.iterdir()
            if (
                candidate.is_file()
                and candidate.suffix == path.suffix
                and candidate.name != path.name
                and re.match(rf"^{re.escape(prefix)}[-_]", candidate.name)
            )
        )
        if not siblings:
            return ValidationResult(valid=True)

        preview = ", ".join(candidate.name for candidate in siblings[:3])
        if len(siblings) > 3:
            preview += ", ..."
        return ValidationResult(
            valid=False,
            reason="New file conflicts with an existing numbered sibling",
            suggestion=(
                f"Reuse the confirmed numbered file in `{path.parent}` instead of "
                f"creating an alternate filename for step {prefix}, for example: {preview}"
            ),
            severity="error",
        )

    def _validate_html_declared_file_creation(
        self,
        file_path: str,
    ) -> ValidationResult:
        normalized = Path(file_path).expanduser()
        if normalized.exists():
            return ValidationResult(valid=True)
        if normalized.suffix.lower() not in {".html", ".htm"}:
            return ValidationResult(valid=True)
        if normalized.name.lower() == "index.html":
            return ValidationResult(valid=True)

        root = self._resolve_html_artifact_root(normalized)
        current_relative = self._relative_html_target(root, normalized)
        if current_relative is None:
            return ValidationResult(valid=True)

        declared_targets, authoritative_root_graph = self._collect_declared_html_targets(
            root,
            normalized,
        )
        if not declared_targets and not authoritative_root_graph:
            return ValidationResult(valid=True)
        if current_relative in declared_targets:
            return ValidationResult(valid=True)

        declared_preview = ", ".join(sorted(declared_targets)[:3])
        if authoritative_root_graph:
            root_index = (root / "index.html").resolve(strict=False)
            suggestion = (
                "Keep new non-root HTML files within the root-declared artifact set and "
                f"update the guide root `{root_index}` before creating undeclared sibling pages, "
                f"for example: {current_relative}"
            )
        else:
            suggestion = (
                "Keep new non-root HTML files within the current declared artifact set and "
                f"avoid creating undeclared sibling pages, for example: {current_relative}"
            )
        if declared_preview:
            suggestion += f". Already-declared local targets include: {declared_preview}"
        declared_suggestions = self._suggest_declared_html_targets(
            declared_targets,
            [current_relative],
        )
        if declared_suggestions:
            suggestion += (
                ". Closest declared local targets include: "
                + ", ".join(declared_suggestions[:3])
            )
        return ValidationResult(
            valid=False,
            reason="HTML file creation falls outside the current declared artifact set",
            suggestion=suggestion,
            severity="error",
        )

    def _validate_read(self, arguments: dict) -> ValidationResult:
        file_path = arguments.get("file_path", "")

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

        sibling_result = self._validate_numbered_sibling_conflict(str(file_path))
        if not sibling_result.valid:
            return ValidationResult(
                valid=False,
                reason="Read target conflicts with an existing numbered sibling",
                suggestion=sibling_result.suggestion,
                severity="error",
            )
        return path_result

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

    def _validate_html_index_links(
        self,
        file_path: str,
        content: str,
    ) -> ValidationResult:
        normalized = Path(file_path).expanduser()
        if normalized.suffix.lower() != ".html" or "<a " not in content:
            return ValidationResult(valid=True)

        link_pairs = re.findall(r'<a\s+href="([^"]+)">([^<]+)</a>', content)
        if not link_pairs:
            return ValidationResult(valid=True)

        root = normalized.parent
        missing: list[str] = []
        for href, _label in link_pairs:
            target_text = href.strip()
            if not target_text or target_text.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            if "://" in target_text:
                continue
            target = (root / href).resolve(strict=False)
            if not target.exists():
                if href not in missing:
                    missing.append(href)

        if missing:
            if self._allows_root_html_graph_seed(str(file_path), str(content), missing):
                return ValidationResult(valid=True)
            preview = ", ".join(missing[:3])
            if len(missing) > 3:
                preview += ", ..."
            return ValidationResult(
                valid=False,
                reason="Edited HTML links point to files that do not exist",
                suggestion=(
                    "Use only existing local targets for href values and avoid "
                    f"introducing missing links, for example fix: {preview}"
                ),
                severity="error",
            )

        return ValidationResult(valid=True)

    def _prospective_edit_content(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
    ) -> str:
        if old_string == "":
            return new_string

        normalized = Path(file_path).expanduser()
        try:
            current = normalized.read_text()
        except OSError:
            return new_string

        if old_string not in current:
            return new_string
        return current.replace(old_string, new_string, 1)

    def _allows_root_html_graph_seed(
        self,
        file_path: str,
        content: str,
        missing: list[str],
    ) -> bool:
        normalized = Path(file_path).expanduser()
        if normalized.suffix.lower() not in {".html", ".htm"}:
            return False
        if normalized.name.lower() != "index.html":
            return False

        root = self._resolve_html_artifact_root(normalized)
        missing_after = self._collect_missing_local_html_targets(normalized, content)
        if not missing_after:
            return False
        existing_missing = self._collect_existing_missing_local_html_targets(normalized)
        if len(missing_after) > len(existing_missing):
            declared_targets, authoritative_root_graph = self._collect_declared_html_targets(
                root,
                normalized,
            )
            if not authoritative_root_graph:
                return False
            newly_missing = [
                href
                for href in missing_after
                if href not in existing_missing
            ]
            if not newly_missing:
                return False
            if any(
                not self._is_next_ordered_html_target(root, href, declared_targets)
                for href in newly_missing
            ):
                return False

        for href in missing:
            resolved = (normalized.parent / href).resolve(strict=False)
            relative = self._relative_html_target(root, resolved)
            if relative is None:
                return False
        return True

    def _is_next_ordered_html_target(
        self,
        root: Path,
        href: str,
        declared_targets: set[str],
    ) -> bool:
        relative_href = self._relative_html_target(root, (root / href).resolve(strict=False))
        if relative_href is None:
            return False

        expected_number = _ordered_html_target_number(relative_href)
        if expected_number is None:
            return False

        parent = Path(relative_href).parent
        sibling_numbers = sorted(
            number
            for target in declared_targets
            if Path(target).parent == parent
            if (number := _ordered_html_target_number(target)) is not None
        )
        if not sibling_numbers:
            return False

        min_number = sibling_numbers[0]
        max_number = sibling_numbers[-1]
        if expected_number != max_number + 1:
            return False

        return sibling_numbers == list(range(min_number, max_number + 1))

    def _collect_existing_missing_local_html_targets(self, file_path: Path) -> list[str]:
        try:
            current = file_path.read_text()
        except OSError:
            return []
        return self._collect_missing_local_html_targets(file_path, current)

    def _collect_missing_local_html_targets(
        self,
        file_path: Path,
        content: str,
    ) -> list[str]:
        missing: list[str] = []
        for href, resolved in self._collect_local_html_targets(file_path, content):
            if resolved.exists():
                continue
            if href not in missing:
                missing.append(href)
        return missing

    def _validate_html_declared_target_set(
        self,
        file_path: str,
        content: str,
    ) -> ValidationResult:
        normalized = Path(file_path).expanduser()
        if normalized.suffix.lower() != ".html" or normalized.name.lower() == "index.html":
            return ValidationResult(valid=True)

        local_targets = self._collect_local_html_targets(normalized, content)
        if not local_targets:
            return ValidationResult(valid=True)

        root = self._resolve_html_artifact_root(normalized)
        current_relative = self._relative_html_target(root, normalized)
        declared_targets, authoritative_root_graph = self._collect_declared_html_targets(root, normalized)
        if not declared_targets and not authoritative_root_graph:
            return ValidationResult(valid=True)

        undeclared_targets: list[str] = []
        for href, resolved in local_targets:
            relative_target = self._relative_html_target(root, resolved)
            if relative_target is None:
                continue
            if relative_target == "index.html" or relative_target == current_relative:
                continue
            if relative_target in declared_targets:
                continue
            if not authoritative_root_graph and resolved.exists():
                continue
            if href not in undeclared_targets:
                undeclared_targets.append(href)

        if not undeclared_targets:
            return ValidationResult(valid=True)

        preview = ", ".join(undeclared_targets[:3])
        if len(undeclared_targets) > 3:
            preview += ", ..."
        declared_preview = ", ".join(sorted(declared_targets)[:3])
        if authoritative_root_graph:
            suggestion = (
                "Keep non-root HTML pages within the root-declared local-link set and "
                "avoid introducing new sibling targets that the guide root does not declare; "
                f"remove or replace undeclared hrefs like: {preview}"
            )
        else:
            suggestion = (
                "Keep non-root HTML pages within the current declared local-link set and "
                f"avoid introducing new missing sibling targets; remove or replace undeclared hrefs like: {preview}"
            )
        if declared_preview:
            suggestion += f". Already-declared local targets include: {declared_preview}"
        declared_suggestions = self._suggest_declared_html_targets(
            declared_targets,
            undeclared_targets,
        )
        if declared_suggestions:
            suggestion += (
                ". Closest declared local targets include: "
                + ", ".join(declared_suggestions[:3])
            )
        return ValidationResult(
            valid=False,
            reason="HTML page introduces new local targets outside the current declared artifact set",
            suggestion=suggestion,
            severity="error",
        )

    def _collect_local_html_targets(
        self,
        file_path: Path,
        content: str,
    ) -> list[tuple[str, Path]]:
        pattern = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
        targets: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for href in pattern.findall(content):
            target_text = href.strip()
            if not self._is_local_html_link_target(target_text):
                continue
            resolved = (file_path.parent / target_text).resolve(strict=False)
            key = f"{target_text}::{resolved}"
            if key in seen:
                continue
            seen.add(key)
            targets.append((target_text, resolved))
        return targets

    def _collect_declared_html_targets(
        self,
        root: Path,
        current_file: Path,
    ) -> tuple[set[str], bool]:
        root_index = root / "index.html"
        if root_index.exists():
            try:
                root_text = root_index.read_text()
            except OSError:
                root_text = ""
            declared_from_root = {
                relative_target
                for _href, resolved in self._collect_local_html_targets(root_index, root_text)
                if (relative_target := self._relative_html_target(root, resolved)) is not None
            }
            if declared_from_root:
                return declared_from_root, True

        html_files = [
            path
            for path in root.rglob("*.html")
            if path.is_file() and path != current_file
        ]
        declared: set[str] = set()
        for html_file in html_files:
            try:
                text = html_file.read_text()
            except OSError:
                continue
            for _href, resolved in self._collect_local_html_targets(html_file, text):
                relative_target = self._relative_html_target(root, resolved)
                if relative_target is not None:
                    declared.add(relative_target)
        return declared, False

    def _resolve_html_artifact_root(self, file_path: Path) -> Path:
        for candidate in [file_path.parent, *file_path.parents]:
            if (candidate / "index.html").exists():
                return candidate
        return file_path.parent

    def _relative_html_target(self, root: Path, target: Path) -> str | None:
        try:
            normalized_root = root.resolve(strict=False)
        except OSError:
            normalized_root = root.expanduser()
        try:
            normalized_target = target.resolve(strict=False)
        except OSError:
            normalized_target = target.expanduser()
        try:
            return str(normalized_target.relative_to(normalized_root))
        except ValueError:
            return None

    @staticmethod
    def _is_local_html_link_target(href: str) -> bool:
        target = href.strip()
        if not target:
            return False
        if target.startswith(("#", "mailto:", "tel:", "javascript:")):
            return False
        if "://" in target:
            return False
        normalized = target.split("#", 1)[0].split("?", 1)[0].strip().lower()
        return normalized.endswith(".html")

    def _suggest_existing_html_targets(self, root: Path, missing: list[str]) -> list[str]:
        available_by_directory: dict[Path, list[str]] = {}
        suggestions: list[str] = []

        for href in missing:
            href_path = Path(href)
            directory = (root / href_path).parent
            if directory not in available_by_directory:
                available_by_directory[directory] = sorted(
                    str(path.relative_to(root))
                    for path in directory.glob("*.html")
                    if path.is_file()
                )

            available = available_by_directory[directory]
            if not available:
                continue

            missing_name = href_path.name
            chapter_match = re.match(r"(\d+)-", missing_name)
            preferred = available
            if chapter_match is not None:
                prefix = f"{chapter_match.group(1)}-"
                same_prefix = [
                    candidate
                    for candidate in available
                    if Path(candidate).name.startswith(prefix)
                ]
                if same_prefix:
                    preferred = same_prefix

            matched_names = get_close_matches(
                missing_name,
                [Path(candidate).name for candidate in preferred],
                n=1,
                cutoff=0.0,
            )
            if matched_names:
                matched_name = matched_names[0]
                candidate = next(
                    (
                        candidate
                        for candidate in preferred
                        if Path(candidate).name == matched_name
                    ),
                    None,
                )
                if candidate is not None and candidate not in suggestions:
                    suggestions.append(candidate)

        return suggestions

    def _suggest_declared_html_targets(
        self,
        declared_targets: set[str],
        undeclared_targets: list[str],
    ) -> list[str]:
        suggestions: list[str] = []
        available = sorted(declared_targets)
        available_names = [Path(candidate).name for candidate in available]

        for href in undeclared_targets:
            href_name = Path(href).name
            chapter_match = re.match(r"(\d+)[-_]", href_name)
            preferred = available
            preferred_names = available_names
            same_prefix_match = False
            if chapter_match is not None:
                prefix = f"{chapter_match.group(1)}-"
                filtered = [
                    candidate
                    for candidate in available
                    if Path(candidate).name.startswith(prefix)
                ]
                if filtered:
                    preferred = filtered
                    preferred_names = [Path(candidate).name for candidate in filtered]
                    same_prefix_match = True

            matched_names = get_close_matches(
                href_name,
                preferred_names,
                n=1,
                cutoff=0.0,
            )
            if not matched_names:
                continue

            candidate = next(
                (
                    declared
                    for declared in preferred
                    if Path(declared).name == matched_names[0]
                ),
                None,
            )
            if candidate is not None and not same_prefix_match:
                href_tokens = _html_target_tokens(href)
                candidate_tokens = _html_target_tokens(candidate)
                if not href_tokens.intersection(candidate_tokens):
                    continue
            if candidate is not None and candidate not in suggestions:
                suggestions.append(candidate)

        return suggestions
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

    def _validate_html_root_link_coverage(
        self,
        file_path: str,
        content: str,
    ) -> ValidationResult:
        normalized = Path(file_path).expanduser()
        if normalized.suffix.lower() != ".html" or normalized.name.lower() != "index.html":
            return ValidationResult(valid=True)
        if not normalized.exists():
            return ValidationResult(valid=True)

        root = self._resolve_html_artifact_root(normalized)
        try:
            existing_text = normalized.read_text()
        except OSError:
            return ValidationResult(valid=True)

        existing_targets = {
            relative_target
            for _href, resolved in self._collect_local_html_targets(normalized, existing_text)
            if (relative_target := self._relative_html_target(root, resolved)) is not None
            and resolved.exists()
        }
        if not existing_targets:
            return ValidationResult(valid=True)

        new_targets = {
            relative_target
            for _href, resolved in self._collect_local_html_targets(normalized, content)
            if (relative_target := self._relative_html_target(root, resolved)) is not None
        }
        dropped_targets = sorted(existing_targets - new_targets)
        if not dropped_targets:
            return ValidationResult(valid=True)

        preview = ", ".join(dropped_targets[:3])
        if len(dropped_targets) > 3:
            preview += ", ..."
        return ValidationResult(
            valid=False,
            reason="Edited HTML root page drops links to existing local pages",
            suggestion=(
                "Keep the existing local page set linked from the root HTML page "
                f"unless you are intentionally removing those files, for example restore: {preview}"
            ),
            severity="error",
        )
