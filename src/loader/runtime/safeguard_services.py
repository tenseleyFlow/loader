"""Runtime-owned safeguard services shared by hooks and agent adapters."""

from __future__ import annotations

import re
import shlex
from difflib import get_close_matches
from dataclasses import dataclass
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


def extract_html_title_from_text(payload: str) -> str | None:
    """Extract one human-readable HTML title from raw file contents."""

    for pattern in (r"<h1[^>]*>(.*?)</h1>", r"<title[^>]*>(.*?)</title>"):
        match = re.search(pattern, payload, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        title = re.sub(r"<[^>]+>", " ", match.group(1))
        normalized = " ".join(title.split()).strip()
        if normalized:
            return normalized
    return None


def read_html_title(path: Path) -> str:
    """Read one HTML file title for inventory and validation helpers."""

    try:
        return extract_html_title_from_text(path.read_text()) or ""
    except OSError:
        return ""


def format_html_inventory_entry(root: Path, candidate: Path) -> str:
    """Format one exact href/title pair for model-facing guidance."""

    normalized_root = root.expanduser().resolve(strict=False)
    normalized_candidate = candidate.expanduser().resolve(strict=False)
    try:
        href = str(normalized_candidate.relative_to(normalized_root))
    except ValueError:
        href = normalized_candidate.name
    title = read_html_title(candidate)
    if title:
        return f"{href} = {title}"
    return href


def _collect_html_inventory_entries(index_path: str | Path) -> list[tuple[str, str]]:
    """Return exact href/title pairs for sibling HTML chapters."""

    index = Path(index_path).expanduser()
    if index.name != "index.html":
        return []

    chapters_dir = index.parent / "chapters"
    if not chapters_dir.is_dir():
        return []

    entries: list[tuple[str, str]] = []
    for candidate in sorted(chapters_dir.glob("*.html")):
        if not candidate.is_file():
            continue
        title = read_html_title(candidate)
        if not title:
            continue
        href = format_html_inventory_entry(index.parent, candidate).split(" = ", 1)[0]
        entries.append((href, title))
    return entries


def summarize_html_inventory(
    index_path: str | Path,
    *,
    limit: int | None = 12,
) -> str | None:
    """Summarize the existing sibling HTML inventory for one index page."""

    index = Path(index_path).expanduser()
    if index.name != "index.html":
        return None

    entries = [f"{href} = {title}" for href, title in _collect_html_inventory_entries(index)]
    if not entries:
        return None

    if limit is not None and len(entries) > limit:
        return "; ".join(entries[:limit]) + "; ..."
    return "; ".join(entries)


def extract_html_toc_excerpt(
    index_path: str | Path,
    *,
    max_lines: int = 16,
) -> str | None:
    """Extract the current HTML table-of-contents block for recovery guidance."""

    index = Path(index_path).expanduser()
    if index.name != "index.html":
        return None

    try:
        text = index.read_text()
    except OSError:
        return None

    match = re.search(
        r"(<h2[^>]*>\s*Table of Contents\s*</h2>.*?</ul>)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r"(<ul[^>]*class=\"[^\"]*chapter-list[^\"]*\"[^>]*>.*?</ul>)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    if not match:
        return None

    snippet_lines = [line.rstrip() for line in match.group(1).splitlines() if line.strip()]
    if not snippet_lines:
        return None
    if len(snippet_lines) > max_lines:
        snippet_lines = snippet_lines[:max_lines] + ["..."]
    return "\n".join(snippet_lines)


def build_html_toc_replacement_block(index_path: str | Path) -> str | None:
    """Build one exact replacement TOC block from the verified sibling inventory."""

    entries = _collect_html_inventory_entries(index_path)
    if not entries:
        return None

    excerpt = extract_html_toc_excerpt(index_path, max_lines=64)
    excerpt_lines = excerpt.splitlines() if excerpt else []

    heading_line = next(
        (line.rstrip() for line in excerpt_lines if "<h2" in line.lower()),
        "<h2>Table of Contents</h2>",
    )
    ul_line = next(
        (
            line.rstrip()
            for line in excerpt_lines
            if "<ul" in line.lower() and "chapter-list" in line.lower()
        ),
        '        <ul class="chapter-list">',
    )
    li_indent = next(
        (
            re.match(r"^\s*", line).group(0)
            for line in excerpt_lines
            if "<li><a " in line
        ),
        re.match(r"^\s*", ul_line).group(0) + "    ",
    )
    closing_line = next(
        (line.rstrip() for line in excerpt_lines if "</ul>" in line.lower()),
        f"{re.match(r'^\s*', ul_line).group(0)}</ul>",
    )

    lines = [heading_line, ul_line]
    lines.extend(
        f'{li_indent}<li><a href="{href}">{title}</a></li>'
        for href, title in entries
    )
    lines.append(closing_line)
    return "\n".join(lines)


def build_html_toc_edit_call_template(index_path: str | Path) -> str | None:
    """Build one concrete `edit(...)` template for replacing the TOC block."""

    index = Path(index_path).expanduser()
    excerpt = extract_html_toc_excerpt(index, max_lines=64)
    replacement = build_html_toc_replacement_block(index)
    if not excerpt or not replacement:
        return None

    return "\n".join(
        [
            "edit(",
            f'  file_path="{index}",',
            '  old_string="""',
            excerpt,
            '""",',
            '  new_string="""',
            replacement,
            '"""',
            ")",
        ]
    )


@dataclass(frozen=True)
class HtmlTocValidationResult:
    """Semantic validation result for one chapter-list table of contents."""

    valid: bool
    link_count: int
    missing: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()


def validate_html_toc(index_path: str | Path) -> HtmlTocValidationResult | None:
    """Validate that one HTML index TOC points at real chapter files with matching titles."""

    index = Path(index_path).expanduser()
    if index.name != "index.html":
        return None

    try:
        text = index.read_text()
    except OSError:
        return None

    section_match = re.search(r'<ul class="chapter-list">(.*?)</ul>', text, re.S)
    if section_match is None:
        return HtmlTocValidationResult(
            valid=False,
            link_count=0,
            missing=("Missing chapter-list table of contents",),
        )

    links = re.findall(r'<a href="([^"]+)">([^<]+)</a>', section_match.group(1))
    if not links:
        return HtmlTocValidationResult(
            valid=False,
            link_count=0,
            missing=("No chapter links found in table of contents",),
        )

    root = index.parent
    missing: list[str] = []
    mismatched: list[str] = []
    for href, label in links:
        target = (root / href).expanduser().resolve(strict=False)
        if not target.exists():
            missing.append(f"{href} -> missing")
            continue
        title = read_html_title(target)
        if title and label.strip() != title:
            mismatched.append(f"{href} -> {label.strip()} != {title}")

    return HtmlTocValidationResult(
        valid=not missing and not mismatched,
        link_count=len(links),
        missing=tuple(missing),
        mismatched=tuple(mismatched),
    )


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
    HTML_CHAPTER_EVIDENCE_THRESHOLD = 3
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
        self._recent_html_directory_reads: dict[str, tuple[int, set[str]]] = {}
        self._recent_path_contexts: list[str] = []
        self._validated_html_tocs: dict[str, int] = {}
        self._verified_html_inventory_dirs: set[str] = set()

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
        self._recent_html_directory_reads.clear()
        self._recent_path_contexts.clear()
        self._validated_html_tocs.clear()
        self._verified_html_inventory_dirs.clear()

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

    def note_validated_html_toc(self, index_path: str) -> None:
        """Record that one index currently satisfies the semantic chapter-link check."""

        normalized = self._normalize_path(index_path)
        if Path(normalized).name != "index.html":
            return
        self._validated_html_tocs[normalized] = self._mutation_epoch

    def note_verified_html_inventory(self, index_path: str) -> None:
        """Record that one sibling chapter inventory is already known exactly."""

        normalized = self._normalize_path(index_path)
        path = Path(normalized)
        chapters_dir = path if path.name == "chapters" else path.parent / "chapters"
        self._verified_html_inventory_dirs.add(self._normalize_path(str(chapters_dir)))

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
            inventory_duplicate, inventory_reason = self._check_verified_html_inventory_observation(
                tool_name,
                arguments,
            )
            if inventory_duplicate:
                return True, inventory_reason
            validated_duplicate, validated_reason = self._check_validated_html_toc_observation(
                tool_name,
                arguments,
            )
            if validated_duplicate:
                return True, validated_reason
            read_key = self._make_read_key(arguments)
            if read_key:
                sufficiency_duplicate, sufficiency_reason = (
                    self._check_html_observation_sufficiency(
                        tool_name,
                        arguments,
                    )
                )
                if sufficiency_duplicate:
                    return True, sufficiency_reason
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
            inventory_duplicate, inventory_reason = self._check_verified_html_inventory_observation(
                tool_name,
                arguments,
            )
            if inventory_duplicate:
                return True, inventory_reason
            validated_duplicate, validated_reason = self._check_validated_html_toc_observation(
                tool_name,
                arguments,
            )
            if validated_duplicate:
                return True, validated_reason
            observation_key = self._make_search_key(tool_name, arguments)
            if observation_key:
                sufficiency_duplicate, sufficiency_reason = (
                    self._check_html_observation_sufficiency(
                        tool_name,
                        arguments,
                    )
                )
                if sufficiency_duplicate:
                    return True, sufficiency_reason
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
                inventory_duplicate, inventory_reason = self._check_verified_html_inventory_observation(
                    tool_name,
                    arguments,
                )
                if inventory_duplicate:
                    return True, inventory_reason
                validated_duplicate, validated_reason = self._check_validated_html_toc_observation(
                    tool_name,
                    arguments,
                )
                if validated_duplicate:
                    return True, validated_reason
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
                self._clear_verified_html_inventory_for_path(file_path)
                self._note_mutation()

        elif tool_name == "edit":
            file_path = arguments.get("file_path", "")
            old_string = arguments.get("old_string", "")
            new_string = arguments.get("new_string", "")
            if file_path:
                self.record_edit(file_path, old_string, new_string)
                self._record_path_context(file_path)
                self._clear_verified_html_inventory_for_path(file_path)
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
                self._clear_verified_html_inventory_for_path(file_path)
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
            self._record_html_directory_read(arguments)

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
                    target = extract_shell_text_rewrite_target(command)
                    if target:
                        self._clear_verified_html_inventory_for_path(target)
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

    def _record_html_directory_read(self, arguments: dict) -> None:
        file_path = str(arguments.get("file_path", "")).strip()
        if not file_path:
            return
        normalized_path = self._normalize_path(file_path)
        path = Path(normalized_path)
        if path.suffix != ".html" or path.name == "index.html" or path.parent.name != "chapters":
            return

        directory = str(path.parent)
        last_seen = self._recent_html_directory_reads.get(directory)
        if last_seen is None or last_seen[0] != self._mutation_epoch:
            self._recent_html_directory_reads[directory] = (
                self._mutation_epoch,
                {path.name},
            )
            return

        _, seen_files = last_seen
        updated = set(seen_files)
        updated.add(path.name)
        self._recent_html_directory_reads[directory] = (
            self._mutation_epoch,
            updated,
        )

    def _check_html_observation_sufficiency(
        self,
        tool_name: str,
        arguments: dict,
    ) -> tuple[bool, str]:
        if tool_name == "read":
            file_path = str(arguments.get("file_path", "")).strip()
            if not file_path:
                return False, ""
            normalized_path = self._normalize_path(file_path)
            path = Path(normalized_path)
            if path.name != "index.html":
                return False, ""
            chapters_dir = str(path.parent / "chapters")
            chapter_count = self._chapter_evidence_count(chapters_dir)
            if chapter_count < self.HTML_CHAPTER_EVIDENCE_THRESHOLD:
                return False, ""
            read_key = self._make_read_key(arguments)
            if read_key is None:
                return False, ""
            last_seen = self._recent_reads.get(read_key)
            if last_seen is None:
                return False, ""
            _, _, repeat_count = last_seen
            if repeat_count < 2:
                return False, ""
            return (
                True,
                "Already confirmed multiple chapter files in the sibling chapters "
                "directory; reuse the known file/title evidence and update index.html "
                "instead of rereading it",
            )

        if tool_name in {"glob", "grep"}:
            search_path = str(arguments.get("path", "")).strip()
            if not search_path:
                return False, ""
            normalized_path = self._normalize_path(search_path)
            path = Path(normalized_path)
            if path.name != "chapters":
                return False, ""
            chapter_count = self._chapter_evidence_count(str(path))
            if chapter_count < self.HTML_CHAPTER_EVIDENCE_THRESHOLD:
                return False, ""
            observation_key = self._make_search_key(tool_name, arguments)
            if observation_key is None or observation_key not in self._recent_searches:
                return False, ""
            return (
                True,
                "Already confirmed multiple chapter files in this directory; reuse "
                "the known filename/title evidence and update the target index instead "
                "of rerunning the directory search",
            )

        return False, ""

    def _chapter_evidence_count(self, directory: str) -> int:
        last_seen = self._recent_html_directory_reads.get(directory)
        if last_seen is None:
            return 0
        last_epoch, seen_files = last_seen
        if last_epoch != self._mutation_epoch:
            return 0
        return len(seen_files)

    def _check_validated_html_toc_observation(
        self,
        tool_name: str,
        arguments: dict,
    ) -> tuple[bool, str]:
        related_paths = self._validated_html_related_paths(tool_name, arguments)
        if not related_paths:
            return False, ""

        for path in related_paths:
            if self._matches_validated_html_toc(path):
                return (
                    True,
                    "The current index.html already passes the validated chapter-link "
                    "check; stop rereading index.html or chapters/ and finish the task "
                    "unless a specific href or title is still unresolved",
                )
        return False, ""

    def _check_verified_html_inventory_observation(
        self,
        tool_name: str,
        arguments: dict,
    ) -> tuple[bool, str]:
        related_paths = self._verified_inventory_related_paths(tool_name, arguments)
        if not related_paths:
            return False, ""

        for path in related_paths:
            if self._matches_verified_html_inventory(path):
                return (
                    True,
                    "The verified chapter inventory already lists the exact href/title "
                    "pairs for this directory; update index.html from that inventory "
                    "instead of rereading chapter files",
                )
        return False, ""

    def _validated_html_related_paths(
        self,
        tool_name: str,
        arguments: dict,
    ) -> list[str]:
        if tool_name == "read":
            file_path = str(arguments.get("file_path", "")).strip()
            return [self._normalize_path(file_path)] if file_path else []

        if tool_name in {"glob", "grep"}:
            search_path = str(arguments.get("path", "")).strip()
            return [self._normalize_path(search_path)] if search_path else []

        if tool_name == "bash":
            command = str(arguments.get("command", "")).strip()
            if not command:
                return []
            return self._extract_observational_bash_paths(command)

        return []

    def _verified_inventory_related_paths(
        self,
        tool_name: str,
        arguments: dict,
    ) -> list[str]:
        if tool_name == "read":
            file_path = str(arguments.get("file_path", "")).strip()
            return [self._normalize_path(file_path)] if file_path else []

        if tool_name in {"glob", "grep"}:
            search_path = str(arguments.get("path", "")).strip()
            return [self._normalize_path(search_path)] if search_path else []

        if tool_name == "bash":
            command = str(arguments.get("command", "")).strip()
            if not command:
                return []
            return self._extract_observational_bash_paths(command)

        return []

    def _matches_validated_html_toc(self, path: str) -> bool:
        normalized = self._normalize_path(path)
        candidate = Path(normalized)
        for index_path, epoch in self._validated_html_tocs.items():
            if epoch != self._mutation_epoch:
                continue
            index = Path(index_path)
            chapters = Path(self._normalize_path(str(index.parent / "chapters")))
            if candidate == index or candidate == chapters:
                return True
            if candidate.parent == chapters:
                return True
        return False

    def _matches_verified_html_inventory(self, path: str) -> bool:
        normalized = self._normalize_path(path)
        candidate = Path(normalized)
        for directory in self._verified_html_inventory_dirs:
            chapters = Path(directory)
            if candidate == chapters or candidate.parent == chapters:
                return True
        return False

    def _clear_verified_html_inventory_for_path(self, path_value: str) -> None:
        normalized = self._normalize_path(path_value)
        candidate = Path(normalized)
        stale: set[str] = set()
        for directory in self._verified_html_inventory_dirs:
            chapters = Path(directory)
            if candidate == chapters or candidate.parent == chapters:
                stale.add(directory)
        self._verified_html_inventory_dirs.difference_update(stale)

    def _extract_observational_bash_paths(self, command: str) -> list[str]:
        norm_cmd = self._normalize_command(command)
        try:
            argv = shlex.split(norm_cmd)
        except ValueError:
            return []
        if not argv:
            return []

        paths: list[str] = []
        for token in argv[1:]:
            candidate = _strip_shell_token(token)
            if not candidate or candidate.startswith("-"):
                continue
            if any(marker in candidate for marker in ("/", "~")) or Path(candidate).suffix == ".html":
                paths.append(self._normalize_path(candidate))
                continue
            if candidate.rstrip("/").endswith("chapters"):
                paths.append(self._normalize_path(candidate))
        return paths


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

        html_index_result = self._validate_html_index_links(str(file_path), str(new_string))
        if not html_index_result.valid:
            return html_index_result

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

    def _validate_html_index_links(
        self,
        file_path: str,
        content: str,
    ) -> ValidationResult:
        normalized = Path(file_path).expanduser()
        if normalized.name != "index.html" or "<a " not in content:
            return ValidationResult(valid=True)

        link_pairs = re.findall(r'<a\s+href="([^"]+)">([^<]+)</a>', content)
        if not link_pairs:
            return ValidationResult(valid=True)

        root = normalized.parent
        missing: list[str] = []
        mismatched: list[str] = []
        for href, label in link_pairs:
            target = (root / href).resolve(strict=False)
            if not target.exists():
                if href not in missing:
                    missing.append(href)
                continue

            title = read_html_title(target)
            if title and label.strip() != title:
                if href not in mismatched:
                    mismatched.append(href)

        if missing:
            suggestions = self._suggest_existing_html_targets(root, missing)
            preview_items = [
                format_html_inventory_entry(root, root / suggestion)
                for suggestion in suggestions
            ]
            if not preview_items:
                preview_items = missing
            preview = ", ".join(preview_items[:3])
            if len(preview_items) > 3:
                preview += ", ..."
            return ValidationResult(
                valid=False,
                reason="Edited TOC references chapter files that do not exist",
                suggestion=(
                    "Use only existing chapter href/title pairs from beside index.html, for example: "
                    f"{preview}"
                ),
                severity="error",
            )

        if mismatched:
            exact_entries = [
                format_html_inventory_entry(root, (root / href).resolve(strict=False))
                for href in mismatched
                if (root / href).resolve(strict=False).exists()
            ]
            if not exact_entries:
                exact_entries = mismatched
            preview = "; ".join(exact_entries[:2])
            if len(exact_entries) > 2:
                preview += "; ..."
            return ValidationResult(
                valid=False,
                reason="Edited TOC labels do not match the linked chapter titles",
                suggestion=(
                    "Copy the exact href/title pair from the linked HTML file, for example: "
                    f"{preview}"
                ),
                severity="error",
            )

        return ValidationResult(valid=True)

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
