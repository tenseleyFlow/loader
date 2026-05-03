"""Definition-of-done state and persistence for runtime turns."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..llm.base import Message, ToolCall
from ..tools.shell_tools import BashTool
from .verification_observations import VerificationAttempt, verification_attempt_id

TaskSize = Literal["small", "standard", "large"]
DoDStatus = Literal["draft", "in_progress", "verifying", "fixing", "done", "failed"]
VerificationConfidence = Literal["high", "medium", "low"]
VerificationKind = Literal["test", "typecheck", "lint", "build", "smoke", "runtime", "manual"]

_DIRECTORY_CONTENT_HINTS = (
    "file",
    "files",
    "chapter",
    "chapters",
    "page",
    "pages",
    "test",
    "tests",
    "artifact",
    "artifacts",
    "document",
    "documents",
    "content",
    "entry",
    "entries",
)
_DIRECTORY_MUTATION_HINTS = (
    "create",
    "creating",
    "generate",
    "generating",
    "write",
    "writing",
    "add",
    "adding",
    "build",
    "building",
    "populate",
    "populating",
)
_READ_ONLY_FILE_CHANGE_HINTS = (
    "read ",
    "reading ",
    "examine ",
    "examining ",
    "inspect ",
    "inspecting ",
    "analyze ",
    "analyzing ",
    "analyse ",
    "analysing ",
    "compare ",
    "comparing ",
    "review ",
    "reviewing ",
    "study ",
    "studying ",
    "look at ",
    "looking at ",
)
_MUTATING_FILE_CHANGE_HINTS = (
    "create",
    "creating",
    "write",
    "writing",
    "update",
    "updating",
    "edit",
    "editing",
    "patch",
    "patching",
    "fix",
    "fixing",
    "modify",
    "modifying",
    "add",
    "adding",
    "generate",
    "generating",
    "build",
    "building",
    "populate",
    "populating",
    "develop",
    "developing",
)
_MINIMUM_SUBSTANTIVE_HTML_GUIDE_PAGES = 4


@dataclass
class VerificationEvidence:
    """Concrete evidence captured during runtime verification."""

    command: str
    passed: bool
    skipped: bool = False
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    output: str = ""
    kind: VerificationKind = "runtime"


@dataclass
class DefinitionOfDone:
    """Single source of truth for task completion state."""

    task_statement: str
    acceptance_criteria: list[str]
    verification_commands: list[str]
    pending_items: list[str]
    completed_items: list[str]
    evidence: list[VerificationEvidence]
    confidence: VerificationConfidence
    status: DoDStatus
    task_size: TaskSize = "small"
    retry_budget: int = 3
    retry_count: int = 0
    touched_files: list[str] = field(default_factory=list)
    successful_commands: list[str] = field(default_factory=list)
    mutating_actions: list[str] = field(default_factory=list)
    line_changes: int = 0
    storage_path: str | None = None
    last_verification_result: str | None = None
    last_verification_signature: str | None = None
    verification_attempt_counter: int = 0
    active_verification_attempt_id: str | None = None
    active_verification_attempt_number: int | None = None
    current_mode: str = "execute"
    mode_history: list[str] = field(default_factory=list)
    clarify_brief: str | None = None
    implementation_plan: str | None = None
    verification_plan: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DoD state for persistence."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DefinitionOfDone:
        """Restore DoD state from persisted JSON."""

        return cls(
            task_statement=data["task_statement"],
            acceptance_criteria=list(data.get("acceptance_criteria", [])),
            verification_commands=list(data.get("verification_commands", [])),
            pending_items=list(data.get("pending_items", [])),
            completed_items=list(data.get("completed_items", [])),
            evidence=[
                VerificationEvidence(**item) for item in data.get("evidence", [])
            ],
            confidence=data.get("confidence", "medium"),
            status=data.get("status", "draft"),
            task_size=data.get("task_size", "small"),
            retry_budget=data.get("retry_budget", 3),
            retry_count=data.get("retry_count", 0),
            touched_files=list(data.get("touched_files", [])),
            successful_commands=list(data.get("successful_commands", [])),
            mutating_actions=list(data.get("mutating_actions", [])),
            line_changes=int(data.get("line_changes", 0)),
            storage_path=data.get("storage_path"),
            last_verification_result=data.get("last_verification_result"),
            last_verification_signature=data.get("last_verification_signature"),
            verification_attempt_counter=int(data.get("verification_attempt_counter", 0)),
            active_verification_attempt_id=data.get("active_verification_attempt_id"),
            active_verification_attempt_number=(
                int(data["active_verification_attempt_number"])
                if data.get("active_verification_attempt_number") is not None
                else None
            ),
            current_mode=data.get("current_mode", "execute"),
            mode_history=list(data.get("mode_history", [])),
            clarify_brief=data.get("clarify_brief"),
            implementation_plan=data.get("implementation_plan"),
            verification_plan=data.get("verification_plan"),
        )


def create_definition_of_done(
    task_statement: str,
    *,
    retry_budget: int = 3,
) -> DefinitionOfDone:
    """Create an initial DoD object for a non-conversational task."""

    acceptance_criteria = [task_statement]
    task_lower = task_statement.lower()
    if any(keyword in task_lower for keyword in ("run", "test", "verify", "make sure")):
        acceptance_criteria.append("Demonstrate the result with runtime verification evidence.")

    return DefinitionOfDone(
        task_statement=task_statement,
        acceptance_criteria=acceptance_criteria,
        verification_commands=[],
        pending_items=["Complete the requested work"],
        completed_items=[],
        evidence=[],
        confidence="medium",
        status="draft",
        retry_budget=retry_budget,
    )


def determine_task_size(file_count: int, line_changes: int) -> TaskSize:
    """Classify task size using Loader's Sprint 02 thresholds."""

    if file_count <= 3 and line_changes < 100:
        return "small"
    if file_count <= 15 and line_changes < 500:
        return "standard"
    return "large"


def is_state_mutating_tool_call(tool_call: ToolCall) -> bool:
    """Return whether a tool call likely mutates user-visible state."""

    if tool_call.name in {"write", "edit", "patch"}:
        return True
    if tool_call.name != "bash":
        return False

    command = str(tool_call.arguments.get("command", ""))
    if not command.strip():
        return False
    return not BashTool()._is_safe_command(command)


def record_successful_tool_call(
    dod: DefinitionOfDone,
    tool_call: ToolCall,
) -> None:
    """Update DoD state with successful tool execution facts."""

    if is_state_mutating_tool_call(tool_call):
        _append_unique(dod.mutating_actions, tool_call.name)

    if tool_call.name == "write":
        file_path = _resolve_touched_path(tool_call.arguments.get("file_path", ""))
        content = str(tool_call.arguments.get("content", ""))
        if file_path:
            _append_unique(dod.touched_files, file_path)
        dod.line_changes += _count_lines(content)
    elif tool_call.name == "edit":
        file_path = _resolve_touched_path(tool_call.arguments.get("file_path", ""))
        old_string = str(tool_call.arguments.get("old_string", ""))
        new_string = str(tool_call.arguments.get("new_string", ""))
        if file_path:
            _append_unique(dod.touched_files, file_path)
        dod.line_changes += max(_count_lines(old_string), _count_lines(new_string))
    elif tool_call.name == "patch":
        file_path = _resolve_touched_path(tool_call.arguments.get("file_path", ""))
        if file_path:
            _append_unique(dod.touched_files, file_path)
        for hunk in tool_call.arguments.get("hunks", []):
            if not isinstance(hunk, dict):
                continue
            old_lines = int(hunk.get("old_lines", 0))
            new_lines = int(hunk.get("new_lines", 0))
            dod.line_changes += max(old_lines, new_lines)
    elif tool_call.name == "bash":
        command = str(tool_call.arguments.get("command", "")).strip()
        if command:
            _append_unique(dod.successful_commands, command)
            for touched_file in _extract_files_from_bash(command):
                _append_unique(dod.touched_files, touched_file)

    dod.task_size = determine_task_size(len(dod.touched_files), dod.line_changes)
    dod.status = "in_progress"


def derive_verification_commands(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    task_statement: str,
    supplement_existing: bool = False,
) -> list[str]:
    """Generate verification commands from execution history and project shape."""

    commands: list[str] = []
    html_link_command = _derive_local_html_link_verification_command(
        dod,
        project_root=project_root,
    )
    planned_artifact_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
    )

    explicit = [cmd for cmd in dod.successful_commands if _is_verification_command(cmd)]
    for command in explicit:
        _append_unique(commands, command)

    task_lower = task_statement.lower()
    if not commands and any(keyword in task_lower for keyword in ("run", "verify", "test")):
        for path_str in dod.touched_files:
            path = Path(path_str)
            if path.suffix == ".py":
                _append_unique(commands, f"python {shlex.quote(path.name)}")

    if html_link_command:
        _append_unique(commands, html_link_command)
    html_quality_command = _derive_multi_page_html_quality_command(
        dod,
        project_root=project_root,
        task_statement=task_statement,
    )
    if html_quality_command:
        _append_unique(commands, html_quality_command)
    for command in _build_planned_artifact_verification_commands(planned_artifact_targets):
        _append_unique(commands, command)

    if commands:
        return commands
    if supplement_existing:
        return commands

    if dod.task_size == "small":
        for path_str in dod.touched_files[:3]:
            path = Path(path_str)
            effective_path = path if path.is_absolute() else (project_root / path)
            _append_unique(commands, f"test -f {shlex.quote(str(effective_path))}")
            if path.suffix == ".py":
                _append_unique(
                    commands,
                    f"python -m py_compile {shlex.quote(str(effective_path))}",
                )
    elif _uses_external_artifacts_only(dod, project_root=project_root):
        for path_str in dod.touched_files[:3]:
            path = Path(path_str)
            effective_path = path if path.is_absolute() else (project_root / path)
            _append_unique(commands, f"test -f {shlex.quote(str(effective_path))}")
    else:
        if (project_root / "pyproject.toml").exists():
            _append_unique(commands, "uv run pytest -q")
        if (project_root / "package.json").exists():
            _append_unique(commands, "npm test --if-present")
            if dod.task_size != "small":
                _append_unique(commands, "npm run lint --if-present")
                _append_unique(commands, "npm run build --if-present")
        if (project_root / "Cargo.toml").exists():
            _append_unique(commands, "cargo test")
            if dod.task_size == "large":
                _append_unique(commands, "cargo check")

    return commands


def build_verification_summary(evidence: list[VerificationEvidence]) -> str:
    """Create a short evidence summary for the final user response."""

    if not evidence:
        return "Verification: skipped (no evidence required)."

    lines = ["Verification:"]
    for item in evidence:
        status = "SKIP" if item.skipped else "PASS" if item.passed else "FAIL"
        detail = _summarize_verification_detail(item)
        if detail:
            lines.append(f"- `{item.command}`: {status} ({detail})")
        else:
            lines.append(f"- `{item.command}`: {status}")
    return "\n".join(lines)


def ensure_active_verification_attempt(dod: DefinitionOfDone) -> VerificationAttempt:
    """Return the current verification attempt, synthesizing one if needed."""

    if (
        dod.active_verification_attempt_id
        and dod.active_verification_attempt_number is not None
    ):
        return VerificationAttempt(
            attempt_id=dod.active_verification_attempt_id,
            attempt_number=dod.active_verification_attempt_number,
        )

    next_number = max(int(dod.verification_attempt_counter or 0), 1)
    dod.verification_attempt_counter = next_number
    dod.active_verification_attempt_number = next_number
    dod.active_verification_attempt_id = verification_attempt_id(next_number)
    return VerificationAttempt(
        attempt_id=dod.active_verification_attempt_id,
        attempt_number=next_number,
    )


def begin_new_verification_attempt(
    dod: DefinitionOfDone,
    *,
    supersedes_attempt_id: str | None = None,
) -> VerificationAttempt:
    """Start the next verification attempt and mark it as active."""

    next_number = max(int(dod.verification_attempt_counter or 0), 0) + 1
    dod.verification_attempt_counter = next_number
    dod.active_verification_attempt_number = next_number
    dod.active_verification_attempt_id = verification_attempt_id(next_number)
    return VerificationAttempt(
        attempt_id=dod.active_verification_attempt_id,
        attempt_number=next_number,
        supersedes_attempt_id=supersedes_attempt_id,
    )


class DefinitionOfDoneStore:
    """Persist DoD state to `.loader/dod/`."""

    def __init__(self, project_root: Path) -> None:
        self.root = project_root / ".loader" / "dod"

    def create_or_resume(
        self,
        task_statement: str,
        *,
        retry_budget: int = 3,
        resume_path: Path | str | None = None,
    ) -> DefinitionOfDone:
        """Resume the active DoD for this session, or create a new one."""

        if resume_path is not None:
            path = Path(resume_path)
            if path.exists():
                existing = self.load(path)
                if (
                    existing.task_statement == task_statement
                    and existing.status not in {"done", "failed"}
                ):
                    return existing

        dod = create_definition_of_done(task_statement, retry_budget=retry_budget)
        slug = slugify(task_statement)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        dod.storage_path = str(self.root / f"{timestamp}-{slug}.json")
        self.save(dod)
        return dod

    def save(self, dod: DefinitionOfDone) -> Path:
        """Write the DoD state to disk."""

        if dod.storage_path is None:
            slug = slugify(dod.task_statement)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            dod.storage_path = str(self.root / f"{timestamp}-{slug}.json")

        path = Path(dod.storage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dod.to_dict(), indent=2, sort_keys=True))
        return path

    def load(self, path: Path) -> DefinitionOfDone:
        """Load one DoD JSON file."""

        return DefinitionOfDone.from_dict(json.loads(path.read_text()))

    def load_latest(self, task_statement: str) -> DefinitionOfDone | None:
        """Load the most recent persisted DoD for the same task statement."""

        if not self.root.exists():
            return None

        candidates = sorted(self.root.glob(f"*-{slugify(task_statement)}.json"))
        for path in reversed(candidates):
            dod = self.load(path)
            if dod.task_statement == task_statement:
                return dod
        return None


def slugify(task_statement: str) -> str:
    """Convert a task statement into a stable filename slug."""

    text = task_statement.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:64] or "task"


def _resolve_touched_path(raw: object) -> str:
    """Expand ~ and resolve a file path from tool arguments for DoD tracking."""
    text = str(raw).strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path.cwd() / path).absolute())


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _uses_external_artifacts_only(dod: DefinitionOfDone, *, project_root: Path) -> bool:
    touched = [Path(path) for path in dod.touched_files if str(path).strip()]
    if not touched:
        return False
    try:
        root = project_root.resolve()
    except FileNotFoundError:
        root = project_root
    external = [path for path in touched if not _path_is_within_root(path, root)]
    return bool(external) and len(external) == len(touched)


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def synthesize_todo_items(dod: DefinitionOfDone) -> list[dict[str, str]]:
    """Build a todo item list from the current DoD state.

    Combines abstract DoD items (pending/completed) with concrete file
    operations (touched_files) so the TUI shows live progress as each
    tool call completes — not just a batch update at the end.
    """
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    # Concrete file operations — these update in real time
    for path in dod.touched_files:
        short = Path(path).name if "/" in path else path
        label = f"Write {short}"
        if label in seen:
            continue
        seen.add(label)
        items.append({"content": label, "status": "completed", "active_form": label})

    # Abstract DoD items
    for label in dod.completed_items:
        if label in seen:
            continue
        seen.add(label)
        items.append({"content": label, "status": "completed", "active_form": label})
    for label in dod.pending_items:
        if label in seen:
            continue
        seen.add(label)
        status = "completed" if dod.status == "done" else "in_progress"
        items.append({"content": label, "status": status, "active_form": label})

    return items


def _count_lines(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + 1


def _is_verification_command(command: str) -> bool:
    command_lower = command.lower()
    signals = (
        "pytest",
        "unittest",
        "cargo test",
        "cargo check",
        "npm test",
        "npm run lint",
        "npm run build",
        "ruff check",
        "mypy",
        "python ",
        "node ",
        "uv run pytest",
    )
    return any(signal in command_lower for signal in signals)


def _extract_files_from_bash(command: str) -> list[str]:
    """Extract obvious target files from simple verification-friendly commands."""

    try:
        parts = shlex.split(command)
    except ValueError:
        return []

    if len(parts) >= 2 and parts[0] == "touch":
        return parts[1:]
    if len(parts) >= 3 and parts[0] in {"chmod", "chown"}:
        return [parts[-1]]
    if len(parts) >= 2 and parts[0] == "python":
        return [parts[1]]
    return []


def _derive_local_html_link_verification_command(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> str | None:
    html_paths: list[Path] = []
    for path_str in dod.touched_files:
        path = Path(path_str)
        effective_path = path if path.is_absolute() else (project_root / path)
        if effective_path.suffix.lower() != ".html" or not effective_path.exists():
            continue
        html_paths.append(effective_path)

    unique_paths = list(dict.fromkeys(str(path) for path in html_paths))
    resolved_paths = [Path(path) for path in unique_paths]
    if not resolved_paths:
        return None
    if not any(_html_file_contains_local_links(path) for path in resolved_paths):
        return None
    return _build_local_html_link_verification_command(resolved_paths)


def _derive_multi_page_html_quality_command(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    task_statement: str,
) -> str | None:
    if not _task_requires_substantive_html_guide_quality(task_statement):
        return None
    html_paths = _multi_page_html_quality_paths(dod, project_root=project_root)
    requires_multiple_pages = _requires_multiple_html_pages(
        dod,
        project_root=project_root,
    )
    if requires_multiple_pages and len(html_paths) < _MINIMUM_SUBSTANTIVE_HTML_GUIDE_PAGES:
        return "\n".join(
            [
                "python3 - <<'PY'",
                f"minimum_pages = {_MINIMUM_SUBSTANTIVE_HTML_GUIDE_PAGES}",
                f"found_pages = {len(html_paths)}",
                "print('HTML guide content quality issues:')",
                "print(",
                "    f'insufficient HTML page count ({found_pages} files, expected at least {minimum_pages})'",
                ")",
                "raise SystemExit(1)",
                "PY",
            ]
        )
    if len(html_paths) < _MINIMUM_SUBSTANTIVE_HTML_GUIDE_PAGES:
        return None

    path_literals = ", ".join(repr(str(path)) for path in html_paths)
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import re",
            "",
            f"paths = [{path_literals}]",
            "tag_pattern = re.compile(r'<[^>]+>')",
            "content_block_pattern = re.compile(r'<(p|li|pre|code|section|article|table|h2|h3|h4)\\b', re.IGNORECASE)",
            "issues = []",
            "checked = 0",
            "for raw_path in paths:",
            "    path = Path(raw_path)",
            "    if not path.exists():",
            "        continue",
            "    checked += 1",
            "    text = path.read_text()",
            "    plain = tag_pattern.sub(' ', text)",
            "    plain = re.sub(r'\\s+', ' ', plain).strip()",
            "    content_blocks = len(content_block_pattern.findall(text))",
            "    has_h1 = bool(re.search(r'<h1\\b', text, re.IGNORECASE))",
            "    minimum_chars = 180 if path.name.lower() == 'index.html' else 220",
            "    minimum_blocks = 2 if path.name.lower() == 'index.html' else 3",
            "    if not has_h1:",
            "        issues.append(f'{path}: missing <h1>')",
            "    if len(plain) < minimum_chars:",
            "        issues.append(",
            "            f'{path}: thin content ({len(plain)} text chars, expected at least {minimum_chars})'",
            "        )",
            "    if content_blocks < minimum_blocks:",
            "        issues.append(",
            "            f'{path}: insufficient structured content ({content_blocks} blocks, expected at least {minimum_blocks})'",
            "        )",
            "if issues:",
            "    print('HTML guide content quality issues:')",
            "    print('\\n'.join(issues))",
            "    raise SystemExit(1)",
            "print(f'Checked HTML guide content quality across {checked} file(s).')",
            "PY",
        ]
    )


def collect_planned_artifact_targets(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    max_paths: int | None = None,
) -> list[tuple[Path, bool]]:
    if not dod.implementation_plan:
        return []

    plan_path = Path(dod.implementation_plan)
    if not plan_path.exists():
        return []

    markdown = plan_path.read_text()
    file_change_lines = _extract_markdown_section_lines(markdown, "File Changes")
    candidates = _extract_file_change_path_literals(file_change_lines)
    if not candidates:
        candidates = _extract_planned_path_literals(file_change_lines or markdown.splitlines())
    if not candidates:
        confirmed_progress_lines = _extract_markdown_section_lines(
            markdown,
            "Confirmed Progress",
        )
        candidates = _extract_planned_path_literals(confirmed_progress_lines)
    targets: list[tuple[Path, bool]] = []
    seen: set[tuple[str, bool]] = set()

    selected_candidates = candidates if max_paths is None else candidates[:max_paths]
    for raw_path in selected_candidates:
        effective_path = _resolve_planned_artifact_path(raw_path, project_root=project_root)
        if effective_path is None:
            continue
        expect_directory = raw_path.endswith("/")
        if not expect_directory and not effective_path.suffix:
            continue
        key = (str(effective_path), expect_directory)
        if key in seen:
            continue
        seen.add(key)
        targets.append((effective_path, expect_directory))
    return targets


def all_planned_artifacts_exist(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    max_paths: int | None = None,
) -> bool:
    if not all_planned_artifact_outputs_exist(
        dod,
        project_root=project_root,
        max_paths=max_paths,
    ):
        return False
    targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=max_paths,
    )
    return not _planned_html_outputs_have_missing_local_links(
        dod,
        project_root=project_root,
        targets=targets,
    )


def all_planned_artifact_outputs_exist(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    max_paths: int | None = None,
) -> bool:
    targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=max_paths,
    )
    if not targets:
        return False
    if not all(
        planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=expect_directory,
            project_root=project_root,
        )
        for target, expect_directory in targets
    ):
        return False
    if _planned_html_outputs_declare_missing_files(
        dod,
        project_root=project_root,
        targets=targets,
    ):
        return False
    if _substantive_multi_page_html_guide_is_incomplete(
        dod,
        project_root=project_root,
    ):
        return False
    return True


def planned_artifact_target_satisfied(
    dod: DefinitionOfDone,
    *,
    target: Path,
    expect_directory: bool,
    project_root: Path,
) -> bool:
    """Return whether one planned file or directory target is substantively satisfied."""

    if not expect_directory:
        return target.is_file()
    if not target.is_dir():
        return False
    if not planned_directory_requires_generated_files(
        dod,
        target=target,
        project_root=project_root,
    ):
        return True
    return _directory_contains_files(target)


def infer_next_declared_html_output_file(
    *,
    target: Path,
    project_root: Path,
) -> Path | None:
    """Return the first missing HTML file already declared within an output directory."""

    missing_targets = collect_missing_declared_html_output_files(
        target=target,
        project_root=project_root,
    )
    return missing_targets[0] if missing_targets else None


def infer_next_output_file(
    *,
    target: Path,
    project_root: Path,
    messages: list[Message] | None = None,
) -> tuple[Path | None, str | None]:
    """Infer the next concrete output file for a planned output directory.

    Returns a tuple of `(path, source)` where source is one of:
    - `"declared"` when inferred from the current artifact graph
    - `"observed"` when mirrored from an already-inspected sibling directory
    """

    declared_target = infer_next_declared_html_output_file(
        target=target,
        project_root=project_root,
    )
    if declared_target is not None:
        return declared_target, "declared"

    observed_target = _infer_next_observed_output_file(
        target=target,
        messages=messages or [],
    )
    if observed_target is not None:
        return observed_target, "observed"
    return None, None


def collect_missing_declared_html_output_files(
    *,
    target: Path,
    project_root: Path,
) -> tuple[Path, ...]:
    """Return missing HTML outputs already declared within the current artifact graph."""

    normalized_target = target.resolve(strict=False)
    scope_target = normalized_target
    if normalized_target.suffix.lower() in {".html", ".htm"}:
        scope_target = normalized_target.parent
    artifact_root = _resolve_declared_html_artifact_root(
        scope_target,
        project_root=project_root.resolve(strict=False),
    )
    if artifact_root is None:
        return ()

    html_files = [path for path in sorted(artifact_root.rglob("*.html")) if path.is_file()]
    if not html_files:
        return ()

    missing_targets: list[Path] = []
    seen: set[str] = set()
    for html_file in html_files:
        try:
            content = html_file.read_text()
        except OSError:
            continue
        for resolved_target in _iter_local_html_targets(html_file, content):
            if resolved_target.exists():
                continue
            if resolved_target.suffix.lower() not in {".html", ".htm"}:
                continue
            try:
                resolved_target.relative_to(artifact_root)
                resolved_target.relative_to(scope_target)
            except ValueError:
                continue
            key = str(resolved_target)
            if key in seen:
                continue
            seen.add(key)
            missing_targets.append(resolved_target)
    return tuple(missing_targets)


def _planned_html_outputs_declare_missing_files(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    targets: list[tuple[Path, bool]],
) -> bool:
    if not _requires_multiple_html_pages(dod, project_root=project_root):
        return False

    seen_scopes: set[str] = set()
    for target, expect_directory in targets:
        if expect_directory:
            scope_target = target
        elif target.suffix.lower() in {".html", ".htm"}:
            scope_target = target
        else:
            continue

        scope_key = str(scope_target.resolve(strict=False))
        if scope_key in seen_scopes:
            continue
        seen_scopes.add(scope_key)
        if collect_missing_declared_html_output_files(
            target=scope_target,
            project_root=project_root,
        ):
            return True
    return False


def _infer_next_observed_output_file(
    *,
    target: Path,
    messages: list[Message],
) -> Path | None:
    normalized_target = target.resolve(strict=False)
    if normalized_target.suffix:
        return None

    existing_names = {
        path.name
        for path in normalized_target.glob("*.html")
        if path.is_file()
    }
    candidate_names: set[str] = set()
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            if tool_call.name != "read":
                continue
            raw_path = str(tool_call.arguments.get("file_path", "")).strip()
            if not raw_path:
                continue
            observed_path = Path(raw_path).expanduser().resolve(strict=False)
            if observed_path.suffix.lower() not in {".html", ".htm"}:
                continue
            if observed_path.name.lower() == "index.html":
                continue
            if observed_path.parent.name != normalized_target.name:
                continue
            try:
                observed_path.relative_to(normalized_target)
                continue
            except ValueError:
                pass
            if observed_path.name in existing_names:
                continue
            candidate_names.add(observed_path.name)

    if not candidate_names:
        return None
    return normalized_target / sorted(candidate_names)[0]


def _build_planned_artifact_verification_commands(
    targets: list[tuple[Path, bool]],
) -> list[str]:
    commands: list[str] = []
    for effective_path, expect_directory in targets:
        command = (
            f"test -d {shlex.quote(str(effective_path))}"
            if expect_directory
            else f"test -f {shlex.quote(str(effective_path))}"
        )
        _append_unique(commands, command)
    return commands


def _multi_page_html_quality_paths(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> list[Path]:
    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=24,
    )
    paths: list[Path] = []
    seen: set[str] = set()

    for target, expect_directory in planned_targets:
        if expect_directory:
            if not target.exists():
                continue
            try:
                discovered = sorted(path for path in target.rglob("*.html") if path.is_file())
            except OSError:
                continue
            for path in discovered:
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
            continue
        if target.suffix.lower() not in {".html", ".htm"} or not target.exists():
            continue
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        paths.append(target)

    if paths:
        return paths

    touched_html = []
    for path_str in dod.touched_files:
        path = Path(path_str)
        effective_path = path if path.is_absolute() else (project_root / path)
        if effective_path.suffix.lower() in {".html", ".htm"}:
            touched_html.append(effective_path)
    return list(dict.fromkeys(touched_html))


def _requires_multiple_html_pages(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> bool:
    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=24,
    )
    if any(
        expect_directory
        and planned_directory_requires_generated_files(
            dod,
            target=target,
            project_root=project_root,
        )
        for target, expect_directory in planned_targets
    ):
        return True

    planned_html = [
        target
        for target, expect_directory in planned_targets
        if not expect_directory and target.suffix.lower() in {".html", ".htm"}
    ]
    if len(planned_html) > 1:
        return True

    lowered = dod.task_statement.lower()
    return "chapter" in lowered or "chapters" in lowered


def _substantive_multi_page_html_guide_is_incomplete(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> bool:
    if not _task_requires_substantive_html_guide_quality(dod.task_statement):
        return False
    if not _requires_multiple_html_pages(dod, project_root=project_root):
        return False
    html_paths = _multi_page_html_quality_paths(dod, project_root=project_root)
    return len(html_paths) < _MINIMUM_SUBSTANTIVE_HTML_GUIDE_PAGES


def _task_requires_substantive_html_guide_quality(task_statement: str) -> bool:
    lowered = task_statement.lower()
    if not any(token in lowered for token in ("guide", "tutorial", "documentation", "docs")):
        return False
    return any(
        token in lowered
        for token in (
            "thorough",
            "comprehensive",
            "detailed",
            "equally",
            "cadence",
            "depth",
            "same structure",
            "same style",
        )
    )


def _extract_markdown_section_lines(markdown: str, heading: str) -> list[str]:
    current_heading: str | None = None
    collected: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_heading = stripped[3:].strip().lower()
            continue
        if current_heading == heading.lower():
            collected.append(line)
    return collected


def _extract_planned_path_literals(lines: list[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()

    for line in lines:
        candidates = re.findall(r"`([^`]+)`", line)
        if not candidates:
            stripped = line.strip()
            stripped = re.sub(r"^[-*+]\s+", "", stripped)
            stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
            stripped = stripped.strip("`'\",.:;()[]{}")
            candidates = [stripped] if _looks_like_path_literal(stripped) else []
        for candidate in candidates:
            normalized = candidate.strip("`'\",.:;()[]{}")
            if not _looks_like_path_literal(normalized) or normalized in seen:
                continue
            seen.add(normalized)
            paths.append(normalized)
    return paths


def _extract_file_change_path_literals(lines: list[str]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    directory_stack: list[tuple[int, str]] = []
    read_only_stack: list[int] = []

    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        while read_only_stack and indent <= read_only_stack[-1]:
            read_only_stack.pop()
        while directory_stack and indent <= directory_stack[-1][0]:
            directory_stack.pop()
        if _line_describes_read_only_file_change(line):
            read_only_stack.append(indent)
            continue
        if read_only_stack:
            continue

        backticked = re.findall(r"`([^`]+)`", line)
        if backticked:
            candidates = backticked
        else:
            stripped = line.strip()
            stripped = re.sub(r"^[-*+]\s+", "", stripped)
            stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
            stripped = stripped.strip("`'\",.:;()[]{}")
            candidates = [stripped] if _looks_like_path_literal(stripped) else []

        for candidate in candidates:
            normalized = candidate.strip("`'\",.:;()[]{}")
            if not _looks_like_file_change_literal(normalized):
                continue
            contextual = _apply_directory_context_to_file_change(
                normalized,
                directory_stack[-1][1] if directory_stack else None,
            )
            if contextual in seen:
                continue
            seen.add(contextual)
            paths.append(contextual)
            if contextual.endswith("/"):
                directory_stack.append((indent, contextual))
    return paths


def _line_describes_read_only_file_change(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return False
    if any(hint in lowered for hint in _MUTATING_FILE_CHANGE_HINTS):
        return False
    return any(hint in lowered for hint in _READ_ONLY_FILE_CHANGE_HINTS)


def _looks_like_file_change_literal(value: str) -> bool:
    return _looks_like_path_literal(value) or bool(Path(value).suffix)


def _apply_directory_context_to_file_change(
    value: str,
    directory_context: str | None,
) -> str:
    if not directory_context:
        return value
    if value.startswith(("~/", "./", "../", "/")) or "/" in value:
        return value
    return directory_context.rstrip("/") + "/" + value


def _resolve_declared_html_artifact_root(
    target: Path,
    *,
    project_root: Path,
) -> Path | None:
    for candidate in [target, *target.parents]:
        if (candidate / "index.html").is_file():
            return candidate
        if candidate == project_root or candidate == candidate.parent:
            break

    fallback = target if target.exists() else target.parent
    if fallback.exists():
        return fallback
    return None


def _iter_local_html_targets(file_path: Path, content: str) -> list[Path]:
    pattern = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
    targets: list[Path] = []
    seen: set[str] = set()
    for href in pattern.findall(content):
        candidate = href.strip()
        if not _is_local_html_link_target(candidate):
            continue
        resolved = (file_path.parent / candidate).resolve(strict=False)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        targets.append(resolved)
    return targets


def _is_local_html_link_target(href: str) -> bool:
    candidate = href.strip()
    if not candidate or candidate.startswith(("#", "http://", "https://", "mailto:")):
        return False
    if "?" in candidate:
        candidate = candidate.split("?", 1)[0]
    if "#" in candidate:
        candidate = candidate.split("#", 1)[0]
    return Path(candidate).suffix.lower() in {".html", ".htm"}


def _looks_like_path_literal(value: str) -> bool:
    if not value or " " in value:
        return False
    if value.startswith(("http://", "https://")):
        return False
    return (
        value.startswith(("~/", "./", "../", "/"))
        or "/" in value
        or value.endswith("/")
    )


def _resolve_planned_artifact_path(
    raw_path: str,
    *,
    project_root: Path,
) -> Path | None:
    text = raw_path.strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def planned_directory_requires_generated_files(
    dod: DefinitionOfDone,
    *,
    target: Path,
    project_root: Path,
) -> bool:
    """Return whether a planned directory is expected to contain generated files."""

    plan_path = Path(dod.implementation_plan) if dod.implementation_plan else None
    if plan_path is not None and plan_path.exists():
        markdown = plan_path.read_text()
        file_change_lines = _extract_markdown_section_lines(markdown, "File Changes")
        if any(
            _line_describes_directory_contents(line, target=target, project_root=project_root)
            for line in file_change_lines
        ):
            return True

        execution_lines = _extract_markdown_section_lines(markdown, "Execution Order")
        if any(
            _line_mentions_directory_generation(line, target=target)
            for line in execution_lines
        ):
            return True

    todo_lines = [*dod.pending_items, *dod.completed_items]
    return any(
        _line_mentions_directory_generation(line, target=target)
        for line in todo_lines
    )


def _line_describes_directory_contents(
    line: str,
    *,
    target: Path,
    project_root: Path,
) -> bool:
    lowered = line.lower()
    if not any(hint in lowered for hint in _DIRECTORY_CONTENT_HINTS):
        return False

    target_text = str(target)
    relative_target = str(target.relative_to(project_root)) if target.is_relative_to(project_root) else ""
    if target_text in line or relative_target and relative_target in line:
        return True
    return _line_mentions_directory_generation(line, target=target)


def _line_mentions_directory_generation(line: str, *, target: Path) -> bool:
    lowered = line.lower()
    if not any(hint in lowered for hint in _DIRECTORY_CONTENT_HINTS):
        return False
    if not any(hint in lowered for hint in _DIRECTORY_MUTATION_HINTS) and "directory for" not in lowered:
        return False
    directory_tokens = _directory_tokens(target)
    return any(token in lowered for token in directory_tokens)


def _directory_tokens(target: Path) -> set[str]:
    tokens: set[str] = set()
    for raw_token in re.split(r"[^a-z0-9]+", target.name.lower()):
        token = raw_token.strip()
        if len(token) < 2:
            continue
        tokens.add(token)
        if token.endswith("ies") and len(token) > 3:
            tokens.add(f"{token[:-3]}y")
        elif token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _directory_contains_files(target: Path) -> bool:
    try:
        return any(child.is_file() for child in target.rglob("*"))
    except OSError:
        return False


def _html_file_contains_local_links(path: Path) -> bool:
    pattern = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
    try:
        text = path.read_text()
    except OSError:
        return False
    return any(_is_local_html_link_target(href) for href in pattern.findall(text))


def _planned_html_outputs_have_missing_local_links(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    targets: list[tuple[Path, bool]],
) -> bool:
    html_paths: list[Path] = []
    for raw_path in dod.touched_files:
        path = Path(raw_path)
        effective_path = path if path.is_absolute() else (project_root / path)
        if effective_path.suffix.lower() != ".html" or not effective_path.exists():
            continue
        html_paths.append(effective_path)

    for target, expect_directory in targets:
        if expect_directory or target.suffix.lower() != ".html" or not target.exists():
            continue
        html_paths.append(target)

    seen: set[str] = set()
    for path in html_paths:
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        if _html_file_has_missing_local_links(path):
            return True
    return False


def _html_file_has_missing_local_links(path: Path) -> bool:
    pattern = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
    try:
        text = path.read_text()
    except OSError:
        return False
    for href in pattern.findall(text):
        target = href.strip()
        if not _is_local_html_link_target(target):
            continue
        normalized = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not normalized:
            continue
        if not (path.parent / normalized).resolve().exists():
            return True
    return False


def _is_local_html_link_target(href: str) -> bool:
    target = href.strip()
    if not target:
        return False
    if target.startswith(("#", "mailto:", "tel:", "javascript:")):
        return False
    if "://" in target:
        return False
    target = target.split("#", 1)[0].split("?", 1)[0].strip()
    return bool(target)


def _build_local_html_link_verification_command(paths: list[Path]) -> str:
    serialized_paths = ", ".join(repr(str(path)) for path in paths)
    return "\n".join(
        [
            "python3 - <<'PY'",
            "from pathlib import Path",
            "import re",
            "",
            f"paths = [{serialized_paths}]",
            (
                r"pattern = re.compile(r'href\s*=\s*[\"\\\']([^\"\\\']+)[\"\\\']', "
                "re.IGNORECASE)"
            ),
            "checked = 0",
            "missing = []",
            "for raw_path in paths:",
            "    html_path = Path(raw_path)",
            "    if not html_path.exists():",
            "        continue",
            "    text = html_path.read_text()",
            "    for href in pattern.findall(text):",
            "        target = href.strip()",
            "        if not target:",
            "            continue",
            "        if target.startswith((\"#\", \"mailto:\", \"tel:\", \"javascript:\")):",
            "            continue",
            "        if \"://\" in target:",
            "            continue",
            "        target = target.split(\"#\", 1)[0].split(\"?\", 1)[0].strip()",
            "        if not target:",
            "            continue",
            "        checked += 1",
            "        resolved = (html_path.parent / target).resolve()",
            "        if not resolved.exists():",
            "            missing.append(f\"{html_path}:{href} -> {resolved}\")",
            "if missing:",
            "    print(\"Missing local HTML links:\")",
            "    print(\"\\n\".join(missing))",
            "    raise SystemExit(1)",
            "print(f\"Checked {checked} local HTML links across {len(paths)} file(s).\")",
            "PY",
        ]
    )


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return ""


def _summarize_verification_detail(item: VerificationEvidence) -> str:
    for candidate in (item.stdout, item.stderr, item.output):
        lines = [line.strip() for line in str(candidate).splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) == 1:
            return lines[0][:240]

        head = lines[0][:120]
        tail = [line[:120] for line in lines[1:3]]
        if head.endswith(":") and tail:
            detail = f"{head} {'; '.join(tail)}"
        else:
            detail = "; ".join([head, *tail[:1]])
        if len(lines) > len(tail) + 1:
            detail += "; ..."
        return detail[:240]
    return ""
