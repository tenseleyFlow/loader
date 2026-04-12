"""Definition-of-done state and persistence for runtime turns."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..llm.base import ToolCall
from ..tools.shell_tools import BashTool
from .verification_observations import VerificationAttempt, verification_attempt_id

TaskSize = Literal["small", "standard", "large"]
DoDStatus = Literal["draft", "in_progress", "verifying", "fixing", "done", "failed"]
VerificationConfidence = Literal["high", "medium", "low"]
VerificationKind = Literal["test", "typecheck", "lint", "build", "smoke", "runtime", "manual"]


@dataclass
class VerificationEvidence:
    """Concrete evidence captured during runtime verification."""

    command: str
    passed: bool
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
) -> list[str]:
    """Generate verification commands from execution history and project shape."""

    commands: list[str] = []

    explicit = [cmd for cmd in dod.successful_commands if _is_verification_command(cmd)]
    for command in explicit:
        _append_unique(commands, command)

    task_lower = task_statement.lower()
    if not commands and any(keyword in task_lower for keyword in ("run", "verify", "test")):
        for path_str in dod.touched_files:
            path = Path(path_str)
            if path.suffix == ".py":
                _append_unique(commands, f"python {shlex.quote(path.name)}")

    if commands:
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
        status = "PASS" if item.passed else "FAIL"
        detail = _first_non_empty_line(item.stdout) or _first_non_empty_line(item.stderr)
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
    ) -> DefinitionOfDone:
        """Load an unfinished DoD for the same task, or create a new one."""

        existing = self.load_latest(task_statement)
        if existing is not None and existing.status not in {"done", "failed"}:
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
    return str(Path(text).expanduser().resolve())


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


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


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return ""
