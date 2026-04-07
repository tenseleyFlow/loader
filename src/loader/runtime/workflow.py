"""Workflow routing and artifact persistence for Loader runtime modes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from .dod import slugify

VERIFICATION_SEPARATOR = "<<<VERIFICATION>>>"

_SECTION_ALIASES = {
    "task statement": "task_statement",
    "desired outcome": "desired_outcome",
    "in scope": "in_scope",
    "out of scope": "non_goals",
    "out of scope non goals": "non_goals",
    "out of scope or non goals": "non_goals",
    "non goals": "non_goals",
    "non-goals": "non_goals",
    "decision boundaries": "decision_boundaries",
    "constraints": "constraints",
    "likely touchpoints": "likely_touchpoints",
    "assumptions": "assumptions",
    "acceptance criteria": "acceptance_criteria",
    "file changes": "file_changes",
    "execution order": "execution_order",
    "risks": "risks",
    "verification commands": "verification_commands",
    "commands": "verification_commands",
    "notes": "notes",
}


class WorkflowMode(StrEnum):
    """High-level runtime modes for one Loader task turn."""

    CLARIFY = "clarify"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"

    @classmethod
    def from_str(cls, value: str | None) -> WorkflowMode | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        for mode in cls:
            if mode.value == normalized:
                return mode
        raise ValueError(f"Unknown workflow mode: {value}")


@dataclass(slots=True)
class ModeDecision:
    """Router output for the entry point of a task turn."""

    mode: WorkflowMode
    reason: str
    ambiguity_score: float = 0.0
    complexity_score: float = 0.0


@dataclass(slots=True)
class ClarifyBrief:
    """Execution-ready brief created from Loader's single-question clarify flow."""

    protocol_label: ClassVar[str] = "single-question clarify brief"
    follow_up_mode: ClassVar[str] = WorkflowMode.EXECUTE.value

    task_statement: str
    desired_outcome: list[str] = field(default_factory=list)
    in_scope: list[str] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    decision_boundaries: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    likely_touchpoints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    question: str | None = None
    answer: str | None = None

    @classmethod
    def from_markdown(
        cls,
        markdown: str,
        *,
        task_statement: str,
        question: str | None = None,
        answer: str | None = None,
    ) -> ClarifyBrief:
        sections = _parse_markdown_sections(markdown)
        brief = cls(
            task_statement=_first_item(sections.get("task_statement")) or task_statement,
            desired_outcome=sections.get("desired_outcome", []),
            in_scope=sections.get("in_scope", []),
            non_goals=sections.get("non_goals", []),
            decision_boundaries=sections.get("decision_boundaries", []),
            constraints=sections.get("constraints", []),
            likely_touchpoints=sections.get("likely_touchpoints", []),
            assumptions=sections.get("assumptions", []),
            acceptance_criteria=sections.get("acceptance_criteria", []),
            question=question,
            answer=answer,
        )
        brief.fill_defaults()
        return brief

    @classmethod
    def fallback(
        cls,
        *,
        task_statement: str,
        question: str,
        answer: str,
    ) -> ClarifyBrief:
        brief = cls(
            task_statement=task_statement,
            desired_outcome=[answer or "Clarify the intended outcome before implementation."],
            in_scope=[task_statement],
            non_goals=["Anything not confirmed in the clarification answer."],
            decision_boundaries=["Escalate if the clarified scope changes materially."],
            constraints=["Honor the clarified answer and existing repository conventions."],
            likely_touchpoints=["Determine the concrete files during execution."],
            assumptions=[f"Clarification answer: {answer or 'No answer provided.'}"],
            question=question,
            answer=answer,
        )
        brief.fill_defaults()
        return brief

    def fill_defaults(self) -> None:
        if not self.desired_outcome:
            self.desired_outcome = [self.task_statement]
        if not self.in_scope:
            self.in_scope = [self.task_statement]
        if not self.non_goals:
            self.non_goals = ["Do not expand beyond the clarified task statement."]
        if not self.decision_boundaries:
            self.decision_boundaries = [
                "Escalate for destructive or preference-dependent changes.",
            ]
        if not self.constraints:
            self.constraints = ["Preserve the existing codebase conventions and tests."]
        if not self.likely_touchpoints:
            self.likely_touchpoints = ["Identify exact files during planning or execution."]
        if not self.assumptions:
            self.assumptions = ["Unspecified details stay unchanged unless evidence says otherwise."]
        if not self.acceptance_criteria:
            self.acceptance_criteria = list(
                dict.fromkeys(self.desired_outcome + self.in_scope[:2])
            )

    def to_markdown(self) -> str:
        lines = [
            "# Task Brief",
            "",
            f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ')}",
            "",
        ]
        lines.extend(_render_section("Workflow Contract", self.workflow_contract()))
        lines.extend(
            [
            "## Task Statement",
            self.task_statement,
            "",
            ]
        )
        lines.extend(_render_section("Desired Outcome", self.desired_outcome))
        lines.extend(_render_section("In Scope", self.in_scope))
        lines.extend(_render_section("Non Goals", self.non_goals))
        lines.extend(_render_section("Decision Boundaries", self.decision_boundaries))
        lines.extend(_render_section("Constraints", self.constraints))
        lines.extend(_render_section("Likely Touchpoints", self.likely_touchpoints))
        lines.extend(_render_section("Assumptions", self.assumptions))
        lines.extend(_render_section("Acceptance Criteria", self.acceptance_criteria))
        if self.question:
            lines.extend(_render_section("Clarify Question", [self.question]))
        if self.answer:
            lines.extend(_render_section("Clarify Answer", [self.answer]))
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def workflow_contract(cls) -> list[str]:
        return [
            f"Protocol: {cls.protocol_label}.",
            (
                "Ask exactly one focused question, persist one brief artifact, "
                f"then return control to `{cls.follow_up_mode}` mode."
            ),
        ]


@dataclass(slots=True)
class PlanningArtifacts:
    """Persistent planning artifacts created before execution."""

    protocol_label: ClassVar[str] = "single-pass planning artifact generation"

    implementation_markdown: str
    verification_markdown: str
    verification_commands: list[str]
    acceptance_criteria: list[str]
    implementation_steps: list[str]

    @classmethod
    def from_model_output(
        cls,
        model_output: str,
        *,
        task_statement: str,
    ) -> PlanningArtifacts:
        implementation_markdown, verification_markdown = _split_plan_output(model_output)
        implementation_sections = _parse_markdown_sections(implementation_markdown)
        verification_sections = _parse_markdown_sections(verification_markdown)

        implementation_steps = (
            implementation_sections.get("execution_order", [])
            or implementation_sections.get("file_changes", [])
        )
        if not implementation_steps:
            implementation_steps = [task_statement]

        verification_commands = _extract_commands(
            verification_sections.get("verification_commands", [])
        )
        acceptance_criteria = (
            verification_sections.get("acceptance_criteria", [])
            or implementation_sections.get("acceptance_criteria", [])
        )
        if not acceptance_criteria:
            acceptance_criteria = [task_statement]

        return cls(
            implementation_markdown=_prepend_workflow_contract(
                _ensure_heading(
                    implementation_markdown,
                    "# Implementation Plan",
                ),
                cls.workflow_contract(),
            ),
            verification_markdown=_prepend_workflow_contract(
                _ensure_heading(
                    verification_markdown,
                    "# Verification Plan",
                ),
                cls.workflow_contract(),
            ),
            verification_commands=verification_commands,
            acceptance_criteria=acceptance_criteria,
            implementation_steps=implementation_steps,
        )

    @classmethod
    def fallback(
        cls,
        *,
        task_statement: str,
    ) -> PlanningArtifacts:
        implementation_markdown = "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- Determine concrete files needed for: {task_statement}",
                "",
                "## Execution Order",
                f"1. Inspect the codebase areas relevant to: {task_statement}",
                "2. Apply the minimum required changes.",
                "3. Re-run the most relevant verification commands.",
                "",
                "## Risks",
                "- Unknown repository conventions may require one discovery pass first.",
                "",
            ]
        )
        verification_markdown = "\n".join(
            [
                "# Verification Plan",
                "",
                "## Acceptance Criteria",
                f"- {task_statement}",
                "",
                "## Verification Commands",
                "- echo \"add verification command\"",
                "",
                "## Notes",
                "- Replace the placeholder verification command with a project-specific check.",
                "",
            ]
        )
        return cls(
            implementation_markdown=_prepend_workflow_contract(
                implementation_markdown,
                cls.workflow_contract(),
            ),
            verification_markdown=_prepend_workflow_contract(
                verification_markdown,
                cls.workflow_contract(),
            ),
            verification_commands=["echo \"add verification command\""],
            acceptance_criteria=[task_statement],
            implementation_steps=[
                f"Inspect the codebase areas relevant to: {task_statement}",
                "Apply the minimum required changes.",
                "Re-run the most relevant verification commands.",
            ],
        )

    @classmethod
    def workflow_contract(cls) -> list[str]:
        return [
            f"Protocol: {cls.protocol_label}.",
            "Loader writes the implementation and verification artifacts in one pass.",
            "It does not run a planner/critic consensus loop.",
        ]


class WorkflowArtifactStore:
    """Persist briefs and plans under `.loader/`."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.loader_root = project_root / ".loader"
        self.briefs_root = self.loader_root / "briefs"
        self.plans_root = self.loader_root / "plans"

    def write_brief(self, task_statement: str, brief: ClarifyBrief) -> Path:
        path = self.briefs_root / f"{_timestamp()}-{slugify(task_statement)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(brief.to_markdown())
        return path

    def write_plan(
        self,
        task_statement: str,
        artifacts: PlanningArtifacts,
    ) -> tuple[Path, Path]:
        plan_root = self.plans_root / f"{_timestamp()}-{slugify(task_statement)}"
        plan_root.mkdir(parents=True, exist_ok=True)
        implementation_path = plan_root / "implementation.md"
        verification_path = plan_root / "verification.md"
        implementation_path.write_text(artifacts.implementation_markdown.rstrip() + "\n")
        verification_path.write_text(artifacts.verification_markdown.rstrip() + "\n")
        return implementation_path, verification_path


class ModeRouter:
    """Simple heuristic router for clarify/plan/execute entry modes."""

    clarify_threshold = 0.55
    plan_threshold = 0.45

    def route(
        self,
        task: str,
        *,
        requested_mode: WorkflowMode | None = None,
        has_brief: bool = False,
        has_plan: bool = False,
        allow_clarify: bool = True,
    ) -> ModeDecision:
        if requested_mode is not None:
            return ModeDecision(
                mode=requested_mode,
                reason=f"explicit {requested_mode.value} request",
            )

        if has_plan:
            return ModeDecision(
                mode=WorkflowMode.EXECUTE,
                reason="reusing existing plan artifacts",
            )

        ambiguity = self._ambiguity_score(task)
        complexity = self._complexity_score(task)

        if allow_clarify and not has_brief and ambiguity >= self.clarify_threshold:
            return ModeDecision(
                mode=WorkflowMode.CLARIFY,
                reason="prompt is broad or missing boundaries",
                ambiguity_score=ambiguity,
                complexity_score=complexity,
            )

        if complexity >= self.plan_threshold:
            return ModeDecision(
                mode=WorkflowMode.PLAN,
                reason="task looks complex enough to benefit from a persisted plan",
                ambiguity_score=ambiguity,
                complexity_score=complexity,
            )

        return ModeDecision(
            mode=WorkflowMode.EXECUTE,
            reason="task appears concrete enough for direct execution",
            ambiguity_score=ambiguity,
            complexity_score=complexity,
        )

    def _ambiguity_score(self, task: str) -> float:
        lowered = task.lower()
        words = re.findall(r"\w+", lowered)
        score = 0.0

        if (
            "--clarify" in lowered
            or "don't assume" in lowered
            or "do not assume" in lowered
            or "not sure" in lowered
            or "figure out" in lowered
            or "interview me" in lowered
            or "ask me" in lowered
            or lowered.startswith("clarify ")
        ):
            score += 0.65

        if any(
            phrase in lowered
            for phrase in (
                "something",
                "somehow",
                "better",
                "improve",
                "fix this",
                "make it",
                "more like",
                "feels more like",
            )
        ):
            score += 0.2

        if not _has_concrete_anchor(task):
            score += 0.2

        if len(words) <= 12 and any(
            verb in lowered
            for verb in ("build", "add", "improve", "refactor", "implement")
        ):
            score += 0.15

        return min(score, 1.0)

    def _complexity_score(self, task: str) -> float:
        lowered = task.lower()
        words = re.findall(r"\w+", lowered)
        score = 0.0

        if len(words) >= 18:
            score += 0.2
        if len(words) >= 30:
            score += 0.15

        if any(
            phrase in lowered
            for phrase in (
                "refactor",
                "architecture",
                "migrate",
                "persistent",
                "workflow",
                "deep dive",
                "report",
                "implementation plan",
                "verification plan",
            )
        ):
            score += 0.3

        if lowered.count(" and ") >= 2 or lowered.count(",") >= 2:
            score += 0.15

        if _has_concrete_anchor(task):
            score += 0.1

        return min(score, 1.0)


def load_brief(path: Path) -> ClarifyBrief:
    """Load a clarify brief from disk."""

    return ClarifyBrief.from_markdown(path.read_text(), task_statement=path.stem)


def load_planning_artifacts(
    implementation_path: Path,
    verification_path: Path,
    *,
    task_statement: str,
) -> PlanningArtifacts:
    """Load persisted planning artifacts from disk."""

    combined = (
        implementation_path.read_text().rstrip()
        + "\n\n"
        + VERIFICATION_SEPARATOR
        + "\n\n"
        + verification_path.read_text().rstrip()
    )
    return PlanningArtifacts.from_model_output(combined, task_statement=task_statement)


def sync_todos_to_definition_of_done(
    dod,
    todos: list[dict[str, str]],
) -> None:
    """Reflect todo state into DoD pending/completed items."""

    special_pending = [
        item for item in dod.pending_items if item in {"Complete the requested work", "Collect verification evidence"}
    ]
    special_completed = [
        item for item in dod.completed_items if item in {"Complete the requested work", "Collect verification evidence"}
    ]

    pending: list[str] = []
    completed: list[str] = []
    for item in todos:
        status = str(item.get("status", "")).strip().lower()
        label = str(
            item.get("active_form") if status == "in_progress" else item.get("content", "")
        ).strip()
        if not label:
            continue
        if status == "completed":
            completed.append(str(item.get("content", label)).strip())
        else:
            pending.append(label)

    dod.pending_items = list(dict.fromkeys(pending + special_pending))
    dod.completed_items = list(dict.fromkeys(completed + special_completed))


def extract_verification_commands_from_markdown(markdown: str) -> list[str]:
    """Extract verification commands from a verification-plan markdown document."""

    sections = _parse_markdown_sections(markdown)
    return _extract_commands(sections.get("verification_commands", []))


def build_execute_bridge(
    brief_path: Path | None,
    implementation_path: Path | None,
    verification_path: Path | None,
) -> str | None:
    """Build a compact execution bridge message from persisted artifacts."""

    parts: list[str] = []
    if brief_path and brief_path.exists():
        parts.append(
            "Use the clarify brief below as the requirements source of truth.\n\n"
            + brief_path.read_text().strip()
        )
    if implementation_path and implementation_path.exists():
        parts.append(
            "Use the implementation plan below to sequence the work.\n\n"
            + implementation_path.read_text().strip()
        )
    if verification_path and verification_path.exists():
        parts.append(
            "Use the verification plan below to determine done-ness.\n\n"
            + verification_path.read_text().strip()
        )
    if not parts:
        return None
    return "\n\n".join(parts)


def _split_plan_output(model_output: str) -> tuple[str, str]:
    if VERIFICATION_SEPARATOR in model_output:
        implementation, verification = model_output.split(VERIFICATION_SEPARATOR, maxsplit=1)
        return implementation.strip(), verification.strip()
    return model_output.strip(), ""


def _ensure_heading(markdown: str, heading: str) -> str:
    stripped = markdown.strip()
    if not stripped:
        return heading + "\n"
    if stripped.startswith("#"):
        return stripped + "\n"
    return f"{heading}\n\n{stripped}\n"


def _prepend_workflow_contract(markdown: str, contract_lines: list[str]) -> str:
    stripped = markdown.rstrip()
    lines = stripped.splitlines()
    if not lines:
        return stripped
    if any(line.strip().lower() == "## workflow contract" for line in lines):
        return stripped + "\n"
    if not lines[0].startswith("#"):
        return stripped + "\n"
    body = "\n".join(lines[1:]).lstrip("\n")
    contract_section = "\n".join(
        [
            lines[0],
            "",
            "## Workflow Contract",
            *[f"- {line}" for line in contract_lines],
        ]
    )
    if body:
        return f"{contract_section}\n\n{body}\n"
    return f"{contract_section}\n"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _normalize_heading(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return _SECTION_ALIASES.get(cleaned, cleaned.replace(" ", "_"))


def _parse_markdown_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in markdown.splitlines():
        heading = re.match(r"^##+\s+(.+?)\s*$", line.strip())
        if heading:
            current_key = _normalize_heading(heading.group(1))
            sections.setdefault(current_key, [])
            continue
        if current_key is None:
            continue
        sections[current_key].append(line.rstrip())
    return {
        key: _extract_items(lines)
        for key, lines in sections.items()
    }


def _extract_items(lines: list[str]) -> list[str]:
    items: list[str] = []
    paragraph_buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if paragraph_buffer:
                items.append(" ".join(paragraph_buffer).strip())
                paragraph_buffer.clear()
            continue

        bullet = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", stripped)
        if bullet:
            if paragraph_buffer:
                items.append(" ".join(paragraph_buffer).strip())
                paragraph_buffer.clear()
            items.append(bullet.group(1).strip())
            continue
        paragraph_buffer.append(stripped)
    if paragraph_buffer:
        items.append(" ".join(paragraph_buffer).strip())
    return [item for item in items if item]


def _render_section(title: str, items: list[str]) -> list[str]:
    lines = [f"## {title}"]
    if items:
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append("- None recorded.")
    lines.append("")
    return lines


def _first_item(items: list[str] | None) -> str | None:
    if not items:
        return None
    return items[0]


def _extract_commands(items: list[str]) -> list[str]:
    commands: list[str] = []
    for item in items:
        match = re.match(r"^`(.+)`$", item)
        commands.append((match.group(1) if match else item).strip())
    return [command for command in commands if command]


def _has_concrete_anchor(task: str) -> bool:
    return any(
        re.search(pattern, task)
        for pattern in (
            r"[./][\w./-]+",  # file path
            r"#\d+",  # issue/pr number
            r"\b[a-z]+[A-Z][A-Za-z0-9_]+\b",  # camelCase
            r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9_]+\b",  # PascalCase symbol
            r"\b[a-z0-9]+_[a-z0-9_]+\b",  # snake_case
            r"```",  # code block
            r"\bpytest\b|\bnpm test\b|\bcargo test\b|\bmypy\b|\bruff\b",
            r"\bacceptance criteria\b",
            r"\bTypeError\b|\bAssertionError\b|\bTraceback\b",
        )
    )
