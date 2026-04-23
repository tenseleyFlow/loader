"""Workflow routing and artifact persistence for Loader runtime modes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from ..llm.base import ToolCall
from .clarify_grounding import ClarifyGrounding
from .dod import (
    all_planned_artifacts_exist,
    collect_planned_artifact_targets,
    planned_artifact_target_satisfied,
    slugify,
)
from .workflow_policy import (
    ArtifactEvidence,
    ArtifactEvidenceKind,
    ArtifactFreshness,
    ClarifyReview,
    ModeDecision,
    ModeRouter,
    WorkflowDecisionKind,
    WorkflowMode,
    WorkflowPolicy,
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
)
from .workflow_signals import WorkflowSignalExtractor, WorkflowSignalPacket

__all__ = [
    "ArtifactEvidence",
    "ArtifactEvidenceKind",
    "ArtifactFreshness",
    "ClarifyBrief",
    "ClarifyReview",
    "ModeDecision",
    "ModeRouter",
    "PlanningArtifacts",
    "VERIFICATION_SEPARATOR",
    "WorkflowArtifactStore",
    "WorkflowDecisionKind",
    "WorkflowMode",
    "WorkflowPolicy",
    "WorkflowSignalExtractor",
    "WorkflowSignalPacket",
    "WorkflowTimelineEntry",
    "WorkflowTimelineEntryKind",
    "advance_todos_from_tool_call",
    "build_execute_bridge",
    "effective_pending_todo_items",
    "enrich_clarify_brief_with_grounding",
    "extract_verification_commands_from_markdown",
    "infer_pending_todo_output_target",
    "load_brief",
    "load_planning_artifacts",
    "merge_refreshed_todos_with_existing_scope",
    "preserve_task_grounded_acceptance_criteria",
    "preferred_pending_todo_item",
    "reconcile_aggregate_completion_steps",
    "sync_todos_to_definition_of_done",
    "todo_file_candidates",
]

VERIFICATION_SEPARATOR = "<<<VERIFICATION>>>"
_VERIFICATION_SEPARATORS = (
    VERIFICATION_SEPARATOR,
    "<<VERIFICATION>>",
)
_GENERIC_TOUCHPOINTS = {
    "Determine the concrete files during execution.",
    "Identify exact files during planning or execution.",
}
_GENERIC_CONSTRAINTS = {
    "Honor the clarified answer and existing repository conventions.",
    "Preserve the existing codebase conventions and tests.",
}
_GENERIC_ASSUMPTIONS = {
    "Unspecified details stay unchanged unless evidence says otherwise.",
}
_SPECIAL_TODO_ITEMS = {
    "Complete the requested work",
    "Collect verification evidence",
}
_READ_STEP_HINTS = (
    "read",
    "examine",
    "inspect",
    "review",
    "check",
    "look at",
    "look through",
    "open",
    "understand",
    "study",
)
_SEARCH_STEP_HINTS = (
    "list",
    "find",
    "search",
    "scan",
    "discover",
    "locate",
    "enumerate",
    "gather",
)
_PARSE_STEP_HINTS = (
    "parse",
    "extract",
    "identify",
    "map",
    "determine",
)
_MUTATION_STEP_HINTS = (
    "create",
    "creating",
    "develop",
    "developing",
    "populate",
    "populating",
    "build",
    "building",
    "update",
    "updating",
    "edit",
    "editing",
    "write",
    "writing",
    "fix",
    "fixing",
    "modify",
    "modifying",
    "change",
    "changing",
    "patch",
    "patching",
    "replace",
    "replacing",
    "correct",
    "correcting",
    "rewrite",
    "rewriting",
)
_CREATION_STEP_HINTS = (
    "create",
    "creating",
    "generate",
    "generating",
    "scaffold",
    "scaffolding",
)
_VERIFY_STEP_HINTS = (
    "verify",
    "validation",
    "validate",
    "test",
    "confirm",
    "check",
)
_AGGREGATE_TODO_HINTS = (
    "each ",
    "all ",
    "every ",
    "sequence",
    "multiple ",
    "across ",
    "consistently",
    "properly linked",
    "directory structure",
)
_ARTIFACT_SET_COMPLETION_HINTS = (
    "link",
    "links",
    "linked",
    "navigation",
    "consistency",
    "consistent",
    "formatted",
    "formatting",
    "review",
)
_BROAD_SETUP_HINTS = (
    "directory structure",
    "directories",
    "folders",
    "folder structure",
    "scaffold",
    "scaffolding",
)
_TODO_FILE_CANDIDATE_PATTERN = re.compile(
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]+"
)
_ACTIONABLE_STEP_VERBS = {
    "add",
    "apply",
    "build",
    "check",
    "confirm",
    "create",
    "document",
    "edit",
    "ensure",
    "fix",
    "implement",
    "inspect",
    "list",
    "move",
    "parse",
    "patch",
    "read",
    "refactor",
    "remove",
    "rename",
    "reorder",
    "rerun",
    "re-run",
    "review",
    "run",
    "search",
    "test",
    "update",
    "validate",
    "verify",
    "write",
}
_RETROSPECTIVE_STEP_VERBS = {
    "added",
    "applied",
    "built",
    "checked",
    "completed",
    "confirmed",
    "created",
    "edited",
    "ensured",
    "examined",
    "generated",
    "implemented",
    "inspected",
    "listed",
    "looked",
    "parsed",
    "patched",
    "read",
    "reviewed",
    "updated",
    "validated",
    "verified",
    "wrote",
}
_TASK_COVERAGE_STOP_WORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "into",
    "your",
    "have",
    "make",
    "will",
    "then",
    "each",
    "file",
    "files",
    "guide",
}
_SHELL_COMMAND_START = re.compile(
    r"(?<![\w/.-])("
    r"ls|grep|pytest|uv|python3?|html5validator|cargo|npm|node|mypy|ruff|find|git|cat|sed|head|tail|test|diff|cmp|bash|sh|make"
    r")\b"
)

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
    explicit_sections: list[str] = field(default_factory=list)

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
            explicit_sections=sorted(sections.keys()),
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
            self.assumptions = [
                "Unspecified details stay unchanged unless evidence says otherwise.",
            ]
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


def enrich_clarify_brief_with_grounding(
    brief: ClarifyBrief,
    grounding: ClarifyGrounding,
) -> ClarifyBrief:
    """Strengthen a clarify brief with grounded workspace hints."""

    hints = grounding.brief_hints()
    if not hints.has_content():
        return brief

    brief.likely_touchpoints, touchpoints_added = _merge_grounded_items(
        brief.likely_touchpoints,
        hints.likely_touchpoints,
        generic_markers=_GENERIC_TOUCHPOINTS,
    )
    brief.constraints, constraints_added = _merge_grounded_items(
        brief.constraints,
        hints.constraints,
        generic_markers=_GENERIC_CONSTRAINTS,
    )
    brief.assumptions, assumptions_added = _merge_grounded_items(
        brief.assumptions,
        hints.assumptions,
        generic_markers=_GENERIC_ASSUMPTIONS,
    )
    brief.acceptance_criteria, acceptance_added = _merge_grounded_items(
        brief.acceptance_criteria,
        hints.acceptance_criteria,
        generic_markers=set(),
    )

    if touchpoints_added:
        _mark_explicit_section(brief, "likely_touchpoints")
    if constraints_added:
        _mark_explicit_section(brief, "constraints")
    if assumptions_added:
        _mark_explicit_section(brief, "assumptions")
    if acceptance_added:
        _mark_explicit_section(brief, "acceptance_criteria")

    brief.fill_defaults()
    return brief


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

    def with_acceptance_criteria(self, acceptance_criteria: list[str]) -> PlanningArtifacts:
        """Return one copy with a rewritten acceptance-criteria section."""

        merged = [item.strip() for item in acceptance_criteria if item.strip()]
        if not merged or merged == self.acceptance_criteria:
            return self

        return PlanningArtifacts(
            implementation_markdown=self.implementation_markdown,
            verification_markdown=_replace_markdown_section_items(
                self.verification_markdown,
                "Acceptance Criteria",
                merged,
            ),
            verification_commands=list(self.verification_commands),
            acceptance_criteria=list(merged),
            implementation_steps=list(self.implementation_steps),
        )

    def with_file_changes(self, file_changes: list[str]) -> PlanningArtifacts:
        """Return one copy with a rewritten file-changes section."""

        normalized = [item.strip() for item in file_changes if item.strip()]
        if not normalized:
            return self

        return PlanningArtifacts(
            implementation_markdown=_replace_markdown_section_items(
                self.implementation_markdown,
                "File Changes",
                normalized,
            ),
            verification_markdown=self.verification_markdown,
            verification_commands=list(self.verification_commands),
            acceptance_criteria=list(self.acceptance_criteria),
            implementation_steps=list(self.implementation_steps),
        )

    def with_progress_context(
        self,
        *,
        touched_files: list[str],
        completed_items: list[str],
    ) -> PlanningArtifacts:
        """Return one copy that preserves already-confirmed execution progress."""

        progress_items: list[str] = []
        for raw_path in touched_files:
            path_text = str(raw_path).strip()
            if not path_text:
                continue
            progress_items.append(f"Already touched during execution: `{path_text}`.")
        for raw_item in completed_items:
            item = str(raw_item).strip()
            if not item or item in _SPECIAL_TODO_ITEMS:
                continue
            progress_items.append(f"Already completed during execution: {item}.")

        if not progress_items:
            return self

        return PlanningArtifacts(
            implementation_markdown=_replace_markdown_section_items(
                self.implementation_markdown,
                "Confirmed Progress",
                list(dict.fromkeys(progress_items)),
            ),
            verification_markdown=self.verification_markdown,
            verification_commands=list(self.verification_commands),
            acceptance_criteria=list(self.acceptance_criteria),
            implementation_steps=list(self.implementation_steps),
        )


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
    *,
    project_root: Path | None = None,
) -> None:
    """Reflect todo state into DoD pending/completed items."""

    special_pending = [
        item
        for item in dod.pending_items
        if item
        in {
            "Complete the requested work",
            "Collect verification evidence",
        }
    ]
    special_completed = [
        item
        for item in dod.completed_items
        if item
        in {
            "Complete the requested work",
            "Collect verification evidence",
        }
    ]
    existing_completed = {
        item.strip()
        for item in dod.completed_items
        if item.strip() and item not in _SPECIAL_TODO_ITEMS
    }

    pending: list[str] = []
    completed: list[str] = []
    for item in todos:
        status = str(item.get("status", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        active_form = str(item.get("active_form", "")).strip()
        label = active_form if status == "in_progress" else content
        if not label and not content:
            continue
        # Treat exact todo items as monotonic. If a successful tool call already
        # marked the same todo complete, a stale TodoWrite snapshot should not
        # regress it back to pending / in progress.
        if status != "completed" and (
            content in existing_completed or active_form in existing_completed
        ):
            completed.append(content or active_form or label)
            continue
        if status == "completed":
            completed.append(content or label)
        else:
            pending.append(label)

    dod.pending_items = list(dict.fromkeys(pending + special_pending))
    dod.completed_items = list(dict.fromkeys(completed + special_completed))

    if project_root is not None:
        _reopen_aggregate_completion_steps_for_missing_artifacts(
            dod,
            project_root=project_root,
        )
        _reopen_directory_content_steps_for_incomplete_artifacts(
            dod,
            project_root=project_root,
        )
        dod.pending_items = effective_pending_todo_items(
            dod,
            project_root=project_root,
        )


def effective_pending_todo_items(
    dod,
    *,
    project_root: Path | None = None,
) -> list[str]:
    """Return pending todo items after filtering stale artifact-expansion drift."""

    pending_items = [item for item in dod.pending_items if item.strip()]
    if not pending_items or project_root is None or dod.status == "fixing":
        return pending_items

    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=24,
    )
    if not planned_targets:
        return pending_items
    if not all_planned_artifacts_exist(dod, project_root=project_root, max_paths=24):
        return pending_items

    planned_files = {
        target.name.lower()
        for target, expect_directory in planned_targets
        if not expect_directory
    }
    if not planned_files:
        return pending_items

    filtered_items = [
        item
        for item in pending_items
        if not _todo_targets_unplanned_artifact(item, planned_files)
    ]
    filtered_items = [
        item
        for item in filtered_items
        if not _todo_describes_stale_creation_after_artifacts_exist(
            item,
            planned_files,
        )
    ]
    return [
        item
        for item in filtered_items
        if not _todo_describes_stale_discovery_after_artifacts_exist(item)
    ]


def preferred_pending_todo_item(
    dod,
    *,
    project_root: Path | None = None,
    missing_artifact: tuple[Path, bool] | None = None,
) -> str | None:
    """Return the most helpful pending todo for user-facing recovery nudges."""

    pending_items = [
        item
        for item in effective_pending_todo_items(dod, project_root=project_root)
        if item not in _SPECIAL_TODO_ITEMS
    ]
    if not pending_items:
        return None
    if missing_artifact is None:
        return pending_items[0]

    for item in pending_items:
        if _todo_is_subsumed_by_concrete_missing_artifact(item, missing_artifact):
            continue
        return item
    return pending_items[0]


def todo_file_candidates(item: str) -> list[Path]:
    """Extract explicit file references from a todo item in source order."""

    if item in _SPECIAL_TODO_ITEMS:
        return []
    seen: set[str] = set()
    candidates: list[Path] = []
    for match in _TODO_FILE_CANDIDATE_PATTERN.findall(item.strip()):
        normalized = match.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Path(normalized))
    return candidates


def infer_pending_todo_output_target(
    dod,
    item: str,
    *,
    project_root: Path | None = None,
) -> Path | None:
    """Infer the concrete file path a pending todo is asking the model to mutate."""

    root = project_root or Path.cwd()
    candidates = todo_file_candidates(item)
    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=root,
        max_paths=12,
    )

    if candidates:
        planned_files = {
            target.name.lower(): target
            for target, expect_directory in planned_targets
            if not expect_directory
        }
        planned_directories = [
            target
            for target, expect_directory in planned_targets
            if expect_directory
        ]
        touched_paths = [
            Path(path)
            for path in dod.touched_files
            if str(path).strip()
        ]

        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate.is_absolute() or candidate_str.startswith("~"):
                return Path(candidate_str).expanduser()

            planned_match = planned_files.get(candidate.name.lower())
            if planned_match is not None:
                return planned_match

            for touched in reversed(touched_paths):
                if touched.name.lower() == candidate.name.lower():
                    continue
                if candidate.suffix and touched.suffix.lower() != candidate.suffix.lower():
                    continue
                return touched.parent / candidate.name

            for directory in planned_directories:
                return directory / candidate.name

    target_label = _normalize_pending_output_label(item)
    if not target_label:
        return None

    matches: list[tuple[int, bool, Path]] = []
    for html_file in _pending_item_html_sources(
        dod,
        project_root=root,
    ):
        try:
            content = html_file.read_text()
        except OSError:
            continue
        for href, link_text in _iter_local_html_links(content):
            resolved = (html_file.parent / href).resolve(strict=False)
            score = _pending_output_link_match_score(
                target_label,
                _normalize_pending_output_label(link_text),
            )
            if score <= 0:
                continue
            matches.append((score, not resolved.exists(), resolved))

    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
    return matches[0][2]


def preserve_task_grounded_acceptance_criteria(
    task_statement: str,
    *,
    existing_acceptance_criteria: list[str],
    refreshed_acceptance_criteria: list[str],
) -> list[str]:
    """Preserve task-grounded scope when refreshed artifacts are narrower."""

    grounded_existing = [
        item
        for item in existing_acceptance_criteria
        if item.strip()
        and item.strip().lower() != task_statement.strip().lower()
        and _task_text_covers_requirement(task_statement, item)
    ]
    return list(dict.fromkeys([*grounded_existing, *refreshed_acceptance_criteria]))


def _pending_item_html_sources(
    dod,
    *,
    project_root: Path,
) -> list[Path]:
    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    )
    html_sources: list[Path] = []
    seen: set[str] = set()

    for raw_path in dod.touched_files:
        path = Path(raw_path).expanduser().resolve(strict=False)
        if path.suffix.lower() not in {".html", ".htm"}:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        html_sources.append(path)

    for target, expect_directory in planned_targets:
        if expect_directory or target.suffix.lower() not in {".html", ".htm"}:
            continue
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        html_sources.append(target)

    return html_sources


def _normalize_pending_output_label(value: str) -> str:
    text = " ".join(str(value).strip().split()).lower()
    if not text:
        return ""
    text = re.sub(
        r"^(?:working on:\s*)?(?:create|creating|write|writing|build|building|develop|developing)\s+",
        "",
        text,
    )
    text = re.sub(r"\bfor nginx guide\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _pending_output_link_match_score(todo_label: str, link_label: str) -> int:
    if not todo_label or not link_label:
        return 0
    if todo_label == link_label:
        return 3
    if todo_label in link_label or link_label in todo_label:
        return 2
    todo_tokens = {token for token in todo_label.split() if len(token) > 2}
    link_tokens = {token for token in link_label.split() if len(token) > 2}
    if not todo_tokens or not link_tokens:
        return 0
    overlap = todo_tokens & link_tokens
    if len(overlap) >= min(3, len(todo_tokens), len(link_tokens)):
        return 1
    return 0


def _iter_local_html_links(content: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"<a\b[^>]*href\s*=\s*[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    links: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for href, inner_html in pattern.findall(content):
        target = href.strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        trimmed_target = target.split("?", 1)[0].split("#", 1)[0]
        if Path(trimmed_target).suffix.lower() not in {".html", ".htm"}:
            continue
        label = re.sub(r"<[^>]+>", " ", inner_html)
        label = " ".join(label.split())
        key = (trimmed_target, label)
        if key in seen:
            continue
        seen.add(key)
        links.append((trimmed_target, label))
    return links


def merge_refreshed_todos_with_existing_scope(
    task_statement: str,
    *,
    existing_pending_items: list[str],
    existing_completed_items: list[str],
    refreshed_steps: list[str],
    planned_files: set[str] | None = None,
) -> list[dict[str, str]]:
    """Merge one refreshed plan with task-grounded todo scope already in flight."""

    grounded_completed = [
        item
        for item in existing_completed_items
        if item.strip()
        and item not in _SPECIAL_TODO_ITEMS
        and _task_text_covers_requirement(task_statement, item)
    ]
    grounded_pending = [
        item
        for item in existing_pending_items
        if item.strip()
        and item not in _SPECIAL_TODO_ITEMS
        and _task_text_covers_requirement(task_statement, item)
    ]
    refreshed_candidates = [
        item.strip()
        for item in refreshed_steps
        if item.strip()
        and (
            not (grounded_completed or grounded_pending)
            or _looks_actionable_refresh_step(item)
        )
    ]
    if planned_files:
        refreshed_candidates = [
            item
            for item in refreshed_candidates
            if not _todo_targets_unplanned_artifact(item, planned_files)
        ]

    todos: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in grounded_completed:
        if item in seen:
            continue
        seen.add(item)
        todos.append(
            {
                "content": item,
                "active_form": f"Working on: {item}",
                "status": "completed",
            }
        )
    for item in [*grounded_pending, *refreshed_candidates]:
        label = item.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        todos.append(
            {
                "content": label,
                "active_form": f"Working on: {label}",
                "status": "pending",
            }
        )
    return todos


def advance_todos_from_tool_call(dod, tool_call: ToolCall) -> bool:
    """Advance the best-matching pending todo from a successful tool call."""

    best_index: int | None = None
    best_score = 0

    for index, item in enumerate(dod.pending_items):
        label = item.strip()
        if not label or label in _SPECIAL_TODO_ITEMS:
            continue
        score = _todo_progress_score(label, tool_call)
        if score > best_score:
            best_index = index
            best_score = score

    if best_index is None or best_score <= 0:
        return False

    completed = dod.pending_items.pop(best_index)
    if completed not in dod.completed_items:
        dod.completed_items.append(completed)
    return True


def _todo_progress_score(item: str, tool_call: ToolCall) -> int:
    text = item.lower()
    name = tool_call.name
    file_path = str(tool_call.arguments.get("file_path", "")).strip().lower()
    path = str(tool_call.arguments.get("path", "")).strip().lower()
    pattern = str(tool_call.arguments.get("pattern", "")).strip().lower()
    command = str(tool_call.arguments.get("command", "")).strip().lower()
    combined = " ".join(part for part in (file_path, path, pattern, command) if part)

    path_hint = file_path or path
    basename = Path(path_hint).name.lower() if path_hint else ""
    parent = Path(path_hint).parent.name.lower() if path_hint else ""

    score = 0
    is_discovery_tool = name in {"read", "glob", "grep"}
    if name == "bash":
        is_discovery_tool = _looks_like_search_command(command) or _looks_like_read_command(command)
    if (
        is_discovery_tool
        and _contains_any(text, _MUTATION_STEP_HINTS)
        and not (
            _contains_any(text, _READ_STEP_HINTS)
            or _contains_any(text, _SEARCH_STEP_HINTS)
            or _contains_any(text, _PARSE_STEP_HINTS)
            or _contains_any(text, _VERIFY_STEP_HINTS)
        )
    ):
        return 0

    if basename and basename in text:
        score += 3
    if parent and parent not in {"", "."} and parent in text:
        score += 2
    if "index" in text and "index" in combined:
        score += 2
    if "chapter" in text and ("chapter" in basename or "chapters" in combined):
        score += 1
    if "html" in text and ".html" in combined:
        score += 1

    if name == "read":
        if _contains_any(text, _READ_STEP_HINTS):
            score += 2
        if _contains_any(text, _PARSE_STEP_HINTS) and ".html" in combined:
            score += 1
    elif name in {"glob", "grep"}:
        if not (
            _contains_any(text, _SEARCH_STEP_HINTS)
            or _contains_any(text, _READ_STEP_HINTS)
            or _contains_any(text, _PARSE_STEP_HINTS)
        ):
            return 0
        if _contains_any(text, _SEARCH_STEP_HINTS):
            score += 2
        if name == "glob" and _contains_any(text, _READ_STEP_HINTS) and ".html" in combined:
            score += 1
    elif name == "bash":
        if _looks_like_verification_command(command):
            if _contains_any(text, _VERIFY_STEP_HINTS):
                score += 3
        elif _looks_like_search_command(command):
            if _contains_any(text, _SEARCH_STEP_HINTS):
                score += 2
        elif _looks_like_read_command(command):
            if _contains_any(text, _READ_STEP_HINTS):
                score += 2
        elif _looks_like_fs_mutation_command(command):
            if _contains_any(text, _MUTATION_STEP_HINTS):
                score += 3
            if "directory" in text and "mkdir" in command:
                score += 2
    elif name in {"write", "edit", "patch"}:
        if _todo_describes_aggregate_mutation(text) and basename and basename not in text:
            return 0
        if _contains_any(text, _MUTATION_STEP_HINTS):
            score += 3

    if name in {"write", "edit", "patch"} and _contains_any(text, _VERIFY_STEP_HINTS):
        return 0
    return score


def _contains_any(text: str, candidates: tuple[str, ...]) -> bool:
    return any(candidate in text for candidate in candidates)


def _todo_describes_aggregate_mutation(text: str) -> bool:
    return (
        _contains_any(text, _AGGREGATE_TODO_HINTS)
        or _todo_mentions_plural_output_set(text)
    ) and _contains_any(
        text,
        _MUTATION_STEP_HINTS,
    )


def _todo_requires_complete_artifact_set(text: str) -> bool:
    return (
        _contains_any(text, _AGGREGATE_TODO_HINTS)
        or _todo_mentions_plural_output_set(text)
    ) and _contains_any(
        text,
        _ARTIFACT_SET_COMPLETION_HINTS,
    )


def _todo_mentions_plural_output_set(text: str) -> bool:
    if _TODO_FILE_CANDIDATE_PATTERN.search(text):
        return False
    return any(
        phrase in text
        for phrase in (
            "chapter files",
            "all chapters",
            "chapters",
            "files following",
            "files with",
            "output files",
            "artifacts",
            "documents",
            "sections",
            "pages",
        )
    )


def _todo_targets_unplanned_artifact(item: str, planned_files: set[str]) -> bool:
    if item in _SPECIAL_TODO_ITEMS:
        return False

    text = item.strip().lower()
    if not text or not _contains_any(text, _MUTATION_STEP_HINTS):
        return False

    candidates = {
        Path(match).name.lower()
        for match in _TODO_FILE_CANDIDATE_PATTERN.findall(text)
    }
    if not candidates:
        return False

    return candidates.isdisjoint(planned_files)


def _todo_describes_stale_discovery_after_artifacts_exist(item: str) -> bool:
    text = item.strip().lower()
    if not text or item in _SPECIAL_TODO_ITEMS:
        return False
    if _contains_any(text, _VERIFY_STEP_HINTS):
        return False
    if _contains_any(text, _MUTATION_STEP_HINTS):
        return False
    if _contains_any(text, _ARTIFACT_SET_COMPLETION_HINTS):
        return False
    return (
        _contains_any(text, _READ_STEP_HINTS)
        or _contains_any(text, _SEARCH_STEP_HINTS)
        or _contains_any(text, _PARSE_STEP_HINTS)
    )


def _todo_describes_stale_creation_after_artifacts_exist(
    item: str,
    planned_files: set[str],
) -> bool:
    text = item.strip().lower()
    if not text or item in _SPECIAL_TODO_ITEMS:
        return False
    if _contains_any(text, _VERIFY_STEP_HINTS):
        return False
    if not _contains_any(text, _CREATION_STEP_HINTS):
        return False
    candidates = {
        Path(match).name.lower()
        for match in _TODO_FILE_CANDIDATE_PATTERN.findall(text)
    }
    if not candidates:
        return False
    return not candidates.isdisjoint(planned_files)


def _todo_is_subsumed_by_concrete_missing_artifact(
    item: str,
    missing_artifact: tuple[Path, bool],
) -> bool:
    text = item.strip().lower()
    if not text or item in _SPECIAL_TODO_ITEMS:
        return False
    if not _contains_any(text, _CREATION_STEP_HINTS):
        return False
    if _TODO_FILE_CANDIDATE_PATTERN.search(text):
        return False

    target, expect_directory = missing_artifact
    if _contains_any(text, _BROAD_SETUP_HINTS):
        if not expect_directory:
            return True
        return target.is_dir()
    return False


def _todo_describes_directory_content_creation(
    item: str,
    directories: list[Path],
) -> bool:
    text = item.strip().lower()
    if not text or item in _SPECIAL_TODO_ITEMS:
        return False
    if not _contains_any(text, _CREATION_STEP_HINTS):
        return False
    if not any(
        token in text
        for token in (
            "file",
            "files",
            "chapter",
            "chapters",
            "page",
            "pages",
            "artifact",
            "artifacts",
            "content",
            "test",
            "tests",
        )
    ):
        return False

    for directory in directories:
        name = directory.name.lower()
        tokens = {name}
        if name.endswith("ies") and len(name) > 3:
            tokens.add(f"{name[:-3]}y")
        elif name.endswith("s") and len(name) > 3:
            tokens.add(name[:-1])
        if any(token in text for token in tokens):
            return True
    return False


def _reopen_aggregate_completion_steps_for_missing_artifacts(
    dod,
    *,
    project_root: Path,
) -> None:
    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    )
    if not planned_targets:
        return

    if all_planned_artifacts_exist(dod, project_root=project_root, max_paths=12):
        return

    retained_completed: list[str] = []
    reopened_pending: list[str] = []
    for item in dod.completed_items:
        text = item.strip().lower()
        if item in _SPECIAL_TODO_ITEMS or not _todo_requires_complete_artifact_set(text):
            retained_completed.append(item)
            continue
        reopened_pending.append(item)

    if not reopened_pending:
        return

    dod.completed_items = retained_completed
    dod.pending_items = list(dict.fromkeys(dod.pending_items + reopened_pending))


def _reopen_directory_content_steps_for_incomplete_artifacts(
    dod,
    *,
    project_root: Path,
) -> None:
    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    )
    if not planned_targets:
        return

    incomplete_directories = [
        target
        for target, expect_directory in planned_targets
        if expect_directory
        and not planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=True,
            project_root=project_root,
        )
    ]
    if not incomplete_directories:
        return

    retained_completed: list[str] = []
    reopened_pending: list[str] = []
    for item in dod.completed_items:
        if item in _SPECIAL_TODO_ITEMS:
            retained_completed.append(item)
            continue
        if _todo_describes_directory_content_creation(item, incomplete_directories):
            reopened_pending.append(item)
            continue
        retained_completed.append(item)

    if not reopened_pending:
        return

    dod.completed_items = retained_completed
    dod.pending_items = list(dict.fromkeys(dod.pending_items + reopened_pending))


def reconcile_aggregate_completion_steps(
    dod,
    *,
    project_root: Path | None,
) -> None:
    """Reopen aggregate completion steps when planned artifacts are still missing."""

    if project_root is None:
        return
    _reopen_aggregate_completion_steps_for_missing_artifacts(
        dod,
        project_root=project_root,
    )


def _looks_like_search_command(command: str) -> bool:
    return any(token in command for token in (" ls", "ls ", "find ", "rg ", "grep ", "glob "))


def _looks_like_read_command(command: str) -> bool:
    return any(token in command for token in ("cat ", "sed ", "head ", "tail "))


def _looks_like_verification_command(command: str) -> bool:
    return any(
        token in command
        for token in (
            "pytest",
            "unittest",
            " test",
            " check",
            " verify",
            "html5validator",
            "mypy",
            "ruff",
            "lint",
            "grep ",
            "diff ",
            "cmp ",
        )
    )


def _looks_like_fs_mutation_command(command: str) -> bool:
    stripped = command.strip()
    return any(
        stripped.startswith(prefix)
        for prefix in (
            "mkdir ",
            "mkdir\t",
            "touch ",
            "touch\t",
            "cp ",
            "cp\t",
            "mv ",
            "mv\t",
            "ln ",
            "ln\t",
            "install ",
            "install\t",
        )
    )


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
    for separator in _VERIFICATION_SEPARATORS:
        if separator not in model_output:
            continue
        implementation, verification = model_output.split(separator, maxsplit=1)
        split = _split_embedded_verification_heading(
            implementation.strip(),
            fallback_verification=verification.strip(),
        )
        if split is not None:
            return split
        return implementation.strip(), verification.strip()

    split = _split_embedded_verification_heading(model_output.strip())
    if split is not None:
        return split
    return model_output.strip(), ""


def _split_embedded_verification_heading(
    implementation_markdown: str,
    *,
    fallback_verification: str = "",
) -> tuple[str, str] | None:
    match = re.search(r"(?m)^#\s+Verification Plan\s*$", implementation_markdown)
    if match is None:
        if fallback_verification.strip():
            return implementation_markdown, fallback_verification.strip()
        return None

    implementation = implementation_markdown[:match.start()].rstrip()
    verification = implementation_markdown[match.start():].strip()
    if not implementation:
        implementation = implementation_markdown.strip()
        verification = fallback_verification.strip()
    if not verification:
        verification = fallback_verification.strip()
    if not implementation or not verification:
        return None
    return implementation, verification


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


def _replace_markdown_section_items(
    markdown: str,
    title: str,
    items: list[str],
) -> str:
    lines = markdown.rstrip().splitlines()
    heading = f"## {title}".lower()
    start_index: int | None = None
    end_index = len(lines)
    for index, line in enumerate(lines):
        if line.strip().lower() == heading:
            start_index = index
            continue
        if start_index is not None and re.match(r"^##+\s+", line.strip()):
            end_index = index
            break

    replacement = _render_section(title, items)
    if start_index is None:
        body = "\n".join(lines).rstrip()
        suffix = "\n\n" if body else ""
        replacement_text = "\n".join(replacement).rstrip()
        return f"{body}{suffix}{replacement_text}\n"

    updated = [*lines[:start_index], *replacement, *lines[end_index:]]
    return "\n".join(updated).rstrip() + "\n"


def _first_item(items: list[str] | None) -> str | None:
    if not items:
        return None
    return items[0]


def _merge_grounded_items(
    existing: list[str],
    grounded: list[str],
    *,
    generic_markers: set[str],
) -> tuple[list[str], bool]:
    current = [item.strip() for item in existing if item.strip()]
    if not grounded:
        return current, False

    meaningful = [item for item in current if item not in generic_markers]
    merged = list(dict.fromkeys([*meaningful, *grounded]))
    return merged, merged != current


def _task_text_covers_requirement(task_text: str, requirement: str) -> bool:
    normalized_text = task_text.lower()
    normalized_requirement = requirement.lower()
    if normalized_requirement in normalized_text:
        return True
    if (
        _requirement_describes_output_scope(normalized_requirement)
        and _task_mentions_multiple_outputs(normalized_text)
    ):
        return True

    tokens = [
        token
        for token in re.findall(r"[a-z0-9_./-]+", normalized_requirement)
        if len(token) > 2 and token not in _TASK_COVERAGE_STOP_WORDS
    ]
    if not tokens:
        return normalized_requirement.strip() in normalized_text
    matches = sum(1 for token in tokens if token in normalized_text)
    threshold = max(1, min(2, len(tokens)))
    return matches >= threshold


def _task_mentions_multiple_outputs(task_text: str) -> bool:
    matches = re.findall(
        r"(?:~/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]+)?|"
        r"/(?:Users|home|tmp|var|private)/(?:[A-Za-z0-9_. -]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9]+)?|"
        r"[A-Za-z0-9_.-]+\.html|chapters/)",
        task_text,
    )
    if len(matches) >= 2:
        return True
    if matches and re.search(
        r"\b(chapter files?|files?|directories|directory structure|folders|pages|artifacts?|outputs?)\b",
        task_text,
    ):
        return True
    return False


def _requirement_describes_output_scope(requirement: str) -> bool:
    return any(
        phrase in requirement
        for phrase in (
            "all files",
            "file naming",
            "correct locations",
            "directory structure",
            "proper directory structure",
            "all links",
            "no broken links",
            "formatted and consistent",
            "consistent in style",
        )
    )


def _looks_actionable_refresh_step(step: str) -> bool:
    normalized = step.strip()
    if not normalized:
        return False
    if re.fullmatch(r"(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+", normalized):
        return False

    lowered = normalized.lower()
    lowered = re.sub(r"^(?:first|next|then|finally|afterward|afterwards)\b[,:]?\s*", "", lowered)
    first_word_match = re.match(r"^[a-z-]+", lowered)
    if first_word_match is None:
        return False

    first_word = first_word_match.group(0)
    if first_word in _RETROSPECTIVE_STEP_VERBS:
        return False
    if first_word in _ACTIONABLE_STEP_VERBS:
        return True
    return False


def _mark_explicit_section(brief: ClarifyBrief, section: str) -> None:
    if section in brief.explicit_sections:
        return
    brief.explicit_sections = sorted([*brief.explicit_sections, section])


def _extract_commands(items: list[str]) -> list[str]:
    commands: list[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue

        # Code fences often contain shell comments plus the actual command lines.
        if "```" in text:
            text = text.replace("```bash", "```").replace("```sh", "```")
            if "\n" not in text:
                commands.extend(_extract_collapsed_shell_commands(text))
                continue

        lines = text.splitlines() if "\n" in text or "```" in text else [text]
        for line in lines:
            candidate = line.strip()
            if not candidate or candidate.startswith("```"):
                continue
            candidate = re.sub(r"^-\s+", "", candidate)
            inline_candidate = _extract_inline_shell_command(candidate)
            if inline_candidate:
                candidate = inline_candidate
            else:
                match = re.match(r"^`(.+)`$", candidate)
                candidate = (match.group(1) if match else candidate).strip()
                candidate = _extract_shell_command_from_text(candidate)
            candidate = candidate.strip().strip("`")
            if candidate:
                commands.append(candidate)
    return _merge_continued_shell_commands([command for command in commands if command])


def _extract_collapsed_shell_commands(text: str) -> list[str]:
    stripped = re.sub(r"```(?:\w+)?", "", text).strip()
    if not stripped:
        return []

    matches = list(_SHELL_COMMAND_START.finditer(stripped))
    if not matches:
        extracted = _extract_shell_command_from_text(stripped)
        return [extracted] if extracted else []

    commands: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        candidate = stripped[start:end].strip()
        if candidate:
            commands.append(candidate)
    return commands


def _extract_inline_shell_command(text: str) -> str:
    for match in re.finditer(r"`([^`\n]+)`", text):
        candidate = _extract_shell_command_from_text(match.group(1).strip())
        if candidate:
            return candidate.strip().strip("`")
    return ""


def _extract_shell_command_from_text(text: str) -> str:
    match = _SHELL_COMMAND_START.search(text)
    if match is None:
        return ""
    return text[match.start():].strip()


def _merge_continued_shell_commands(commands: list[str]) -> list[str]:
    merged: list[str] = []
    pending: str | None = None

    for command in commands:
        stripped = command.strip()
        if not stripped:
            continue

        if pending is not None:
            combined = f"{pending} {stripped}".strip()
            if _has_dangling_shell_continuation(combined):
                pending = combined
                continue
            merged.append(combined)
            pending = None
            continue

        if _has_dangling_shell_continuation(stripped):
            pending = stripped
            continue
        merged.append(stripped)

    if pending is not None:
        merged.append(pending.rstrip("|& ").strip())

    return [command for command in merged if command]


def _has_dangling_shell_continuation(command: str) -> bool:
    stripped = command.rstrip()
    return stripped.endswith("|") or stripped.endswith("&&") or stripped.endswith("||")


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
