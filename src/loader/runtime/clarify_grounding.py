"""Lightweight workspace evidence for clarify-mode follow-up questions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..context.project import detect_project_type, get_directory_structure
from .clarify_strategy import ClarifyPressureKind, ClarifySlot

_PATH_TOKEN_RE = re.compile(
    r"(?P<path>(?:[\w.-]+/)+[\w./-]+|[\w.-]+\.(?:py|md|toml|json|yaml|yml|txt|js|ts|tsx|jsx|rs|go|sh))"
)
_TOKEN_RE = re.compile(r"[a-z0-9_./-]+")
_SKIP_DIRS = {
    ".git",
    ".loader",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "refs",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "avoid",
    "broad",
    "change",
    "changes",
    "claw",
    "code",
    "do",
    "does",
    "for",
    "from",
    "how",
    "improve",
    "loader",
    "make",
    "more",
    "need",
    "project",
    "repo",
    "repository",
    "should",
    "task",
    "that",
    "the",
    "this",
    "tool",
    "use",
    "want",
    "work",
    "workflow",
}
_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".rs",
    ".go",
    ".sh",
}


@dataclass(slots=True)
class ClarifyRepoFact:
    """One concise fact Loader extracted from a matching workspace file."""

    path: str
    summary: str

    def render(self) -> str:
        """Render one repo fact for prompt display."""

        return f"`{self.path}`: {self.summary}"


@dataclass(slots=True)
class ClarifyBriefHints:
    """Grounded hints that can strengthen a persisted clarify brief."""

    likely_touchpoints: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)

    def has_content(self) -> bool:
        """Return whether any grounded hint is available."""

        return any(
            (
                self.likely_touchpoints,
                self.constraints,
                self.assumptions,
                self.acceptance_criteria,
            )
        )


@dataclass(slots=True)
class ClarifyGrounding:
    """Cheap workspace evidence that clarify mode can reference."""

    project_type: str = "unknown"
    top_level_entries: list[str] = field(default_factory=list)
    existing_references: list[str] = field(default_factory=list)
    missing_references: list[str] = field(default_factory=list)
    candidate_touchpoints: list[str] = field(default_factory=list)
    repo_facts: list[ClarifyRepoFact] = field(default_factory=list)

    def has_evidence(self) -> bool:
        return any(
            (
                self.top_level_entries,
                self.existing_references,
                self.missing_references,
                self.candidate_touchpoints,
                self.repo_facts,
            )
        )

    def prompt_block(self) -> str:
        """Render grounding as a concise prompt block."""

        lines: list[str] = []
        if self.project_type != "unknown":
            lines.append(f"- Project type: {self.project_type}")
        if self.top_level_entries:
            lines.append(
                "- Top-level entries: " + ", ".join(self.top_level_entries)
            )
        if self.existing_references:
            lines.append(
                "- Referenced paths that exist: "
                + ", ".join(self.existing_references)
            )
        if self.candidate_touchpoints:
            lines.append(
                "- Nearby repo touchpoints: "
                + ", ".join(self.candidate_touchpoints)
            )
        if self.repo_facts:
            lines.append(
                "- Observed repo facts: "
                + "; ".join(fact.render() for fact in self.repo_facts)
            )
        if self.missing_references:
            lines.append(
                "- Referenced paths not found: "
                + ", ".join(self.missing_references)
            )
        return "\n".join(lines) if lines else "- none"

    def slot_prompt_block(
        self,
        focus_slot: ClarifySlot | str | None,
        pressure_kind: ClarifyPressureKind | str | None = None,
    ) -> str:
        """Render the most relevant workspace evidence for one clarify slot."""

        slot = _resolve_slot(focus_slot)
        pressure = _resolve_pressure(pressure_kind)
        lines: list[str] = []
        if self.project_type != "unknown":
            lines.append(f"- Project type: {self.project_type}")

        relevant_facts = self.relevant_facts(slot, pressure)
        if relevant_facts:
            lines.append(
                "- Relevant repo facts: "
                + "; ".join(fact.render() for fact in relevant_facts)
            )
        elif self.repo_facts:
            lines.append(
                "- Observed repo facts: "
                + "; ".join(fact.render() for fact in self.repo_facts)
            )

        relevant_paths = self.relevant_paths(slot, pressure)
        if relevant_paths:
            lines.append(
                "- Relevant paths: " + ", ".join(relevant_paths)
            )
        elif self.existing_references:
            lines.append(
                "- Referenced paths that exist: "
                + ", ".join(self.existing_references)
            )

        if self.missing_references:
            lines.append(
                "- Referenced paths not found: "
                + ", ".join(self.missing_references)
            )
        if not lines:
            return self.prompt_block()
        return "\n".join(lines)

    def brief_hints(self) -> ClarifyBriefHints:
        """Return grounded hints for clarify brief synthesis and fallback repair."""

        primary_path = self.primary_touchpoint()
        secondary_path = self.secondary_touchpoint()
        primary_fact = self.primary_fact()
        secondary_fact = self.secondary_fact()

        likely_touchpoints = [
            path for path in [primary_path, secondary_path] if path is not None
        ][:2]

        constraints: list[str] = []
        if primary_path is not None:
            constraints.append(
                f"Keep the primary implementation scoped to `{primary_path}` "
                "unless evidence requires a wider edit."
            )
        if secondary_path is not None:
            constraints.append(
                f"Preserve existing behavior in `{secondary_path}` unless the user broadens scope."
            )

        assumptions: list[str] = []
        for fact in [primary_fact, secondary_fact]:
            if fact is None:
                continue
            assumptions.append(
                f"Workspace evidence: `{fact.path}` currently contains `{fact.summary}`."
            )

        acceptance_criteria: list[str] = []
        if primary_path is not None:
            acceptance_criteria.append(
                f"Primary work stays scoped to `{primary_path}`."
            )
        if secondary_path is not None:
            acceptance_criteria.append(
                f"Nearby surface `{secondary_path}` stays unchanged unless "
                "the user confirms otherwise."
            )

        return ClarifyBriefHints(
            likely_touchpoints=likely_touchpoints,
            constraints=constraints,
            assumptions=assumptions,
            acceptance_criteria=acceptance_criteria,
        )

    def brief_prompt_block(self) -> str:
        """Render brief-oriented grounding hints for the brief synthesis prompt."""

        hints = self.brief_hints()
        if not hints.has_content():
            return self.prompt_block()

        lines: list[str] = []
        if hints.likely_touchpoints:
            lines.append(
                "- Seed likely touchpoints: " + ", ".join(hints.likely_touchpoints)
            )
        if hints.constraints:
            lines.append(
                "- Preserve constraints: " + "; ".join(hints.constraints)
            )
        if hints.assumptions:
            lines.append(
                "- Grounded assumptions: " + "; ".join(hints.assumptions)
            )
        if hints.acceptance_criteria:
            lines.append(
                "- Scope acceptance criteria: " + "; ".join(hints.acceptance_criteria)
            )
        return "\n".join(lines)

    def primary_touchpoint(self) -> str | None:
        """Return the best available repo anchor for a focused question."""

        if self.existing_references:
            return self.existing_references[0]
        if self.candidate_touchpoints:
            return self.candidate_touchpoints[0]
        return None

    def secondary_touchpoint(self) -> str | None:
        """Return a nearby path that differs from the primary touchpoint."""

        primary = self.primary_touchpoint()
        for candidate in [*self.existing_references, *self.candidate_touchpoints]:
            if candidate != primary:
                return candidate
        return None

    def primary_fact(self) -> ClarifyRepoFact | None:
        """Return the best extracted repo fact for grounding a question."""

        if not self.repo_facts:
            return None
        anchor = self.primary_touchpoint()
        if anchor is not None:
            for fact in self.repo_facts:
                if fact.path == anchor:
                    return fact
        return self.repo_facts[0]

    def secondary_fact(self) -> ClarifyRepoFact | None:
        """Return a nearby repo fact that differs from the primary touchpoint."""

        primary = self.primary_touchpoint()
        for fact in self.repo_facts:
            if fact.path != primary:
                return fact
        return None

    def relevant_facts(
        self,
        focus_slot: ClarifySlot | str | None,
        pressure_kind: ClarifyPressureKind | str | None = None,
    ) -> list[ClarifyRepoFact]:
        """Return the repo facts most useful for the requested clarify slot."""

        slot = _resolve_slot(focus_slot)
        pressure = _resolve_pressure(pressure_kind)
        primary = self.primary_fact()
        secondary = self.secondary_fact()
        if slot == ClarifySlot.LIKELY_TOUCHPOINTS:
            if pressure in {
                ClarifyPressureKind.EXAMPLE,
                ClarifyPressureKind.TRADEOFF,
                ClarifyPressureKind.ASSUMPTION,
            }:
                return [fact for fact in [primary, secondary] if fact is not None]
            return [fact for fact in [primary] if fact is not None]
        if slot in {ClarifySlot.NON_GOALS, ClarifySlot.DECISION_BOUNDARIES}:
            if pressure == ClarifyPressureKind.ASSUMPTION:
                return [fact for fact in [secondary, primary] if fact is not None]
            return [fact for fact in [primary, secondary] if fact is not None]
        if slot == ClarifySlot.CONSTRAINTS and secondary is not None:
            return [secondary]
        return [fact for fact in [primary] if fact is not None]

    def relevant_paths(
        self,
        focus_slot: ClarifySlot | str | None,
        pressure_kind: ClarifyPressureKind | str | None = None,
    ) -> list[str]:
        """Return the paths most useful for the requested clarify slot."""

        slot = _resolve_slot(focus_slot)
        pressure = _resolve_pressure(pressure_kind)
        primary = self.primary_touchpoint()
        secondary = self.secondary_touchpoint()
        if slot == ClarifySlot.LIKELY_TOUCHPOINTS:
            if pressure in {
                ClarifyPressureKind.EXAMPLE,
                ClarifyPressureKind.TRADEOFF,
                ClarifyPressureKind.ASSUMPTION,
            }:
                return [path for path in [primary, secondary] if path is not None]
            return [path for path in [primary] if path is not None]
        if slot in {ClarifySlot.NON_GOALS, ClarifySlot.DECISION_BOUNDARIES}:
            return [path for path in [primary, secondary] if path is not None]
        return [path for path in [primary] if path is not None]


class ClarifyGroundingProbe:
    """Collect a bounded amount of local evidence for clarify prompts."""

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        max_top_level_entries: int = 6,
        max_candidates: int = 4,
        max_walk_entries: int = 400,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.max_top_level_entries = max_top_level_entries
        self.max_candidates = max_candidates
        self.max_walk_entries = max_walk_entries

    def collect(
        self,
        *,
        task: str,
        rounds: list[tuple[str, str]] | None = None,
    ) -> ClarifyGrounding:
        """Return lightweight workspace evidence for the active clarify task."""

        texts = [task]
        for question, answer in rounds or []:
            texts.extend([question, answer])
        existing_references, missing_references = self._resolve_references(texts)
        keywords = self._keywords(texts)
        candidate_touchpoints = self._discover_touchpoints(
            keywords=keywords,
            excluded=set(existing_references),
        )
        repo_facts = self._collect_repo_facts(
            paths=[*existing_references, *candidate_touchpoints],
            keywords=keywords,
        )
        project_type, _ = detect_project_type(self.workspace_root)
        return ClarifyGrounding(
            project_type=project_type,
            top_level_entries=get_directory_structure(
                self.workspace_root,
                max_items=self.max_top_level_entries,
            ),
            existing_references=existing_references,
            missing_references=missing_references,
            candidate_touchpoints=candidate_touchpoints,
            repo_facts=repo_facts,
        )

    def _resolve_references(
        self,
        texts: list[str],
    ) -> tuple[list[str], list[str]]:
        existing: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()

        for text in texts:
            for match in _PATH_TOKEN_RE.finditer(text):
                token = match.group("path").strip("`'\",.:;()[]{}")
                normalized = token.rstrip("/")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                candidate = (self.workspace_root / normalized).resolve()
                if not self._is_within_workspace(candidate):
                    continue
                if candidate.exists():
                    suffix = "/" if candidate.is_dir() else ""
                    existing.append(f"{normalized}{suffix}")
                else:
                    missing.append(normalized)

        return existing[: self.max_candidates], missing[: self.max_candidates]

    def _discover_touchpoints(
        self,
        *,
        keywords: list[str],
        excluded: set[str],
    ) -> list[str]:
        if not keywords:
            return []

        scored: dict[str, tuple[int, int]] = {}
        visited = 0

        for root, dirs, files in os.walk(self.workspace_root):
            dirs[:] = [
                name
                for name in dirs
                if name not in _SKIP_DIRS and not name.startswith(".")
            ]
            relative_root = Path(root).relative_to(self.workspace_root)

            for name in sorted(dirs + files):
                path = Path(root) / name
                relative_path = (
                    Path(name)
                    if relative_root == Path(".")
                    else relative_root / name
                )
                rendered = str(relative_path)
                if rendered in excluded:
                    continue
                if path.is_file() and path.suffix and path.suffix not in _TEXT_SUFFIXES:
                    continue

                score = self._score_path(rendered, keywords)
                if score <= 0:
                    visited += 1
                    if visited >= self.max_walk_entries:
                        return self._finalize_candidates(scored)
                    continue

                depth = len(relative_path.parts)
                bonus = 1 if path.is_file() else 0
                previous = scored.get(rendered)
                candidate_score = (score + bonus, depth)
                if previous is None or candidate_score > previous:
                    scored[rendered] = candidate_score

                visited += 1
                if visited >= self.max_walk_entries:
                    return self._finalize_candidates(scored)

        return self._finalize_candidates(scored)

    @staticmethod
    def _keywords(texts: list[str]) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for token in _TOKEN_RE.findall(text.lower()):
                normalized = token.strip("./-_")
                if (
                    len(normalized) < 4
                    or normalized in seen
                    or normalized in _STOPWORDS
                    or "/" in normalized
                    or "." in normalized
                ):
                    continue
                seen.add(normalized)
                tokens.append(normalized)
        return tokens[:8]

    @staticmethod
    def _score_path(path_text: str, keywords: list[str]) -> int:
        lowered = path_text.lower()
        score = 0
        for keyword in keywords:
            if keyword in lowered:
                score += 2
            if any(part == keyword for part in Path(lowered).parts):
                score += 1
        return score

    def _finalize_candidates(
        self,
        scored: dict[str, tuple[int, int]],
    ) -> list[str]:
        ordered = sorted(
            scored.items(),
            key=lambda item: (-item[1][0], item[1][1], item[0]),
        )
        return [path for path, _ in ordered[: self.max_candidates]]

    def _is_within_workspace(self, path: Path) -> bool:
        try:
            path.relative_to(self.workspace_root)
        except ValueError:
            return False
        return True

    def _collect_repo_facts(
        self,
        *,
        paths: list[str],
        keywords: list[str],
    ) -> list[ClarifyRepoFact]:
        facts: list[ClarifyRepoFact] = []
        seen: set[str] = set()

        for rendered_path in paths:
            normalized = rendered_path.rstrip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidate = self.workspace_root / normalized
            if not candidate.exists() or candidate.is_dir():
                continue
            if candidate.suffix and candidate.suffix not in _TEXT_SUFFIXES:
                continue
            summary = self._extract_fact_summary(candidate, keywords)
            if not summary:
                continue
            facts.append(ClarifyRepoFact(path=normalized, summary=summary))
            if len(facts) >= self.max_candidates:
                break

        return facts

    def _extract_fact_summary(
        self,
        path: Path,
        keywords: list[str],
    ) -> str | None:
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read(8000)
        except OSError:
            return None

        best_line: str | None = None
        best_score = 0

        for raw_line in text.splitlines()[:120]:
            summary = self._normalize_fact_line(raw_line)
            if not summary:
                continue
            score = self._score_fact_line(summary, keywords)
            if score > best_score:
                best_score = score
                best_line = summary

        return best_line

    @staticmethod
    def _normalize_fact_line(raw_line: str) -> str | None:
        line = raw_line.strip()
        if not line:
            return None
        if line.startswith(("import ", "from ")):
            return None

        for prefix in ('"""', "'''", "#", "-", "*"):
            if line.startswith(prefix):
                line = line.removeprefix(prefix).strip()
        for suffix in ('"""', "'''"):
            if line.endswith(suffix):
                line = line.removesuffix(suffix).strip()

        if not line or len(line) < 4:
            return None
        if line in {"{", "}", "[", "]"}:
            return None
        if len(line) > 120:
            line = line[:117].rstrip() + "..."
        return line

    @staticmethod
    def _score_fact_line(line: str, keywords: list[str]) -> int:
        lowered = line.lower()
        score = 1
        if lowered.startswith("class "):
            score += 4
        elif lowered.startswith("def "):
            score += 3
        elif lowered.startswith(("## ", "# ", "[", "export ", "function ")):
            score += 2

        for keyword in keywords:
            if keyword in lowered:
                score += 2

        return score


def build_grounded_clarify_question(
    *,
    task: str,
    focus_slot: ClarifySlot | str | None,
    grounding: ClarifyGrounding,
    pressure_kind: ClarifyPressureKind | str | None = None,
) -> str | None:
    """Return a repo-grounded clarify question when local evidence is strong."""

    slot = _resolve_slot(focus_slot)
    anchor = grounding.primary_touchpoint()
    if anchor is None:
        return None
    fact = grounding.primary_fact()
    nearby_fact = grounding.secondary_fact()
    pressure = _resolve_pressure(pressure_kind)
    fact_clause = _render_repo_fact_clause(fact, anchor=anchor)
    nearby_clause = _render_nearby_repo_fact_clause(
        nearby_fact,
        anchor=anchor,
    )

    if slot == ClarifySlot.LIKELY_TOUCHPOINTS:
        if pressure == ClarifyPressureKind.EXAMPLE:
            return (
                f"I found `{anchor}` in the repo.{fact_clause}{nearby_clause} "
                "Should that be the first concrete touchpoint, and what nearby file "
                "should still count as the counterexample surface I leave alone?"
            )
        if pressure == ClarifyPressureKind.TRADEOFF:
            return (
                f"I found `{anchor}` in the repo.{fact_clause}{nearby_clause} "
                "Should I keep the work scoped there, "
                "and what nearby file or surface should stay unchanged?"
            )
        if pressure == ClarifyPressureKind.ASSUMPTION:
            return (
                f"I found `{anchor}` in the repo.{fact_clause}{nearby_clause} "
                "What assumption about the right touchpoint or nearby spillover "
                "would be risky for me to make without checking first?"
            )
        return (
            f"I found `{anchor}` in the repo.{fact_clause} Should I keep this task scoped there, "
            "or is there a different file or subsystem you want me to prioritize?"
        )

    if slot == ClarifySlot.NON_GOALS:
        if pressure == ClarifyPressureKind.EXAMPLE:
            return (
                f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
                f"{nearby_clause} What concrete change in `{anchor}` should count as "
                "in scope, and what nearby change should still count as out of scope?"
            )
        if pressure == ClarifyPressureKind.TRADEOFF:
            return (
                f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
                f"{nearby_clause} Should I keep the change scoped there and leave that nearby "
                "surface unchanged, even if broader edits would be easier?"
            )
        if pressure == ClarifyPressureKind.ASSUMPTION:
            return (
                f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
                f"{nearby_clause} What assumption about touching that nearby surface "
                "would be risky for me to make without checking first?"
            )
        return (
            f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
            f"{nearby_clause} Should I keep the change scoped there, and what nearby surface "
            "should stay unchanged?"
        )

    if slot == ClarifySlot.DECISION_BOUNDARIES:
        if pressure == ClarifyPressureKind.EXAMPLE:
            return (
                f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
                f"{nearby_clause} What concrete change here would you want me to make "
                "without asking, and what nearby change should still force a stop?"
            )
        if pressure == ClarifyPressureKind.TRADEOFF:
            return (
                f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
                f"{nearby_clause} If the fix starts pulling in that nearby surface, "
                "should that be a stop-and-confirm boundary?"
            )
        if pressure == ClarifyPressureKind.ASSUMPTION:
            return (
                f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
                f"{nearby_clause} What assumption about crossing into that nearby "
                "surface would be risky unless I stop and confirm first?"
            )
        return (
            f"I can already see `{anchor}` in the workspace for `{task}`.{fact_clause}"
            f"{nearby_clause} If the work starts expanding beyond `{anchor}`, "
            "what should count as the boundary where I stop and confirm?"
        )

    return None


def _resolve_slot(
    focus_slot: ClarifySlot | str | None,
) -> ClarifySlot:
    return (
        focus_slot
        if isinstance(focus_slot, ClarifySlot)
        else ClarifySlot(focus_slot)
        if focus_slot
        else ClarifySlot.DESIRED_OUTCOME
    )


def _resolve_pressure(
    pressure_kind: ClarifyPressureKind | str | None,
) -> ClarifyPressureKind | None:
    return (
        pressure_kind
        if isinstance(pressure_kind, ClarifyPressureKind)
        else ClarifyPressureKind(pressure_kind)
        if pressure_kind
        else None
    )


def _render_repo_fact_clause(
    fact: ClarifyRepoFact | None,
    *,
    anchor: str,
) -> str:
    if fact is None:
        return ""
    if fact.path != anchor:
        return f" Nearby, `{fact.path}` currently contains `{fact.summary}`."
    return f" It currently contains `{fact.summary}`."


def _render_nearby_repo_fact_clause(
    fact: ClarifyRepoFact | None,
    *,
    anchor: str,
) -> str:
    if fact is None or fact.path == anchor:
        return ""
    return f" Nearby, `{fact.path}` currently contains `{fact.summary}`."
