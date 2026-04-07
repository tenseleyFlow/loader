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
class ClarifyGrounding:
    """Cheap workspace evidence that clarify mode can reference."""

    project_type: str = "unknown"
    top_level_entries: list[str] = field(default_factory=list)
    existing_references: list[str] = field(default_factory=list)
    missing_references: list[str] = field(default_factory=list)
    candidate_touchpoints: list[str] = field(default_factory=list)

    def has_evidence(self) -> bool:
        return any(
            (
                self.top_level_entries,
                self.existing_references,
                self.missing_references,
                self.candidate_touchpoints,
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
        if self.missing_references:
            lines.append(
                "- Referenced paths not found: "
                + ", ".join(self.missing_references)
            )
        return "\n".join(lines) if lines else "- none"

    def primary_touchpoint(self) -> str | None:
        """Return the best available repo anchor for a focused question."""

        if self.existing_references:
            return self.existing_references[0]
        if self.candidate_touchpoints:
            return self.candidate_touchpoints[0]
        return None


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


def build_grounded_clarify_question(
    *,
    task: str,
    focus_slot: ClarifySlot | str | None,
    grounding: ClarifyGrounding,
    pressure_kind: ClarifyPressureKind | str | None = None,
) -> str | None:
    """Return a repo-grounded clarify question when local evidence is strong."""

    anchor = grounding.primary_touchpoint()
    if anchor is None:
        return None

    slot = (
        focus_slot
        if isinstance(focus_slot, ClarifySlot)
        else ClarifySlot(focus_slot)
        if focus_slot
        else ClarifySlot.DESIRED_OUTCOME
    )
    pressure = (
        pressure_kind
        if isinstance(pressure_kind, ClarifyPressureKind)
        else ClarifyPressureKind(pressure_kind)
        if pressure_kind
        else None
    )

    if slot == ClarifySlot.LIKELY_TOUCHPOINTS:
        if pressure == ClarifyPressureKind.EXAMPLE:
            return (
                f"I found `{anchor}` in the repo. Should that be the first concrete "
                "touchpoint, or is there a different file or subsystem I should use instead?"
            )
        if pressure == ClarifyPressureKind.TRADEOFF:
            return (
                f"I found `{anchor}` in the repo. Should I keep the work scoped there, "
                "and what nearby file or surface should stay unchanged?"
            )
        if pressure == ClarifyPressureKind.ASSUMPTION:
            return (
                f"I found `{anchor}` in the repo. What assumption about the right "
                "touchpoint would be risky for me to make without checking first?"
            )
        return (
            f"I found `{anchor}` in the repo. Should I keep this task scoped there, "
            "or is there a different file or subsystem you want me to prioritize?"
        )

    if slot in {ClarifySlot.NON_GOALS, ClarifySlot.DECISION_BOUNDARIES}:
        if pressure == ClarifyPressureKind.TRADEOFF:
            return (
                f"I can already see `{anchor}` in the workspace for `{task}`. "
                "Should I keep the change scoped there even if broader edits would be easier?"
            )
        return (
            f"I can already see `{anchor}` in the workspace for `{task}`. "
            "Should I keep the change scoped there, and what nearby surface should stay unchanged?"
        )

    return None
