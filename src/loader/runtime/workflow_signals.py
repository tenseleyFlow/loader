"""Typed workflow-signal extraction for runtime policy decisions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow_policy import WorkflowTimelineEntry


@dataclass(slots=True)
class WorkflowSignalPacket:
    """Typed route context consumed by workflow policy."""

    task: str
    requested_mode: str | None = None
    has_brief: bool = False
    has_plan: bool = False
    allow_clarify: bool = True
    ambiguity_score: float = 0.0
    complexity_score: float = 0.0
    verification_pressure: float = 0.0
    mutation_pressure: float = 0.0
    artifact_reuse_pressure: float = 0.0
    stale_artifact_pressure: float = 0.0
    unresolved_questions: list[str] = field(default_factory=list)
    recent_clarify_count: int = 0
    recent_reentry_count: int = 0
    recent_plan_refresh_count: int = 0
    recent_verify_skip_count: int = 0
    signal_summary: list[str] = field(default_factory=list)


class WorkflowSignalExtractor:
    """Build typed workflow-signal packets from runtime state."""

    def extract_route_signals(
        self,
        task: str,
        *,
        requested_mode: str | None = None,
        has_brief: bool = False,
        has_plan: bool = False,
        allow_clarify: bool = True,
        verification_pressure: bool = False,
        mutating_history: bool = False,
        stale_plan: bool = False,
        unresolved_questions: list[str] | None = None,
        timeline: list[WorkflowTimelineEntry] | None = None,
    ) -> WorkflowSignalPacket:
        """Derive workflow signals from task state and recent timeline context."""

        unresolved_questions = list(unresolved_questions or [])
        recent_timeline = list(timeline or [])[-6:]
        recent_clarify_count = sum(
            1
            for entry in recent_timeline
            if entry.mode == "clarify" or entry.kind.startswith("clarify")
        )
        recent_reentry_count = sum(
            1
            for entry in recent_timeline
            if entry.kind == "reentry" or entry.decision_kind == "reentry"
        )
        recent_plan_refresh_count = sum(
            1
            for entry in recent_timeline
            if entry.kind == "plan_refresh" or "plan_refresh" in entry.reason_code
        )
        recent_verify_skip_count = sum(
            1
            for entry in recent_timeline
            if entry.kind == "verify_skip"
        )
        ambiguity_score = self._ambiguity_score(task)
        complexity_score = self._complexity_score(task)
        verification_signal = 0.18 if verification_pressure else 0.0
        mutation_signal = 0.12 if mutating_history else 0.0
        artifact_reuse_signal = 0.2 if has_plan else 0.0
        stale_artifact_signal = 0.45 if stale_plan else 0.0

        signal_summary: list[str] = [
            f"ambiguity={ambiguity_score:.2f}",
            f"complexity={complexity_score:.2f}",
        ]
        if requested_mode:
            signal_summary.append(f"requested_mode={requested_mode}")
        if has_brief:
            signal_summary.append("clarify_brief=available")
        if has_plan:
            signal_summary.append("plan_artifacts=available")
        if stale_plan:
            signal_summary.append("plan_artifacts=stale")
        if unresolved_questions:
            signal_summary.append(
                f"open_questions={min(len(unresolved_questions), 9)}"
            )
        if verification_pressure:
            signal_summary.append("verification_pressure=active")
        if mutating_history:
            signal_summary.append("mutation_pressure=active")
        if recent_clarify_count:
            signal_summary.append(f"recent_clarify={recent_clarify_count}")
        if recent_reentry_count:
            signal_summary.append(f"recent_reentry={recent_reentry_count}")
        if recent_plan_refresh_count:
            signal_summary.append(f"recent_plan_refresh={recent_plan_refresh_count}")
        if recent_verify_skip_count:
            signal_summary.append(f"recent_verify_skip={recent_verify_skip_count}")

        return WorkflowSignalPacket(
            task=task,
            requested_mode=requested_mode,
            has_brief=has_brief,
            has_plan=has_plan,
            allow_clarify=allow_clarify,
            ambiguity_score=ambiguity_score,
            complexity_score=complexity_score,
            verification_pressure=verification_signal,
            mutation_pressure=mutation_signal,
            artifact_reuse_pressure=artifact_reuse_signal,
            stale_artifact_pressure=stale_artifact_signal,
            unresolved_questions=unresolved_questions,
            recent_clarify_count=recent_clarify_count,
            recent_reentry_count=recent_reentry_count,
            recent_plan_refresh_count=recent_plan_refresh_count,
            recent_verify_skip_count=recent_verify_skip_count,
            signal_summary=signal_summary,
        )

    @staticmethod
    def _ambiguity_score(task: str) -> float:
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

        return round(min(score, 1.0), 3)

    @staticmethod
    def _complexity_score(task: str) -> float:
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

        return round(min(score, 1.0), 3)


def _has_concrete_anchor(task: str) -> bool:
    return bool(
        re.search(r"[./_\\-]", task)
        or re.search(r"`[^`]+`", task)
        or any(
            token in task.lower()
            for token in ("test", "file", "function", "class")
        )
    )
