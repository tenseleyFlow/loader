"""Workflow policy, clarify review, and timeline contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .workflow_signals import WorkflowSignalExtractor, WorkflowSignalPacket


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


class WorkflowDecisionKind(StrEnum):
    """Classification for why a workflow mode was selected."""

    INITIAL_ROUTE = "initial_route"
    REQUESTED = "requested"
    ARTIFACT_REUSE = "artifact_reuse"
    HANDOFF = "handoff"
    REENTRY = "reentry"
    FORCED = "forced"


class WorkflowTimelineEntryKind(StrEnum):
    """Workflow timeline entry categories."""

    ROUTE = "route"
    HANDOFF = "handoff"
    REENTRY = "reentry"
    CLARIFY_CONTINUE = "clarify_continue"
    CLARIFY_EXIT = "clarify_exit"
    PLAN_REFRESH = "plan_refresh"
    VERIFY_SKIP = "verify_skip"


@dataclass(slots=True)
class ModeDecision:
    """Workflow-policy output for one route or handoff."""

    mode: WorkflowMode
    reason_code: str
    reason_summary: str
    decision_kind: WorkflowDecisionKind = WorkflowDecisionKind.INITIAL_ROUTE
    ambiguity_score: float = 0.0
    complexity_score: float = 0.0
    route_score: float = 0.0
    runner_up_mode: WorkflowMode | None = None
    runner_up_score: float = 0.0
    scheduled_next_mode: WorkflowMode | None = None
    unresolved_questions: list[str] = field(default_factory=list)
    pressure_summary: list[str] = field(default_factory=list)
    signal_summary: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return self.reason_summary

    @classmethod
    def transition(
        cls,
        mode: WorkflowMode,
        *,
        reason_code: str,
        reason_summary: str,
        decision_kind: WorkflowDecisionKind = WorkflowDecisionKind.HANDOFF,
        ambiguity_score: float = 0.0,
        complexity_score: float = 0.0,
        route_score: float = 0.0,
        runner_up_mode: WorkflowMode | None = None,
        runner_up_score: float = 0.0,
        scheduled_next_mode: WorkflowMode | None = None,
        unresolved_questions: list[str] | None = None,
        pressure_summary: list[str] | None = None,
        signal_summary: list[str] | None = None,
    ) -> ModeDecision:
        """Build a non-router workflow decision for handoffs and reentry."""

        return cls(
            mode=mode,
            reason_code=reason_code,
            reason_summary=reason_summary,
            decision_kind=decision_kind,
            ambiguity_score=ambiguity_score,
            complexity_score=complexity_score,
            route_score=route_score,
            runner_up_mode=runner_up_mode,
            runner_up_score=runner_up_score,
            scheduled_next_mode=scheduled_next_mode,
            unresolved_questions=list(unresolved_questions or []),
            pressure_summary=list(pressure_summary or []),
            signal_summary=list(signal_summary or []),
        )

    def with_context(
        self,
        *,
        reason_code: str | None = None,
        reason_summary: str | None = None,
        decision_kind: WorkflowDecisionKind | None = None,
        route_score: float | None = None,
        runner_up_mode: WorkflowMode | None = None,
        runner_up_score: float | None = None,
        scheduled_next_mode: WorkflowMode | None = None,
        unresolved_questions: list[str] | None = None,
        pressure_summary: list[str] | None = None,
        signal_summary: list[str] | None = None,
    ) -> ModeDecision:
        """Return a copy with updated contextual routing metadata."""

        return ModeDecision(
            mode=self.mode,
            reason_code=reason_code or self.reason_code,
            reason_summary=reason_summary or self.reason_summary,
            decision_kind=decision_kind or self.decision_kind,
            ambiguity_score=self.ambiguity_score,
            complexity_score=self.complexity_score,
            route_score=self.route_score if route_score is None else route_score,
            runner_up_mode=self.runner_up_mode if runner_up_mode is None else runner_up_mode,
            runner_up_score=(
                self.runner_up_score if runner_up_score is None else runner_up_score
            ),
            scheduled_next_mode=(
                self.scheduled_next_mode
                if scheduled_next_mode is None
                else scheduled_next_mode
            ),
            unresolved_questions=list(
                self.unresolved_questions
                if unresolved_questions is None
                else unresolved_questions
            ),
            pressure_summary=list(
                self.pressure_summary
                if pressure_summary is None
                else pressure_summary
            ),
            signal_summary=list(
                self.signal_summary if signal_summary is None else signal_summary
            ),
        )


@dataclass(slots=True)
class ClarifyReview:
    """Outcome of one clarify-round review."""

    should_continue: bool
    reason_code: str
    reason_summary: str
    unresolved_questions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ArtifactFreshness:
    """Whether persisted workflow artifacts still fit the current task state."""

    stale_brief: bool = False
    stale_plan: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def requires_refresh(self) -> bool:
        return self.stale_brief or self.stale_plan


@dataclass(slots=True)
class WorkflowTimelineEntry:
    """One persisted workflow history item."""

    timestamp: str
    kind: str
    mode: str
    reason_code: str
    summary: str
    decision_kind: str | None = None
    route_score: float | None = None
    runner_up_mode: str | None = None
    runner_up_score: float | None = None
    scheduled_next_mode: str | None = None
    unresolved_questions: list[str] = field(default_factory=list)
    signal_summary: list[str] = field(default_factory=list)
    prompt_format: str | None = None
    prompt_sections: list[str] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "mode": self.mode,
            "reason_code": self.reason_code,
            "summary": self.summary,
            "decision_kind": self.decision_kind,
            "route_score": self.route_score,
            "runner_up_mode": self.runner_up_mode,
            "runner_up_score": self.runner_up_score,
            "scheduled_next_mode": self.scheduled_next_mode,
            "unresolved_questions": list(self.unresolved_questions),
            "signal_summary": list(self.signal_summary),
            "prompt_format": self.prompt_format,
            "prompt_sections": list(self.prompt_sections),
            "artifact_paths": list(self.artifact_paths),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowTimelineEntry:
        return cls(
            timestamp=str(data.get("timestamp", "")),
            kind=str(data.get("kind", WorkflowTimelineEntryKind.ROUTE.value)),
            mode=str(data.get("mode", WorkflowMode.EXECUTE.value)),
            reason_code=str(data.get("reason_code", "")),
            summary=str(data.get("summary", "")),
            decision_kind=_optional_text(data.get("decision_kind")),
            route_score=_optional_float(data.get("route_score")),
            runner_up_mode=_optional_text(data.get("runner_up_mode")),
            runner_up_score=_optional_float(data.get("runner_up_score")),
            scheduled_next_mode=_optional_text(data.get("scheduled_next_mode")),
            unresolved_questions=_string_list(data.get("unresolved_questions")),
            signal_summary=_string_list(data.get("signal_summary")),
            prompt_format=_optional_text(data.get("prompt_format")),
            prompt_sections=_string_list(data.get("prompt_sections")),
            artifact_paths=_string_list(data.get("artifact_paths")),
        )

    @classmethod
    def from_decision(
        cls,
        decision: ModeDecision,
        *,
        kind: WorkflowTimelineEntryKind,
        prompt_format: str | None = None,
        prompt_sections: list[str] | None = None,
        artifact_paths: list[str] | None = None,
    ) -> WorkflowTimelineEntry:
        summary = f"{decision.mode.value}: {decision.reason_summary}"
        return cls(
            timestamp=_utc_now(),
            kind=kind.value,
            mode=decision.mode.value,
            reason_code=decision.reason_code,
            summary=summary,
            decision_kind=decision.decision_kind.value,
            route_score=decision.route_score,
            runner_up_mode=(
                decision.runner_up_mode.value
                if decision.runner_up_mode is not None
                else None
            ),
            runner_up_score=decision.runner_up_score,
            scheduled_next_mode=(
                decision.scheduled_next_mode.value
                if decision.scheduled_next_mode is not None
                else None
            ),
            unresolved_questions=list(decision.unresolved_questions),
            signal_summary=list(decision.signal_summary),
            prompt_format=prompt_format,
            prompt_sections=list(prompt_sections or []),
            artifact_paths=list(artifact_paths or []),
        )


class WorkflowPolicy:
    """Scored workflow-policy engine for route and clarify decisions."""

    clarify_threshold = 0.55
    plan_threshold = 0.45

    def __init__(self, signal_extractor: WorkflowSignalExtractor | None = None) -> None:
        self.signal_extractor = signal_extractor or WorkflowSignalExtractor()

    def route(
        self,
        task: str,
        *,
        requested_mode: WorkflowMode | None = None,
        has_brief: bool = False,
        has_plan: bool = False,
        allow_clarify: bool = True,
        verification_pressure: bool = False,
        mutating_history: bool = False,
        stale_plan: bool = False,
        unresolved_questions: list[str] | None = None,
        timeline: list[WorkflowTimelineEntry] | None = None,
    ) -> ModeDecision:
        signals = self.signal_extractor.extract_route_signals(
            task,
            requested_mode=requested_mode.value if requested_mode is not None else None,
            has_brief=has_brief,
            has_plan=has_plan,
            allow_clarify=allow_clarify,
            verification_pressure=verification_pressure,
            mutating_history=mutating_history,
            stale_plan=stale_plan,
            unresolved_questions=unresolved_questions,
            timeline=timeline,
        )
        return self.route_from_signals(signals)

    def route_from_signals(self, signals: WorkflowSignalPacket) -> ModeDecision:
        """Route from a typed workflow-signal packet."""

        requested_mode = WorkflowMode.from_str(signals.requested_mode)
        if requested_mode is not None:
            return ModeDecision(
                mode=requested_mode,
                reason_code="explicit_request",
                reason_summary=f"explicit {requested_mode.value} request",
                decision_kind=WorkflowDecisionKind.REQUESTED,
                route_score=1.0,
                scheduled_next_mode=(
                    WorkflowMode.EXECUTE
                    if requested_mode in {WorkflowMode.CLARIFY, WorkflowMode.PLAN}
                    else None
                ),
                signal_summary=list(signals.signal_summary),
            )

        if signals.stale_artifact_pressure > 0:
            return ModeDecision(
                mode=WorkflowMode.PLAN,
                reason_code="stale_plan_artifacts",
                reason_summary="existing plan artifacts no longer match the task state",
                decision_kind=WorkflowDecisionKind.REENTRY,
                route_score=0.95,
                runner_up_mode=WorkflowMode.EXECUTE,
                runner_up_score=0.6,
                scheduled_next_mode=WorkflowMode.EXECUTE,
                unresolved_questions=list(signals.unresolved_questions),
                pressure_summary=[
                    "plan refresh pressure: stale artifacts require a refreshed plan",
                    "execute pressure: continue directly with the stale artifacts",
                ],
                signal_summary=list(signals.signal_summary),
            )

        if signals.has_plan:
            return ModeDecision(
                mode=WorkflowMode.EXECUTE,
                reason_code="existing_plan_artifacts",
                reason_summary="reusing existing plan artifacts",
                decision_kind=WorkflowDecisionKind.ARTIFACT_REUSE,
                route_score=0.9,
                runner_up_mode=WorkflowMode.PLAN,
                runner_up_score=0.45,
                unresolved_questions=list(signals.unresolved_questions),
                pressure_summary=[
                    "execute pressure: persisted plan artifacts already exist",
                    "plan pressure: a plan refresh is available but not required",
                ],
                signal_summary=list(signals.signal_summary),
            )

        ambiguity = signals.ambiguity_score
        complexity = signals.complexity_score

        clarify_pressure = ambiguity
        if signals.allow_clarify and not signals.has_brief:
            clarify_pressure += 0.15
        if signals.unresolved_questions:
            clarify_pressure += min(0.12, 0.04 * len(signals.unresolved_questions))
        if complexity < 0.55:
            clarify_pressure += 0.05
        if signals.recent_clarify_count and signals.unresolved_questions:
            clarify_pressure += 0.04
        if not signals.allow_clarify:
            clarify_pressure = 0.0

        plan_pressure = complexity
        plan_pressure += signals.verification_pressure
        plan_pressure += signals.mutation_pressure
        if signals.has_brief:
            plan_pressure += 0.06
        if signals.unresolved_questions:
            plan_pressure += 0.06
        if signals.recent_reentry_count:
            plan_pressure += 0.06
        if signals.recent_plan_refresh_count:
            plan_pressure += 0.04

        execute_pressure = 0.35
        if signals.has_brief:
            execute_pressure += 0.14
        if ambiguity < 0.35:
            execute_pressure += 0.16
        if complexity < 0.45:
            execute_pressure += 0.12
        if not signals.unresolved_questions:
            execute_pressure += 0.05
        if signals.recent_verify_skip_count and not signals.verification_pressure:
            execute_pressure += 0.03

        scores = {
            WorkflowMode.CLARIFY: round(min(clarify_pressure, 1.0), 3),
            WorkflowMode.PLAN: round(min(plan_pressure, 1.0), 3),
            WorkflowMode.EXECUTE: round(min(execute_pressure, 1.0), 3),
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        winner, winner_score = ordered[0]
        runner_up, runner_up_score = ordered[1]

        pressure_summary = [
            f"{mode.value} pressure={score:.2f}"
            for mode, score in ordered
        ]

        if (
            winner == WorkflowMode.CLARIFY
            and winner_score >= self.clarify_threshold
            and signals.allow_clarify
        ):
            return ModeDecision(
                mode=WorkflowMode.CLARIFY,
                reason_code="task_is_ambiguous",
                reason_summary="workflow pressure favors clarification before execution",
                ambiguity_score=ambiguity,
                complexity_score=complexity,
                route_score=winner_score,
                runner_up_mode=runner_up,
                runner_up_score=runner_up_score,
                scheduled_next_mode=WorkflowMode.EXECUTE,
                unresolved_questions=list(signals.unresolved_questions),
                pressure_summary=pressure_summary,
                signal_summary=list(signals.signal_summary),
            )

        if winner == WorkflowMode.PLAN and winner_score >= self.plan_threshold:
            reason_code = (
                "verification_pressure_requires_plan"
                if signals.verification_pressure
                else "task_is_complex"
            )
            reason_summary = (
                "verification pressure and task complexity favor a persisted plan"
                if signals.verification_pressure
                else "workflow pressure favors a persisted plan before execution"
            )
            return ModeDecision(
                mode=WorkflowMode.PLAN,
                reason_code=reason_code,
                reason_summary=reason_summary,
                ambiguity_score=ambiguity,
                complexity_score=complexity,
                route_score=winner_score,
                runner_up_mode=runner_up,
                runner_up_score=runner_up_score,
                scheduled_next_mode=WorkflowMode.EXECUTE,
                unresolved_questions=list(signals.unresolved_questions),
                pressure_summary=pressure_summary,
                signal_summary=list(signals.signal_summary),
            )

        return ModeDecision(
            mode=WorkflowMode.EXECUTE,
            reason_code="task_is_concrete",
            reason_summary="workflow pressure favors direct execution",
            ambiguity_score=ambiguity,
            complexity_score=complexity,
            route_score=winner_score,
            runner_up_mode=runner_up,
            runner_up_score=runner_up_score,
            unresolved_questions=list(signals.unresolved_questions),
            pressure_summary=pressure_summary,
            signal_summary=list(signals.signal_summary),
        )

    def review_clarify(
        self,
        *,
        task: str,
        answer: str,
        non_goals: list[str],
        round_index: int,
        max_rounds: int,
    ) -> ClarifyReview:
        """Determine whether clarify should continue for another round."""

        unresolved: list[str] = []
        normalized_answer = answer.strip()
        if not normalized_answer:
            unresolved.append("No answer was provided to the clarification question.")
        if len(re.findall(r"\w+", normalized_answer)) < 4:
            unresolved.append("The answer is still too short to lock task boundaries.")
        answer_ambiguity = self._ambiguity_score(f"{task} {normalized_answer}")
        if answer_ambiguity >= 0.5:
            unresolved.append("The clarified scope still uses broad or ambiguous language.")
        if any(
            "anything not confirmed" in item.lower()
            for item in non_goals
        ):
            unresolved.append("Out-of-scope boundaries are still underspecified.")

        if unresolved and round_index < max_rounds:
            return ClarifyReview(
                should_continue=True,
                reason_code="clarify_follow_up_needed",
                reason_summary="clarify pressure remains high after the latest answer",
                unresolved_questions=unresolved,
            )

        if unresolved:
            return ClarifyReview(
                should_continue=False,
                reason_code="clarify_budget_exhausted",
                reason_summary="clarify budget exhausted; carrying unresolved questions forward",
                unresolved_questions=unresolved,
            )

        return ClarifyReview(
            should_continue=False,
            reason_code="clarify_complete",
            reason_summary="clarify gathered enough boundaries to proceed",
            unresolved_questions=[],
        )

    def assess_artifact_freshness(
        self,
        *,
        implementation_text: str | None,
        verification_text: str | None,
        touched_files: list[str],
    ) -> ArtifactFreshness:
        """Detect whether persisted plan artifacts have drifted from execution."""

        if not implementation_text and not verification_text:
            return ArtifactFreshness()

        combined = f"{implementation_text or ''}\n{verification_text or ''}".lower()
        reasons: list[str] = []
        unexpected_paths: list[str] = []
        for file_path in touched_files:
            name = Path(file_path).name.strip()
            if name and name.lower() not in combined:
                unexpected_paths.append(name)

        if unexpected_paths:
            reasons.append(
                "Touched files outside the current plan: "
                + ", ".join(dict.fromkeys(unexpected_paths))
            )

        return ArtifactFreshness(
            stale_plan=bool(reasons),
            reasons=reasons,
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


class ModeRouter(WorkflowPolicy):
    """Backward-compatible alias for the workflow policy router."""


def _has_concrete_anchor(task: str) -> bool:
    return bool(
        re.search(r"[./_\\-]", task)
        or re.search(r"`[^`]+`", task)
        or any(token in task.lower() for token in ("test", "file", "function", "class"))
    )


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
