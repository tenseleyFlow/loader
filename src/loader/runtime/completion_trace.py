"""Completion-policy trace entries and timeline projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .workflow_policy import WorkflowTimelineEntry


@dataclass(slots=True)
class CompletionTraceEntry:
    """One inspectable completion-policy decision."""

    stage: str
    outcome: str
    decision_code: str
    decision_summary: str
    evidence_summary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, str]:
        """Serialize the entry into persisted session state."""

        return {
            "stage": self.stage,
            "outcome": self.outcome,
            "decision_code": self.decision_code,
            "decision_summary": self.decision_summary,
            "evidence_summary": list(self.evidence_summary),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionTraceEntry:
        """Load one persisted completion-policy entry."""

        return cls(
            stage=str(data.get("stage", "")),
            outcome=str(data.get("outcome", "")),
            decision_code=str(data.get("decision_code", "")),
            decision_summary=str(data.get("decision_summary", "")),
            evidence_summary=[
                str(item)
                for item in data.get("evidence_summary", [])
                if str(item).strip()
            ],
        )


def normalize_completion_trace(value: Any) -> list[CompletionTraceEntry]:
    """Coerce persisted completion traces into typed entries."""

    if not isinstance(value, list):
        return []
    entries: list[CompletionTraceEntry] = []
    for item in value:
        if isinstance(item, dict):
            entries.append(CompletionTraceEntry.from_dict(item))
    return entries


def completion_trace_from_workflow_timeline(
    timeline: list[WorkflowTimelineEntry],
    *,
    last_decision_code: str | None = None,
    fallback: list[CompletionTraceEntry] | None = None,
    max_entries: int = 8,
) -> list[CompletionTraceEntry]:
    """Project the latest completion trace from canonical workflow timeline entries."""

    if not last_decision_code:
        return list(fallback or [])

    end_index: int | None = None
    for index in range(len(timeline) - 1, -1, -1):
        entry = timeline[index]
        if _is_completion_timeline_entry(entry) and entry.reason_code == last_decision_code:
            end_index = index
            break
    if end_index is None:
        return list(fallback or [])

    start_index = end_index
    while start_index > 0 and _is_completion_timeline_entry(timeline[start_index - 1]):
        start_index -= 1

    return [
        _completion_trace_entry_from_timeline_entry(entry)
        for entry in timeline[start_index : end_index + 1]
        if _is_completion_timeline_entry(entry)
    ][-max_entries:]


def has_canonical_completion_trace(
    timeline: list[WorkflowTimelineEntry],
    *,
    last_decision_code: str | None = None,
) -> bool:
    """Return whether the workflow timeline already carries the latest completion trace."""

    return bool(
        completion_trace_from_workflow_timeline(
            timeline,
            last_decision_code=last_decision_code,
        )
    )


def _is_completion_timeline_entry(entry: WorkflowTimelineEntry) -> bool:
    return str(entry.kind).startswith("completion_")


def _completion_trace_entry_from_timeline_entry(
    entry: WorkflowTimelineEntry,
) -> CompletionTraceEntry:
    summary = str(entry.summary)
    if summary.startswith("completion:"):
        summary = summary.split(":", 1)[1].strip()
    return CompletionTraceEntry(
        stage=entry.policy_stage or "unknown",
        outcome=entry.policy_outcome or _completion_outcome_from_kind(entry.kind),
        decision_code=entry.reason_code,
        decision_summary=summary,
        evidence_summary=list(entry.evidence_summary),
    )


def _completion_outcome_from_kind(kind: str) -> str:
    if kind == "completion_continue":
        return "continue"
    if kind == "completion_finalize":
        return "finalize"
    if kind == "completion_check":
        return "accept"
    return "complete"
