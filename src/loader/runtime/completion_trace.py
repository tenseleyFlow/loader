"""Persisted completion-policy trace entries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CompletionTraceEntry:
    """One inspectable completion-policy decision."""

    stage: str
    outcome: str
    decision_code: str
    decision_summary: str

    def to_dict(self) -> dict[str, str]:
        """Serialize the entry into persisted session state."""

        return {
            "stage": self.stage,
            "outcome": self.outcome,
            "decision_code": self.decision_code,
            "decision_summary": self.decision_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompletionTraceEntry:
        """Load one persisted completion-policy entry."""

        return cls(
            stage=str(data.get("stage", "")),
            outcome=str(data.get("outcome", "")),
            decision_code=str(data.get("decision_code", "")),
            decision_summary=str(data.get("decision_summary", "")),
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
