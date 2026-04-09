"""Typed evidence provenance carried through runtime policy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EvidenceProvenanceStatus(StrEnum):
    """How one evidence item relates to a runtime decision."""

    SUPPORTS = "supports"
    MISSING = "missing"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"


@dataclass(slots=True)
class EvidenceProvenance:
    """One typed piece of evidence behind a completion or verification decision."""

    category: str
    source: str
    summary: str
    status: str = EvidenceProvenanceStatus.CONTEXT.value
    subject: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize one provenance item for persisted runtime state."""

        return {
            "category": self.category,
            "source": self.source,
            "summary": self.summary,
            "status": self.status,
            "subject": self.subject,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceProvenance:
        """Load one persisted provenance item."""

        return cls(
            category=str(data.get("category", "")),
            source=str(data.get("source", "")),
            summary=str(data.get("summary", "")),
            status=str(data.get("status", EvidenceProvenanceStatus.CONTEXT.value)),
            subject=_optional_text(data.get("subject")),
            detail=_optional_text(data.get("detail")),
        )

    def render_summary(self) -> str:
        """Render one concise human-facing summary."""

        return self.summary


def normalize_evidence_provenance(value: Any) -> list[EvidenceProvenance]:
    """Coerce persisted provenance payloads into typed entries."""

    if not isinstance(value, list):
        return []
    entries: list[EvidenceProvenance] = []
    for item in value:
        if isinstance(item, dict):
            entries.append(EvidenceProvenance.from_dict(item))
    return entries


def summarize_evidence_provenance(
    entries: list[EvidenceProvenance],
    *,
    max_items: int | None = None,
) -> list[str]:
    """Project typed provenance into concise evidence-summary strings."""

    summaries: list[str] = []
    limit = len(entries) if max_items is None else max_items
    for entry in entries[:limit]:
        summary = entry.render_summary().strip()
        if summary and summary not in summaries:
            summaries.append(summary)
    return summaries


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
