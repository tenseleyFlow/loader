"""Durable workflow ledger state for assumptions, anchors, and boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .workflow import ClarifyBrief
    from .workflow_policy import ArtifactFreshness


@dataclass(slots=True)
class WorkflowLedgerItem:
    """One durable workflow-ledger item."""

    text: str
    status: str
    introduced_phase: str
    updated_phase: str | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status,
            "introduced_phase": self.introduced_phase,
            "updated_phase": self.updated_phase,
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowLedgerItem:
        return cls(
            text=str(data.get("text", "")).strip(),
            status=str(data.get("status", "open")).strip() or "open",
            introduced_phase=str(data.get("introduced_phase", "unknown")).strip()
            or "unknown",
            updated_phase=_optional_text(data.get("updated_phase")),
            evidence=[str(item).strip() for item in data.get("evidence", []) if str(item).strip()],
        )

    def with_evidence(self, summary: str, *, phase: str, status: str | None = None) -> None:
        """Update one item with fresh evidence."""

        cleaned = summary.strip()
        if cleaned and cleaned not in self.evidence:
            self.evidence.append(cleaned)
        if status is not None:
            self.status = status
        self.updated_phase = phase


@dataclass(slots=True)
class WorkflowLedger:
    """Persisted semantic workflow state used for inspection and recovery."""

    assumptions: list[WorkflowLedgerItem] = field(default_factory=list)
    acceptance_anchors: list[WorkflowLedgerItem] = field(default_factory=list)
    decision_boundaries: list[WorkflowLedgerItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumptions": [item.to_dict() for item in self.assumptions],
            "acceptance_anchors": [item.to_dict() for item in self.acceptance_anchors],
            "decision_boundaries": [item.to_dict() for item in self.decision_boundaries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowLedger:
        return cls(
            assumptions=_items_from_dict(data.get("assumptions")),
            acceptance_anchors=_items_from_dict(data.get("acceptance_anchors")),
            decision_boundaries=_items_from_dict(data.get("decision_boundaries")),
        )

    def copy(self) -> WorkflowLedger:
        """Return a detached copy safe for mutation."""

        return WorkflowLedger.from_dict(self.to_dict())

    def has_items(self) -> bool:
        """Return whether any semantic ledger state exists."""

        return bool(
            self.assumptions
            or self.acceptance_anchors
            or self.decision_boundaries
        )


def seed_workflow_ledger_from_brief(
    ledger: WorkflowLedger,
    brief: ClarifyBrief,
    *,
    phase: str = "clarify",
) -> WorkflowLedger:
    """Merge clarify-brief semantics into the durable workflow ledger."""

    next_ledger = ledger.copy()
    _merge_text_items(
        next_ledger.assumptions,
        brief.assumptions,
        status="open",
        phase=phase,
    )
    _merge_text_items(
        next_ledger.acceptance_anchors,
        brief.acceptance_criteria,
        status="active",
        phase=phase,
    )
    _merge_text_items(
        next_ledger.decision_boundaries,
        brief.decision_boundaries,
        status="tracked",
        phase=phase,
    )
    return next_ledger


def seed_workflow_ledger_from_acceptance_criteria(
    ledger: WorkflowLedger,
    acceptance_criteria: list[str],
    *,
    phase: str = "plan",
) -> WorkflowLedger:
    """Merge acceptance anchors discovered during planning or verification."""

    next_ledger = ledger.copy()
    _merge_text_items(
        next_ledger.acceptance_anchors,
        acceptance_criteria,
        status="active",
        phase=phase,
    )
    return next_ledger


def apply_freshness_to_workflow_ledger(
    ledger: WorkflowLedger,
    freshness: ArtifactFreshness,
    *,
    phase: str = "recovery",
) -> WorkflowLedger:
    """Apply drift evidence to the durable workflow ledger."""

    next_ledger = ledger.copy()
    for evidence in freshness.evidence:
        summary = evidence.summary.strip()
        if not summary:
            continue

        if evidence.kind == "contradicted_assumption":
            item = _find_best_match(next_ledger.assumptions, summary)
            if item is None:
                item = WorkflowLedgerItem(
                    text=_extract_focus_text(summary),
                    status="contradicted",
                    introduced_phase=phase,
                )
                next_ledger.assumptions.append(item)
            item.with_evidence(summary, phase=phase, status="contradicted")
            continue

        if evidence.kind in {"acceptance_anchor", "verification_contradiction"}:
            item = _find_best_match(next_ledger.acceptance_anchors, summary)
            if item is None:
                item = WorkflowLedgerItem(
                    text=_extract_focus_text(summary),
                    status="changed",
                    introduced_phase=phase,
                )
                next_ledger.acceptance_anchors.append(item)
            item.with_evidence(summary, phase=phase, status="changed")
            continue

        if evidence.kind == "task_boundary_change":
            item = _find_best_match(next_ledger.decision_boundaries, summary)
            if item is None:
                item = WorkflowLedgerItem(
                    text=_extract_focus_text(summary),
                    status="reopened",
                    introduced_phase=phase,
                )
                next_ledger.decision_boundaries.append(item)
            item.with_evidence(summary, phase=phase, status="reopened")

    return next_ledger


def workflow_ledger_highlights(ledger: WorkflowLedger) -> list[str]:
    """Return concise operator-facing highlights for one ledger."""

    highlights: list[str] = []
    contradicted = [item.text for item in ledger.assumptions if item.status == "contradicted"]
    changed_anchors = [
        item.text for item in ledger.acceptance_anchors if item.status == "changed"
    ]
    reopened = [
        item.text for item in ledger.decision_boundaries if item.status == "reopened"
    ]
    if contradicted:
        highlights.append(f"Contradicted assumptions: {', '.join(contradicted[:2])}")
    if changed_anchors:
        highlights.append(f"Changed acceptance anchors: {', '.join(changed_anchors[:2])}")
    if reopened:
        highlights.append(f"Reopened boundaries: {', '.join(reopened[:2])}")
    return highlights


def _items_from_dict(value: Any) -> list[WorkflowLedgerItem]:
    if not isinstance(value, list):
        return []
    items: list[WorkflowLedgerItem] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        item = WorkflowLedgerItem.from_dict(raw)
        if item.text:
            items.append(item)
    return items


def _merge_text_items(
    items: list[WorkflowLedgerItem],
    values: list[str],
    *,
    status: str,
    phase: str,
) -> None:
    for text in values:
        normalized = _normalized_text(text)
        if not normalized:
            continue
        existing = _find_exact_match(items, normalized)
        if existing is not None:
            existing.updated_phase = phase
            continue
        items.append(
            WorkflowLedgerItem(
                text=text.strip(),
                status=status,
                introduced_phase=phase,
            )
        )


def _find_exact_match(
    items: list[WorkflowLedgerItem],
    normalized_text: str,
) -> WorkflowLedgerItem | None:
    for item in items:
        if _normalized_text(item.text) == normalized_text:
            return item
    return None


def _find_best_match(
    items: list[WorkflowLedgerItem],
    summary: str,
) -> WorkflowLedgerItem | None:
    normalized_summary = _normalized_text(summary)
    summary_tokens = _semantic_tokens(summary)
    best_item: WorkflowLedgerItem | None = None
    best_score = 0

    for item in items:
        normalized_item = _normalized_text(item.text)
        if not normalized_item:
            continue
        if normalized_item in normalized_summary or normalized_summary in normalized_item:
            return item

        overlap = len(_semantic_tokens(item.text) & summary_tokens)
        if overlap > best_score:
            best_item = item
            best_score = overlap

    if best_score >= 2:
        return best_item
    return None


def _extract_focus_text(summary: str) -> str:
    quoted = re.findall(r"`([^`]+)`", summary)
    if quoted:
        return quoted[0].strip()
    shortened = " ".join(summary.split()).strip()
    if len(shortened) <= 96:
        return shortened
    return shortened[:93].rstrip() + "..."


def _normalized_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(re.sub(r"[`*_]+", "", str(value)).lower().split()).strip()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _semantic_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_./-]+", _normalized_text(text))
        if len(token) > 2 and token not in _STOP_WORDS
    }


_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "before",
    "after",
    "current",
    "still",
    "brief",
    "plan",
    "task",
    "scope",
    "exists",
    "runtime",
}
