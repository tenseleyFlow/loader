"""Tests for typed evidence provenance on policy timelines and traces."""

from __future__ import annotations

from loader.runtime.completion_trace import completion_trace_from_workflow_timeline
from loader.runtime.evidence_provenance import (
    EvidenceProvenance,
    EvidenceProvenanceStatus,
)
from loader.runtime.workflow_policy import WorkflowTimelineEntry, WorkflowTimelineEntryKind


def test_workflow_timeline_entry_derives_evidence_summary_from_provenance() -> None:
    entry = WorkflowTimelineEntry.accountability(
        kind=WorkflowTimelineEntryKind.COMPLETION_FINALIZE,
        mode="execute",
        reason_code="continuation_budget_exhausted",
        summary="completion: stopped because follow-through evidence was still missing",
        policy_stage="continuation_check",
        policy_outcome="finalize",
        evidence_provenance=[
            EvidenceProvenance(
                category="verification",
                source="dod.evidence",
                summary="verification evidence was still missing for `pytest -q`",
                status=EvidenceProvenanceStatus.MISSING.value,
                subject="pytest -q",
            )
        ],
    )

    assert entry.evidence_summary == [
        "verification evidence was still missing for `pytest -q`"
    ]
    assert entry.evidence_provenance[0].status == EvidenceProvenanceStatus.MISSING.value


def test_completion_trace_projection_preserves_evidence_provenance() -> None:
    timeline = [
        WorkflowTimelineEntry.accountability(
            kind=WorkflowTimelineEntryKind.COMPLETION_FINALIZE,
            mode="execute",
            reason_code="continuation_budget_exhausted",
            summary="completion: stopped because follow-through evidence was still missing",
            policy_stage="continuation_check",
            policy_outcome="finalize",
            evidence_provenance=[
                EvidenceProvenance(
                    category="verification",
                    source="dod.evidence",
                    summary="verification evidence was still missing for `pytest -q`",
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject="pytest -q",
                ),
                EvidenceProvenance(
                    category="action",
                    source="actions_taken",
                    summary="recorded work already showed the requested edit happened",
                    status=EvidenceProvenanceStatus.SUPPORTS.value,
                ),
            ],
        )
    ]

    trace = completion_trace_from_workflow_timeline(
        timeline,
        last_decision_code="continuation_budget_exhausted",
    )

    assert len(trace) == 1
    assert trace[0].decision_code == "continuation_budget_exhausted"
    assert trace[0].evidence_summary == [
        "verification evidence was still missing for `pytest -q`",
        "recorded work already showed the requested edit happened",
    ]
    assert [item.status for item in trace[0].evidence_provenance] == [
        EvidenceProvenanceStatus.MISSING.value,
        EvidenceProvenanceStatus.SUPPORTS.value,
    ]
