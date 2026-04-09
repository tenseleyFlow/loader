"""Tests for shared workflow timeline read models."""

from __future__ import annotations

from loader.runtime.evidence_provenance import EvidenceProvenance
from loader.runtime.verification_observations import VerificationObservation
from loader.runtime.workflow_ledger import WorkflowLedger, WorkflowLedgerItem
from loader.runtime.workflow_policy import WorkflowTimelineEntry
from loader.runtime.workflow_timeline_read_model import project_workflow_timeline


def test_project_workflow_timeline_builds_policy_views_and_highlights() -> None:
    entries = [
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:00:00Z",
            kind="repair_retry",
            mode="execute",
            reason_code="raw_text_tool_recovered",
            summary="repair: recovered raw-text tool calls into executable tool invocations",
            decision_kind="forced",
            policy_stage="raw_text_tool_fallback",
            policy_outcome="retry",
        ),
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:01:00Z",
            kind="completion_continue",
            mode="execute",
            reason_code="verification_failed_reentry",
            summary=(
                "completion: continued after verification failed and the runtime "
                "re-entered execute mode"
            ),
            decision_kind="forced",
            policy_stage="definition_of_done",
            policy_outcome="continue",
            evidence_provenance=[
                EvidenceProvenance(
                    category="verification",
                    source="dod.evidence",
                    summary="verification failed for `pytest -q`",
                    status="contradicts",
                    subject="pytest -q",
                )
            ],
            verification_observations=[
                VerificationObservation(
                    status="failed",
                    summary="verification failed for `pytest -q`",
                    command="pytest -q",
                    kind="test",
                    detail="1 failed",
                )
            ],
        ),
    ]
    ledger = WorkflowLedger(
        assumptions=[
            WorkflowLedgerItem(
                text="notes.txt stays out of scope.",
                status="contradicted",
                introduced_phase="clarify",
                updated_phase="recovery",
                evidence=["Execution already touched notes.txt."],
            )
        ]
    )

    projection = project_workflow_timeline(entries, workflow_ledger=ledger)

    assert projection.total_entries == 2
    assert [entry.kind for entry in projection.policy_entries] == [
        "repair_retry",
        "completion_continue",
    ]
    assert projection.latest_policy_summary is not None
    assert "verification_failed_reentry" in projection.latest_policy_summary
    assert "provenance=contradicts:verification@dod.evidence(pytest -q)" in (
        projection.latest_policy_summary
    )
    assert "observed=verification failed for `pytest -q` [1 failed]" in (
        projection.latest_policy_summary
    )
    assert projection.latest_policy_evidence is not None
    assert projection.latest_policy_evidence.blocking == [
        "verification failed for `pytest -q`"
    ]
    assert projection.latest_policy_observed_verification == [
        "verification failed for `pytest -q` [1 failed]"
    ]
    assert any(item.startswith("Repair path:") for item in projection.highlights)
    assert any(item.startswith("Completion decision:") for item in projection.highlights)
    assert any(item.startswith("Contradicted assumptions:") for item in projection.highlights)


def test_project_workflow_timeline_treats_verify_observation_as_accountability() -> None:
    entries = [
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:03:00Z",
            kind="verify_observation",
            mode="verify",
            reason_code="verification_command_failed",
            summary="verify: verification failed for `pytest -q`",
            decision_kind="forced",
            policy_stage="verification",
            policy_outcome="failed",
            verification_observations=[
                VerificationObservation(
                    status="failed",
                    summary="verification failed for `pytest -q`",
                    command="pytest -q",
                    kind="test",
                    detail="1 failed",
                )
            ],
        )
    ]

    projection = project_workflow_timeline(entries, accountability_only=True)

    assert [entry.kind for entry in projection.policy_entries] == ["verify_observation"]
    assert [entry.kind for entry in projection.entries] == ["verify_observation"]
    assert projection.latest_policy_summary is not None
    assert "policy-stage=verification" in projection.latest_policy_summary
    assert "observed=verification failed for `pytest -q` [1 failed]" in (
        projection.latest_policy_summary
    )
    assert any(item.startswith("Verify observed:") for item in projection.highlights)


def test_project_workflow_timeline_highlights_pending_verification() -> None:
    entries = [
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:03:00Z",
            kind="verify_observation",
            mode="verify",
            reason_code="verification_pending",
            summary="verify: verification is pending for the active command set",
            decision_kind="forced",
            policy_stage="verification",
            policy_outcome="pending",
            verification_observations=[
                VerificationObservation(
                    status="pending",
                    summary="verification pending for `pytest -q`",
                    command="pytest -q",
                    kind="test",
                )
            ],
        )
    ]

    projection = project_workflow_timeline(entries, accountability_only=True)

    assert projection.latest_policy_summary is not None
    assert "policy-outcome=pending" in projection.latest_policy_summary
    assert "observed=verification pending for `pytest -q`" in (
        projection.latest_policy_summary
    )
    assert any(item.startswith("Verify pending:") for item in projection.highlights)


def test_project_workflow_timeline_highlights_stale_verification() -> None:
    entries = [
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:04:00Z",
            kind="verify_observation",
            mode="execute",
            reason_code="verification_stale",
            summary="verify: previous verification became stale after new mutating work",
            decision_kind="forced",
            policy_stage="verification",
            policy_outcome="stale",
            verification_observations=[
                VerificationObservation(
                    status="stale",
                    summary=(
                        "verification became stale for `pytest -q` after new mutating work"
                    ),
                    command="pytest -q",
                    kind="runtime",
                    detail="write changed README.md",
                )
            ],
        )
    ]

    projection = project_workflow_timeline(entries, accountability_only=True)

    assert projection.latest_policy_summary is not None
    assert "policy-outcome=stale" in projection.latest_policy_summary
    assert "observed=verification became stale for `pytest -q` after new mutating work [write changed README.md]" in (
        projection.latest_policy_summary
    )
    assert any(item.startswith("Verify stale:") for item in projection.highlights)


def test_project_workflow_timeline_applies_policy_filters_and_limits() -> None:
    entries = [
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:00:00Z",
            kind="handoff",
            mode="plan",
            reason_code="task_is_complex",
            summary="plan: workflow pressure favors a persisted plan before execution",
            decision_kind="handoff",
        ),
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:01:00Z",
            kind="repair_retry",
            mode="execute",
            reason_code="raw_text_tool_recovered",
            summary="repair: recovered raw-text tool calls into executable tool invocations",
            decision_kind="forced",
            policy_stage="raw_text_tool_fallback",
            policy_outcome="retry",
        ),
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:02:00Z",
            kind="completion_check",
            mode="execute",
            reason_code="completion_response_accepted",
            summary=(
                "completion: accepted the response because completion heuristics "
                "found no missing follow-through"
            ),
            decision_kind="forced",
            policy_stage="continuation_check",
            policy_outcome="accept",
        ),
    ]

    projection = project_workflow_timeline(
        entries,
        accountability_only=True,
        mode="execute",
        limit=1,
    )

    assert projection.total_entries == 3
    assert [entry.kind for entry in projection.entries] == ["completion_check"]
    assert [entry.kind for entry in projection.policy_entries] == [
        "repair_retry",
        "completion_check",
    ]


def test_project_workflow_timeline_rolls_up_supporting_and_missing_evidence() -> None:
    entries = [
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:02:00Z",
            kind="completion_finalize",
            mode="verify",
            reason_code="verification_budget_exhausted",
            summary="completion: stopped because verification evidence was still missing",
            decision_kind="forced",
            policy_stage="definition_of_done",
            policy_outcome="finalize",
            evidence_provenance=[
                EvidenceProvenance(
                    category="verification",
                    source="dod.evidence",
                    summary="verification evidence was still missing for `pytest -q`",
                    status="missing",
                    subject="pytest -q",
                ),
                EvidenceProvenance(
                    category="tracked_work",
                    source="dod.pending_items",
                    summary="all tracked work items except verification were complete",
                    status="supports",
                ),
            ],
        )
    ]

    projection = project_workflow_timeline(entries)

    assert projection.latest_policy_evidence is not None
    assert projection.latest_policy_evidence.missing == [
        "verification evidence was still missing for `pytest -q`"
    ]
    assert projection.latest_policy_evidence.supporting == [
        "all tracked work items except verification were complete"
    ]
