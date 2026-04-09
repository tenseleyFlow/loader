"""Tests for typed verification observations on policy timelines and traces."""

from __future__ import annotations

from loader.runtime.completion_trace import (
    CompletionTraceEntry,
    completion_trace_from_workflow_timeline,
)
from loader.runtime.verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
    normalize_verification_observations,
)
from loader.runtime.workflow_policy import (
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
)


def test_normalize_verification_observations_round_trips_entries() -> None:
    entries = normalize_verification_observations(
        [
            {
                "status": "PASSED",
                "summary": "verification passed for `uv run pytest -q`",
                "command": "uv run pytest -q",
                "kind": "test",
                "exit_code": 0,
                "detail": "219 passed",
            }
        ]
    )

    assert entries == [
        VerificationObservation(
            status=VerificationObservationStatus.PASSED.value,
            summary="verification passed for `uv run pytest -q`",
            command="uv run pytest -q",
            kind="test",
            exit_code=0,
            detail="219 passed",
        )
    ]


def test_workflow_timeline_entry_derives_evidence_summary_from_verification_observations() -> None:
    entry = WorkflowTimelineEntry.accountability(
        kind=WorkflowTimelineEntryKind.VERIFY_SKIP,
        mode="execute",
        reason_code="verification_not_required",
        summary="verification skipped because the turn made no mutating changes",
        verification_observations=[
            VerificationObservation(
                status=VerificationObservationStatus.SKIPPED.value,
                summary="verification was skipped because no mutating work required checks",
            )
        ],
    )

    assert entry.evidence_summary == [
        "verification was skipped because no mutating work required checks"
    ]


def test_completion_trace_projection_preserves_verification_observations() -> None:
    timeline = [
        WorkflowTimelineEntry.accountability(
            kind=WorkflowTimelineEntryKind.COMPLETION_COMPLETE,
            mode="execute",
            reason_code="verification_passed",
            summary="completion: accepted the response after verification evidence passed",
            policy_stage="definition_of_done",
            policy_outcome="complete",
            verification_observations=[
                VerificationObservation(
                    status=VerificationObservationStatus.PASSED.value,
                    summary="verification passed for `uv run pytest -q`",
                    command="uv run pytest -q",
                    kind="test",
                    exit_code=0,
                    detail="219 passed",
                )
            ],
        )
    ]

    trace = completion_trace_from_workflow_timeline(
        timeline,
        last_decision_code="verification_passed",
    )

    assert trace == [
        CompletionTraceEntry(
            stage="definition_of_done",
            outcome="complete",
            decision_code="verification_passed",
            decision_summary="accepted the response after verification evidence passed",
            evidence_summary=["verification passed for `uv run pytest -q`"],
            verification_observations=[
                VerificationObservation(
                    status=VerificationObservationStatus.PASSED.value,
                    summary="verification passed for `uv run pytest -q`",
                    command="uv run pytest -q",
                    kind="test",
                    exit_code=0,
                    detail="219 passed",
                )
            ],
        )
    ]
