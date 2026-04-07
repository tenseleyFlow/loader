"""Tests for semantic artifact invalidation and recovery selection."""

from __future__ import annotations

from loader.runtime.artifact_invalidation import (
    ArtifactInvalidationAssessor,
    WorkflowRecoveryStrategy,
)
from loader.runtime.workflow import ArtifactEvidenceKind


def test_artifact_invalidation_requests_plan_refresh_for_plan_only_drift() -> None:
    assessor = ArtifactInvalidationAssessor()

    freshness = assessor.assess(
        task_statement="Implement the runtime report artifact.",
        clarify_text=None,
        implementation_text="# Implementation Plan\n- Create report.md only\n",
        verification_text="# Verification Plan\n- report.md exists\n",
        acceptance_criteria=["report.md exists"],
        touched_files=["/tmp/notes.md"],
        last_verification_result=None,
    )

    assert freshness.stale_plan is True
    assert freshness.stale_brief is False
    assert freshness.recovery_strategy == WorkflowRecoveryStrategy.PLAN_REFRESH.value
    assert "touched_files_outside_plan" in freshness.reason_codes
    assert any(
        item.kind == ArtifactEvidenceKind.CONFIRMED_TOUCHPOINT.value
        and "notes.md" in item.summary
        for item in freshness.evidence
    )
    assert any(
        item.kind == ArtifactEvidenceKind.ACCEPTANCE_ANCHOR.value
        and "report.md exists" in item.summary
        for item in freshness.evidence
    )


def test_artifact_invalidation_can_force_full_replan_when_brief_and_plan_drift() -> None:
    assessor = ArtifactInvalidationAssessor()

    freshness = assessor.assess(
        task_statement="Improve Loader runtime workflow discipline.",
        clarify_text="# Task Brief\n\n## Desired Outcome\n- Improve Loader runtime workflow.\n",
        implementation_text="# Implementation Plan\n- Touch planned.txt only\n",
        verification_text="# Verification Plan\n## Acceptance Criteria\n- planned.txt exists.\n",
        acceptance_criteria=["notes.txt exists in the workspace root."],
        touched_files=["/tmp/notes.txt"],
        last_verification_result="failed",
    )

    assert freshness.stale_plan is True
    assert freshness.stale_brief is True
    assert freshness.recovery_strategy == WorkflowRecoveryStrategy.FULL_REPLAN.value
    assert "touchpoints_outside_brief" in freshness.reason_codes
    assert "acceptance_criteria_outside_plan" in freshness.reason_codes
    assert any(
        item.kind == ArtifactEvidenceKind.VERIFICATION_CONTRADICTION.value
        and "notes.txt exists in the workspace root." in item.summary
        for item in freshness.evidence
    )
    assert any(
        item.kind == ArtifactEvidenceKind.CONTRADICTED_ASSUMPTION.value
        and "notes.txt" in item.summary
        for item in freshness.evidence
    )
    assert freshness.evidence_summary
