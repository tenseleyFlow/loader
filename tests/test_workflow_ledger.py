"""Tests for durable workflow-ledger semantics."""

from __future__ import annotations

from loader.runtime.workflow import ArtifactEvidence, ArtifactFreshness, ClarifyBrief
from loader.runtime.workflow_ledger import (
    WorkflowLedger,
    apply_freshness_to_workflow_ledger,
    seed_workflow_ledger_from_acceptance_criteria,
    seed_workflow_ledger_from_brief,
    workflow_ledger_highlights,
)


def test_workflow_ledger_seeds_from_brief_and_tracks_drift() -> None:
    brief = ClarifyBrief(
        task_statement="Keep the runtime artifact aligned with the actual work.",
        assumptions=["notes.txt stays out of scope unless clarified otherwise."],
        acceptance_criteria=["planned.txt exists in the workspace root."],
        decision_boundaries=["Escalate before broad UX changes."],
    )
    brief.fill_defaults()

    ledger = seed_workflow_ledger_from_brief(WorkflowLedger(), brief)
    ledger = seed_workflow_ledger_from_acceptance_criteria(
        ledger,
        ["planned.txt exists in the workspace root."],
        phase="plan",
    )
    freshness = ArtifactFreshness(
        evidence=[
            ArtifactEvidence(
                kind="contradicted_assumption",
                summary="Clarify scope assumed `notes.txt` stayed out of scope.",
            ),
            ArtifactEvidence(
                kind="verification_contradiction",
                summary=(
                    "Failed verification exposed missing brief coverage for "
                    "`notes.txt exists in the workspace root.`."
                ),
            ),
            ArtifactEvidence(
                kind="task_boundary_change",
                summary="The active task framing outgrew the persisted clarify brief.",
            ),
        ]
    )

    updated = apply_freshness_to_workflow_ledger(ledger, freshness)

    assert any(
        item.text == "notes.txt stays out of scope unless clarified otherwise."
        and item.status == "contradicted"
        for item in updated.assumptions
    )
    assert any(
        "notes.txt exists in the workspace root." in item.text
        and item.status == "changed"
        for item in updated.acceptance_anchors
    )
    assert any(item.status == "reopened" for item in updated.decision_boundaries)
    assert workflow_ledger_highlights(updated)
