"""Tests for the Sprint 10 workflow policy and timeline core."""

from __future__ import annotations

from loader.runtime.workflow import (
    WorkflowMode,
    WorkflowPolicy,
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
)


def test_workflow_policy_reports_winner_and_runner_up() -> None:
    policy = WorkflowPolicy()

    decision = policy.route("Improve Loader so it feels more like claw-code.")

    assert decision.mode == WorkflowMode.CLARIFY
    assert decision.route_score >= policy.clarify_threshold
    assert decision.runner_up_mode is not None
    assert decision.runner_up_score > 0
    assert decision.pressure_summary


def test_workflow_policy_prefers_plan_refresh_for_stale_plan() -> None:
    policy = WorkflowPolicy()

    decision = policy.route(
        "Keep working on the runtime task.",
        has_plan=True,
        stale_plan=True,
    )

    assert decision.mode == WorkflowMode.PLAN
    assert decision.reason_code == "stale_plan_artifacts"
    assert decision.decision_kind == "reentry"
    assert decision.scheduled_next_mode == WorkflowMode.EXECUTE


def test_workflow_policy_marks_unplanned_touched_files_as_stale() -> None:
    policy = WorkflowPolicy()

    freshness = policy.assess_artifact_freshness(
        implementation_text="# Implementation Plan\n- Update loader.py only\n",
        verification_text="# Verification Plan\n- Run pytest\n",
        touched_files=["/tmp/loader.py", "/tmp/unplanned.py"],
    )

    assert freshness.stale_plan is True
    assert "unplanned.py" in freshness.reasons[0]


def test_workflow_timeline_entry_round_trips() -> None:
    entry = WorkflowTimelineEntry(
        timestamp="2026-04-07T12:00:00Z",
        kind=WorkflowTimelineEntryKind.ROUTE.value,
        mode=WorkflowMode.PLAN.value,
        reason_code="task_is_complex",
        summary="plan: workflow pressure favors a persisted plan before execution",
        decision_kind="initial_route",
        route_score=0.81,
        runner_up_mode="clarify",
        runner_up_score=0.66,
        scheduled_next_mode="execute",
        unresolved_questions=["Scope is still broad."],
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context"],
        artifact_paths=["/tmp/implementation.md"],
    )

    restored = WorkflowTimelineEntry.from_dict(entry.to_dict())

    assert restored == entry
