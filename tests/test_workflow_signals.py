"""Tests for typed workflow-signal extraction."""

from __future__ import annotations

from loader.runtime.workflow import (
    WorkflowSignalExtractor,
    WorkflowTimelineEntry,
)


def test_workflow_signal_extractor_captures_recent_timeline_pressure() -> None:
    extractor = WorkflowSignalExtractor()
    timeline = [
        WorkflowTimelineEntry(
            timestamp="2026-04-07T12:00:00Z",
            kind="clarify_continue",
            mode="clarify",
            reason_code="clarify_follow_up_needed",
            summary="clarify: clarify pressure remains high",
            decision_kind="forced",
        ),
        WorkflowTimelineEntry(
            timestamp="2026-04-07T12:01:00Z",
            kind="reentry",
            mode="execute",
            reason_code="verification_failed_reentry",
            summary="execute: verification failed; returning to execute",
            decision_kind="reentry",
        ),
        WorkflowTimelineEntry(
            timestamp="2026-04-07T12:02:00Z",
            kind="verify_skip",
            mode="verify",
            reason_code="verification_not_required",
            summary="verify: verification skipped",
            decision_kind="forced",
        ),
    ]

    signals = extractor.extract_route_signals(
        "Improve Loader so it feels more like claw-code.",
        has_brief=True,
        unresolved_questions=["Scope is still broad."],
        timeline=timeline,
    )

    assert signals.ambiguity_score > 0
    assert signals.has_brief is True
    assert signals.recent_clarify_count == 1
    assert signals.recent_reentry_count == 1
    assert signals.recent_verify_skip_count == 1
    assert "clarify_brief=available" in signals.signal_summary
    assert "open_questions=1" in signals.signal_summary
    assert "recent_reentry=1" in signals.signal_summary

