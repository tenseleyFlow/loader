"""Tests for intent-aware clarify strategy."""

from __future__ import annotations

from loader.runtime.clarify_strategy import (
    ClarifySlot,
    ClarifySnapshot,
    assess_clarify_snapshot,
    build_clarify_question,
)


def test_assess_clarify_snapshot_prioritizes_touchpoints_for_broad_answer() -> None:
    assessment = assess_clarify_snapshot(
        task="Improve Loader so it feels more like claw-code.",
        answer="Make it nicer.",
        snapshot=ClarifySnapshot(
            task_statement="Improve Loader so it feels more like claw-code.",
            explicit_sections=[],
        ),
    )

    assert assessment.unresolved_slots
    assert assessment.focus_slot == ClarifySlot.LIKELY_TOUCHPOINTS
    assert "Out-of-scope boundaries are still underspecified." in assessment.unresolved_questions


def test_build_clarify_question_targets_requested_slot() -> None:
    question = build_clarify_question(
        "Tighten the runtime behavior.",
        ClarifySlot.NON_GOALS,
    )

    assert "out of scope" in question.lower()

