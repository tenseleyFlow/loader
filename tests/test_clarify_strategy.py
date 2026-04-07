"""Tests for intent-aware clarify strategy."""

from __future__ import annotations

from loader.runtime.clarify_strategy import (
    ClarifyPressureKind,
    ClarifySlot,
    ClarifySnapshot,
    ClarifyStage,
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


def test_assess_clarify_snapshot_requests_tradeoff_pressure_pass_on_later_round() -> None:
    assessment = assess_clarify_snapshot(
        task="Improve Loader runtime behavior.",
        answer="Focus on src/loader/runtime/conversation.py.",
        snapshot=ClarifySnapshot(
            task_statement="Improve Loader runtime behavior.",
            explicit_sections=["desired_outcome", "likely_touchpoints"],
            desired_outcome=["Make the runtime flow more disciplined."],
            likely_touchpoints=["src/loader/runtime/conversation.py"],
        ),
        round_index=2,
    )

    assert assessment.stage == ClarifyStage.READINESS
    assert assessment.pressure_kind == ClarifyPressureKind.TRADEOFF
    assert assessment.pressure_pass_complete is False
    assert "non_goals" in assessment.missing_readiness_gates
    assert "decision_boundaries" in assessment.missing_readiness_gates


def test_assess_clarify_snapshot_marks_pressure_pass_complete_for_boundary_answer() -> None:
    assessment = assess_clarify_snapshot(
        task="Improve Loader runtime behavior.",
        answer="Keep the CLI unchanged and do not broaden the UX without confirming first.",
        snapshot=ClarifySnapshot(
            task_statement="Improve Loader runtime behavior.",
            explicit_sections=["desired_outcome", "non_goals", "decision_boundaries"],
            desired_outcome=["Make the runtime flow more disciplined."],
            non_goals=["Keep the CLI unchanged."],
            decision_boundaries=["Confirm before broad UX changes."],
        ),
        round_index=2,
    )

    assert assessment.pressure_pass_complete is True
    assert "pressure_pass" not in assessment.missing_readiness_gates


def test_build_clarify_question_can_render_pressure_pass_question() -> None:
    question = build_clarify_question(
        "Tighten the runtime behavior.",
        ClarifySlot.NON_GOALS,
        ClarifyPressureKind.TRADEOFF,
    )

    assert "unchanged" in question.lower() or "avoid" in question.lower()
