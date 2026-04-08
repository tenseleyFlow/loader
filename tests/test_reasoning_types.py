"""Tests for runtime-owned reasoning type surfaces."""

from __future__ import annotations

from loader.runtime.reasoning_types import (
    ConfidenceAssessment,
    ConfidenceLevel,
    SelfCritique,
    Subtask,
    TaskCompletionCheck,
    TaskDecomposition,
)


def test_task_decomposition_tracks_progress_and_retry_state() -> None:
    decomposition = TaskDecomposition(
        original_task="Ship feature",
        subtasks=[
            Subtask(id="1", description="Read spec"),
            Subtask(id="2", description="Implement", dependencies=["1"]),
        ],
    )

    assert decomposition.next_subtask() is decomposition.subtasks[0]
    assert decomposition.progress_str() == "[0/2]"

    decomposition.mark_completed("1", "done")

    assert decomposition.progress_str() == "[1/2]"
    assert decomposition.next_subtask() is decomposition.subtasks[1]

    decomposition.mark_failed("2", "test failure")

    assert decomposition.can_retry("2") is True
    decomposition.reset_for_retry("2")
    assert decomposition.subtasks[1].status == "pending"


def test_confidence_assessment_exposes_score_helpers() -> None:
    assessment = ConfidenceAssessment(
        action="Write file",
        tool_name="write",
        tool_args={"file_path": "notes.txt"},
        level=ConfidenceLevel.LOW,
    )

    assert assessment.score == 2
    assert assessment.is_low_confidence is True


def test_self_critique_and_completion_defaults_are_stable() -> None:
    critique = SelfCritique(
        original_response="draft",
        should_revise=True,
        revision_count=1,
        max_revisions=2,
    )
    completion = TaskCompletionCheck(original_task="Ship feature")

    assert critique.can_revise() is True
    assert completion.is_complete is False
    assert completion.accomplished == []
