"""Tests for clarify grounding and brownfield repo evidence."""

from __future__ import annotations

from pathlib import Path

from loader.runtime.clarify_grounding import (
    ClarifyGrounding,
    ClarifyGroundingProbe,
    ClarifyRepoFact,
    build_grounded_clarify_question,
)
from loader.runtime.clarify_strategy import ClarifyPressureKind, ClarifySlot


def test_probe_collects_workspace_references_and_candidate_touchpoints(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='loader'\n")
    (tmp_path / "src" / "loader" / "runtime").mkdir(parents=True)
    (tmp_path / "src" / "loader" / "runtime" / "workflow_lanes.py").write_text(
        '"""Runtime lane orchestration for Loader."""\n\n'
        "class WorkflowLaneRunner:\n"
        "    pass\n"
    )
    (tmp_path / "src" / "loader" / "runtime" / "clarify_strategy.py").write_text(
        '"""Intent-aware clarify strategy for runtime follow-up."""\n'
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_workflow_runtime.py").write_text("pass\n")

    grounding = ClarifyGroundingProbe(tmp_path).collect(
        task=(
            "Tighten Loader runtime clarify behavior around "
            "src/loader/runtime/workflow_lanes.py without broad test churn."
        )
    )

    assert grounding.project_type == "python"
    assert "pyproject.toml" in grounding.top_level_entries
    assert "src/" in grounding.top_level_entries
    assert "src/loader/runtime/workflow_lanes.py" in grounding.existing_references
    assert any("clarify_strategy.py" in path for path in grounding.candidate_touchpoints)
    assert grounding.repo_facts
    assert grounding.repo_facts[0].path == "src/loader/runtime/workflow_lanes.py"
    assert "WorkflowLaneRunner" in grounding.repo_facts[0].summary


def test_grounding_prompt_block_renders_repo_evidence() -> None:
    grounding = ClarifyGrounding(
        project_type="python",
        top_level_entries=["src/", "tests/", "pyproject.toml"],
        existing_references=["src/loader/runtime/workflow_lanes.py"],
        candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
        repo_facts=[
            ClarifyRepoFact(
                path="src/loader/runtime/workflow_lanes.py",
                summary="class WorkflowLaneRunner:",
            )
        ],
        missing_references=["src/loader/runtime/unknown.py"],
    )

    block = grounding.prompt_block()

    assert "Project type: python" in block
    assert "Referenced paths that exist: src/loader/runtime/workflow_lanes.py" in block
    assert "Nearby repo touchpoints: src/loader/runtime/clarify_strategy.py" in block
    assert (
        "Observed repo facts: `src/loader/runtime/workflow_lanes.py`: "
        "class WorkflowLaneRunner:"
    ) in block
    assert "Referenced paths not found: src/loader/runtime/unknown.py" in block


def test_slot_prompt_block_prefers_primary_and_nearby_facts_for_non_goals() -> None:
    grounding = ClarifyGrounding(
        project_type="python",
        existing_references=["src/loader/runtime/workflow_lanes.py"],
        candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
        repo_facts=[
            ClarifyRepoFact(
                path="src/loader/runtime/workflow_lanes.py",
                summary="class WorkflowLaneRunner:",
            ),
            ClarifyRepoFact(
                path="src/loader/runtime/clarify_strategy.py",
                summary="Intent-aware clarify strategy for runtime follow-up.",
            ),
        ],
    )

    block = grounding.slot_prompt_block(ClarifySlot.NON_GOALS)

    assert "Relevant repo facts" in block
    assert "workflow_lanes.py" in block
    assert "clarify_strategy.py" in block
    assert "Relevant paths: src/loader/runtime/workflow_lanes.py" in block


def test_slot_prompt_block_adds_secondary_touchpoint_for_example_pressure() -> None:
    grounding = ClarifyGrounding(
        project_type="python",
        existing_references=["src/loader/runtime/workflow_lanes.py"],
        candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
        repo_facts=[
            ClarifyRepoFact(
                path="src/loader/runtime/workflow_lanes.py",
                summary="class WorkflowLaneRunner:",
            ),
            ClarifyRepoFact(
                path="src/loader/runtime/clarify_strategy.py",
                summary="Intent-aware clarify strategy for runtime follow-up.",
            ),
        ],
    )

    block = grounding.slot_prompt_block(
        ClarifySlot.LIKELY_TOUCHPOINTS,
        ClarifyPressureKind.EXAMPLE,
    )

    assert "Relevant repo facts" in block
    assert "workflow_lanes.py" in block
    assert "clarify_strategy.py" in block
    assert "Relevant paths: src/loader/runtime/workflow_lanes.py" in block


def test_brief_hints_seed_touchpoints_constraints_and_acceptance() -> None:
    grounding = ClarifyGrounding(
        existing_references=["src/loader/runtime/workflow_lanes.py"],
        candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
        repo_facts=[
            ClarifyRepoFact(
                path="src/loader/runtime/workflow_lanes.py",
                summary="class WorkflowLaneRunner:",
            ),
            ClarifyRepoFact(
                path="src/loader/runtime/clarify_strategy.py",
                summary="Intent-aware clarify strategy for runtime follow-up.",
            ),
        ],
    )

    hints = grounding.brief_hints()

    assert hints.likely_touchpoints == [
        "src/loader/runtime/workflow_lanes.py",
        "src/loader/runtime/clarify_strategy.py",
    ]
    assert any("workflow_lanes.py" in item for item in hints.constraints)
    assert any("clarify_strategy.py" in item for item in hints.constraints)
    assert any("WorkflowLaneRunner" in item for item in hints.assumptions)
    assert any("Primary work stays scoped" in item for item in hints.acceptance_criteria)


def test_brief_prompt_block_renders_grounded_brief_hints() -> None:
    grounding = ClarifyGrounding(
        existing_references=["src/loader/runtime/workflow_lanes.py"],
        candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
        repo_facts=[
            ClarifyRepoFact(
                path="src/loader/runtime/workflow_lanes.py",
                summary="class WorkflowLaneRunner:",
            ),
            ClarifyRepoFact(
                path="src/loader/runtime/clarify_strategy.py",
                summary="Intent-aware clarify strategy for runtime follow-up.",
            ),
        ],
    )

    block = grounding.brief_prompt_block()

    assert "Seed likely touchpoints" in block
    assert "Preserve constraints" in block
    assert "Grounded assumptions" in block
    assert "Scope acceptance criteria" in block
    assert "workflow_lanes.py" in block
    assert "clarify_strategy.py" in block


def test_build_grounded_question_anchors_touchpoint_tradeoff_with_repo_fact() -> None:
    question = build_grounded_clarify_question(
        task="Tighten Loader runtime clarify behavior.",
        focus_slot=ClarifySlot.LIKELY_TOUCHPOINTS,
        grounding=ClarifyGrounding(
            existing_references=["src/loader/runtime/workflow_lanes.py"],
            repo_facts=[
                ClarifyRepoFact(
                    path="src/loader/runtime/workflow_lanes.py",
                    summary="class WorkflowLaneRunner:",
                )
            ],
        ),
        pressure_kind=ClarifyPressureKind.TRADEOFF,
    )

    assert question is not None
    assert "workflow_lanes.py" in question
    assert "WorkflowLaneRunner" in question
    assert "stay unchanged" in question.lower()


def test_build_grounded_question_uses_counterexample_surface_for_example_pressure() -> None:
    question = build_grounded_clarify_question(
        task="Tighten Loader runtime clarify behavior.",
        focus_slot=ClarifySlot.LIKELY_TOUCHPOINTS,
        grounding=ClarifyGrounding(
            existing_references=["src/loader/runtime/workflow_lanes.py"],
            candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
            repo_facts=[
                ClarifyRepoFact(
                    path="src/loader/runtime/workflow_lanes.py",
                    summary="class WorkflowLaneRunner:",
                ),
                ClarifyRepoFact(
                    path="src/loader/runtime/clarify_strategy.py",
                    summary="Intent-aware clarify strategy for runtime follow-up.",
                ),
            ],
        ),
        pressure_kind=ClarifyPressureKind.EXAMPLE,
    )

    assert question is not None
    assert "counterexample surface" in question
    assert "clarify_strategy.py" in question


def test_build_grounded_question_uses_nearby_fact_for_decision_boundaries() -> None:
    question = build_grounded_clarify_question(
        task="Tighten Loader runtime clarify behavior.",
        focus_slot=ClarifySlot.DECISION_BOUNDARIES,
        grounding=ClarifyGrounding(
            existing_references=["src/loader/runtime/workflow_lanes.py"],
            candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
            repo_facts=[
                ClarifyRepoFact(
                    path="src/loader/runtime/workflow_lanes.py",
                    summary="class WorkflowLaneRunner:",
                ),
                ClarifyRepoFact(
                    path="src/loader/runtime/clarify_strategy.py",
                    summary="Intent-aware clarify strategy for runtime follow-up.",
                ),
            ],
        ),
        pressure_kind=ClarifyPressureKind.TRADEOFF,
    )

    assert question is not None
    assert "workflow_lanes.py" in question
    assert "clarify_strategy.py" in question
    assert "stop-and-confirm boundary" in question


def test_build_grounded_question_uses_risky_assumption_for_non_goal_pressure() -> None:
    question = build_grounded_clarify_question(
        task="Tighten Loader runtime clarify behavior.",
        focus_slot=ClarifySlot.NON_GOALS,
        grounding=ClarifyGrounding(
            existing_references=["src/loader/runtime/workflow_lanes.py"],
            candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
            repo_facts=[
                ClarifyRepoFact(
                    path="src/loader/runtime/workflow_lanes.py",
                    summary="class WorkflowLaneRunner:",
                ),
                ClarifyRepoFact(
                    path="src/loader/runtime/clarify_strategy.py",
                    summary="Intent-aware clarify strategy for runtime follow-up.",
                ),
            ],
        ),
        pressure_kind=ClarifyPressureKind.ASSUMPTION,
    )

    assert question is not None
    assert "risky" in question.lower()
    assert "clarify_strategy.py" in question
