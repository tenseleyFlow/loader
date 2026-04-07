"""Tests for clarify grounding and brownfield repo evidence."""

from __future__ import annotations

from pathlib import Path

from loader.runtime.clarify_grounding import (
    ClarifyGrounding,
    ClarifyGroundingProbe,
    build_grounded_clarify_question,
)
from loader.runtime.clarify_strategy import ClarifyPressureKind, ClarifySlot


def test_probe_collects_workspace_references_and_candidate_touchpoints(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='loader'\n")
    (tmp_path / "src" / "loader" / "runtime").mkdir(parents=True)
    (tmp_path / "src" / "loader" / "runtime" / "workflow_lanes.py").write_text(
        '"""runtime lanes"""\n'
    )
    (tmp_path / "src" / "loader" / "runtime" / "clarify_strategy.py").write_text(
        '"""clarify strategy"""\n'
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


def test_grounding_prompt_block_renders_repo_evidence() -> None:
    grounding = ClarifyGrounding(
        project_type="python",
        top_level_entries=["src/", "tests/", "pyproject.toml"],
        existing_references=["src/loader/runtime/workflow_lanes.py"],
        candidate_touchpoints=["src/loader/runtime/clarify_strategy.py"],
        missing_references=["src/loader/runtime/unknown.py"],
    )

    block = grounding.prompt_block()

    assert "Project type: python" in block
    assert "Referenced paths that exist: src/loader/runtime/workflow_lanes.py" in block
    assert "Nearby repo touchpoints: src/loader/runtime/clarify_strategy.py" in block
    assert "Referenced paths not found: src/loader/runtime/unknown.py" in block


def test_build_grounded_question_anchors_touchpoint_tradeoff() -> None:
    question = build_grounded_clarify_question(
        task="Tighten Loader runtime clarify behavior.",
        focus_slot=ClarifySlot.LIKELY_TOUCHPOINTS,
        grounding=ClarifyGrounding(
            existing_references=["src/loader/runtime/workflow_lanes.py"]
        ),
        pressure_kind=ClarifyPressureKind.TRADEOFF,
    )

    assert question is not None
    assert "workflow_lanes.py" in question
    assert "stay unchanged" in question.lower()
