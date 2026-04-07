"""Tests for Sprint 04 workflow routing and artifact persistence."""

from __future__ import annotations

from pathlib import Path

from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.workflow import (
    ClarifyBrief,
    ModeRouter,
    PlanningArtifacts,
    WorkflowArtifactStore,
    WorkflowMode,
    build_execute_bridge,
    extract_verification_commands_from_markdown,
    sync_todos_to_definition_of_done,
)


def test_mode_router_routes_ambiguous_prompt_to_clarify() -> None:
    router = ModeRouter()

    decision = router.route("Improve Loader so it feels more like claw-code.")

    assert decision.mode == WorkflowMode.CLARIFY
    assert decision.ambiguity_score >= router.clarify_threshold


def test_mode_router_routes_complex_prompt_to_plan() -> None:
    router = ModeRouter()

    decision = router.route(
        "Implement a persistent workflow mode router with clarify artifacts, "
        "planning artifacts, and verification-plan wiring in the runtime."
    )

    assert decision.mode == WorkflowMode.PLAN
    assert decision.complexity_score >= router.plan_threshold


def test_mode_router_routes_simple_prompt_to_execute() -> None:
    router = ModeRouter()

    decision = router.route("Read pyproject.toml and tell me the package name.")

    assert decision.mode == WorkflowMode.EXECUTE


def test_clarify_brief_round_trips_and_seeds_acceptance_criteria() -> None:
    brief = ClarifyBrief.fallback(
        task_statement="Clarify the authentication change.",
        question="What outcome matters most?",
        answer="Add login without touching the signup flow.",
    )
    markdown = brief.to_markdown()

    loaded = ClarifyBrief.from_markdown(
        markdown,
        task_statement=brief.task_statement,
        question=brief.question,
        answer=brief.answer,
    )

    assert "single-question clarify brief" in markdown
    assert "return control to `execute` mode" in markdown
    assert loaded.task_statement == brief.task_statement
    assert "Add login" in loaded.acceptance_criteria[0]
    assert loaded.non_goals


def test_planning_artifacts_round_trip_and_extract_commands() -> None:
    artifacts = PlanningArtifacts.from_model_output(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## Execution Order",
                "1. Inspect auth files.",
                "2. Implement the change.",
                "",
                "## Risks",
                "- Regression in signup.",
                "",
                "<<<VERIFICATION>>>",
                "",
                "# Verification Plan",
                "",
                "## Acceptance Criteria",
                "- Login works without changing signup.",
                "",
                "## Verification Commands",
                "- `uv run pytest tests/test_auth.py -q`",
                "- `uv run mypy src/loader`",
            ]
        ),
        task_statement="Clarify and implement the auth change.",
    )

    assert artifacts.implementation_steps[:2] == [
        "Inspect auth files.",
        "Implement the change.",
    ]
    assert artifacts.acceptance_criteria == ["Login works without changing signup."]
    assert artifacts.verification_commands == [
        "uv run pytest tests/test_auth.py -q",
        "uv run mypy src/loader",
    ]
    assert extract_verification_commands_from_markdown(artifacts.verification_markdown) == [
        "uv run pytest tests/test_auth.py -q",
        "uv run mypy src/loader",
    ]


def test_workflow_artifact_store_and_bridge_round_trip(tmp_path: Path) -> None:
    store = WorkflowArtifactStore(tmp_path)
    brief = ClarifyBrief.fallback(
        task_statement="Clarify the runtime changes.",
        question="What matters most?",
        answer="Close the tool-use gap first.",
    )
    artifacts = PlanningArtifacts.fallback(task_statement=brief.task_statement)

    brief_path = store.write_brief(brief.task_statement, brief)
    implementation_path, verification_path = store.write_plan(
        brief.task_statement,
        artifacts,
    )
    bridge = build_execute_bridge(brief_path, implementation_path, verification_path)

    assert brief_path.exists()
    assert implementation_path.exists()
    assert verification_path.exists()
    assert bridge is not None
    assert "Task Brief" in bridge
    assert "Implementation Plan" in bridge
    assert "Verification Plan" in bridge


def test_definition_of_done_round_trip_preserves_workflow_links(tmp_path: Path) -> None:
    store = DefinitionOfDoneStore(tmp_path)
    dod = create_definition_of_done("Implement Loader workflow routing.")
    dod.current_mode = "plan"
    dod.mode_history = ["clarify", "plan"]
    dod.clarify_brief = str(tmp_path / ".loader" / "briefs" / "brief.md")
    dod.implementation_plan = str(tmp_path / ".loader" / "plans" / "impl.md")
    dod.verification_plan = str(tmp_path / ".loader" / "plans" / "verify.md")

    saved_path = store.save(dod)
    reloaded = store.load(saved_path)

    assert reloaded.current_mode == "plan"
    assert reloaded.mode_history == ["clarify", "plan"]
    assert reloaded.clarify_brief == dod.clarify_brief
    assert reloaded.implementation_plan == dod.implementation_plan
    assert reloaded.verification_plan == dod.verification_plan


def test_sync_todos_to_definition_of_done_preserves_runtime_items() -> None:
    dod = create_definition_of_done("Implement Loader workflow routing.")
    dod.pending_items.append("Collect verification evidence")

    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Write router",
                "active_form": "Writing router",
                "status": "in_progress",
            },
            {
                "content": "Update tests",
                "active_form": "Updating tests",
                "status": "completed",
            },
        ],
    )

    assert "Writing router" in dod.pending_items
    assert "Collect verification evidence" in dod.pending_items
    assert "Update tests" in dod.completed_items
