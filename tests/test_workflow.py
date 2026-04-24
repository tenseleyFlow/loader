"""Tests for Sprint 04 workflow routing and artifact persistence."""

from __future__ import annotations

from pathlib import Path

from loader.llm.base import ToolCall
from loader.runtime.clarify_grounding import ClarifyGrounding, ClarifyRepoFact
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.workflow import (
    ClarifyBrief,
    ModeRouter,
    PlanningArtifacts,
    WorkflowArtifactStore,
    WorkflowMode,
    advance_todos_from_tool_call,
    build_execute_bridge,
    effective_pending_todo_items,
    enrich_clarify_brief_with_grounding,
    extract_verification_commands_from_markdown,
    infer_pending_todo_output_target,
    merge_refreshed_todos_with_existing_scope,
    preferred_pending_todo_item,
    preserve_task_grounded_acceptance_criteria,
    reconcile_aggregate_completion_steps,
    sync_todos_to_definition_of_done,
)


def test_mode_router_routes_ambiguous_prompt_to_clarify() -> None:
    router = ModeRouter()

    decision = router.route("Improve Loader so it feels more like claw-code.")

    assert decision.mode == WorkflowMode.CLARIFY
    assert decision.reason_code == "task_is_ambiguous"
    assert decision.route_score >= decision.runner_up_score


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


def test_enrich_clarify_brief_with_grounding_replaces_generic_sections() -> None:
    brief = ClarifyBrief.fallback(
        task_statement="Tighten Loader runtime clarify behavior.",
        question="What part should change most?",
        answer="Keep the work narrow.",
    )
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

    enriched = enrich_clarify_brief_with_grounding(brief, grounding)

    assert enriched.likely_touchpoints[:2] == [
        "src/loader/runtime/workflow_lanes.py",
        "src/loader/runtime/clarify_strategy.py",
    ]
    assert any("clarify_strategy.py" in item for item in enriched.constraints)
    assert any("WorkflowLaneRunner" in item for item in enriched.assumptions)
    assert any("Primary work stays scoped" in item for item in enriched.acceptance_criteria)
    assert "likely_touchpoints" in enriched.explicit_sections


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

    assert "single-pass planning artifact generation" in artifacts.implementation_markdown
    assert "planner/critic consensus loop" in artifacts.implementation_markdown
    assert "single-pass planning artifact generation" in artifacts.verification_markdown
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


def test_planning_artifacts_recover_embedded_verification_from_legacy_separator() -> None:
    artifacts = PlanningArtifacts.from_model_output(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## Execution Order",
                "1. Inspect index.html.",
                "2. Fix the chapter links.",
                "",
                "# Verification Plan",
                "",
                "## Acceptance Criteria",
                "- All chapter links point to real files.",
                "",
                "## Verification Commands",
                "- `grep -o 'href=\"[^\"]*\"' index.html`",
                "- `ls chapters`",
                "",
                "<<VERIFICATION>>",
            ]
        ),
        task_statement="Fix the broken chapter links in index.html.",
    )

    assert "## Verification Commands" not in artifacts.implementation_markdown
    assert "## Verification Commands" in artifacts.verification_markdown
    assert artifacts.verification_commands == [
        "grep -o 'href=\"[^\"]*\"' index.html",
        "ls chapters",
    ]


def test_extract_verification_commands_from_markdown_splits_code_blocks() -> None:
    markdown = "\n".join(
        [
            "# Verification Plan",
            "",
            "## Verification Commands",
            "```bash",
            "# Check chapter files",
            "ls chapters",
            "grep -n \"href=\" index.html",
            "```",
        ]
    )

    assert extract_verification_commands_from_markdown(markdown) == [
        "ls chapters",
        'grep -n "href=" index.html',
    ]


def test_extract_verification_commands_from_markdown_ignores_prose_only_bullets() -> None:
    markdown = "\n".join(
        [
            "# Verification Plan",
            "",
            "## Verification Commands",
            "- Check that all chapter links in index.html resolve to existing files",
            "- Validate chapter titles with `python3 scripts/check_titles.py`",
            "- `test -f index.html`",
        ]
    )

    assert extract_verification_commands_from_markdown(markdown) == [
        "python3 scripts/check_titles.py",
        "test -f index.html",
    ]


def test_extract_verification_commands_from_markdown_trims_inline_command_explanations() -> None:
    markdown = "\n".join(
        [
            "# Verification Plan",
            "",
            "## Verification Commands",
            (
                '- `grep -n "href" ~/Loader/guides/fortran/index.html` '
                "- to identify all href attributes"
            ),
        ]
    )

    assert extract_verification_commands_from_markdown(markdown) == [
        'grep -n "href" ~/Loader/guides/fortran/index.html',
    ]


def test_extract_verification_commands_keeps_shell_pipelines_intact() -> None:
    markdown = "\n".join(
        [
            "# Verification Plan",
            "",
            "## Verification Commands",
            "```bash",
            "ls -la chapters/",
            "cat index.html | head -20",
            "```",
        ]
    )

    assert extract_verification_commands_from_markdown(markdown) == [
        "ls -la chapters/",
        "cat index.html | head -20",
    ]


def test_preserve_task_grounded_acceptance_criteria_keeps_original_scope_on_refresh() -> None:
    task = (
        "Create an equally thorough nginx guide with index.html plus chapter files "
        "covering getting started, installation, first website setup, configs, and "
        "advanced topics."
    )

    preserved = preserve_task_grounded_acceptance_criteria(
        task,
        existing_acceptance_criteria=[
            "All files are created in the correct locations with proper directory structure",
            "Content covers all required topics: getting started, installation, first website, configuration basics, advanced configurations, and troubleshooting",
        ],
        refreshed_acceptance_criteria=[
            "At least one chapter file exists in ~/Loader/guides/nginx/chapters/",
            "~/Loader/guides/nginx/index.html exists and contains proper table of contents",
        ],
    )

    assert (
        "All files are created in the correct locations with proper directory structure"
        in preserved
    )
    assert (
        "Content covers all required topics: getting started, installation, first website, configuration basics, advanced configurations, and troubleshooting"
        in preserved
    )
    assert "At least one chapter file exists in ~/Loader/guides/nginx/chapters/" in preserved


def test_preserve_task_grounded_acceptance_criteria_drops_stale_plan_specific_scope() -> None:
    task = (
        "Implement a persistent workflow artifact with planning artifacts, "
        "verification commands, and plan refresh discipline."
    )

    preserved = preserve_task_grounded_acceptance_criteria(
        task,
        existing_acceptance_criteria=["planned.txt exists in the workspace root."],
        refreshed_acceptance_criteria=["notes.txt exists in the workspace root."],
    )

    assert preserved == ["notes.txt exists in the workspace root."]


def test_planning_artifacts_with_acceptance_criteria_rewrites_verification_markdown() -> None:
    artifacts = PlanningArtifacts.from_model_output(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## Execution Order",
                "1. Create the guide files.",
                "",
                "<<<VERIFICATION>>>",
                "",
                "# Verification Plan",
                "",
                "## Acceptance Criteria",
                "- At least one chapter file exists.",
                "",
                "## Verification Commands",
                "- `find chapters -name \"*.html\" | wc -l`",
            ]
        ),
        task_statement="Create a thorough nginx guide.",
    )

    updated = artifacts.with_acceptance_criteria(
        [
            "All files are created in the correct locations.",
            "Content covers getting started, installation, and advanced topics.",
        ]
    )

    assert "At least one chapter file exists." not in updated.verification_markdown
    assert "All files are created in the correct locations." in updated.verification_markdown
    assert (
        "Content covers getting started, installation, and advanced topics."
        in updated.verification_markdown
    )


def test_planning_artifacts_with_progress_context_records_touched_and_completed_work() -> None:
    artifacts = PlanningArtifacts.from_model_output(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## Execution Order",
                "1. Create the guide files.",
                "",
                "<<<VERIFICATION>>>",
                "",
                "# Verification Plan",
                "",
                "## Acceptance Criteria",
                "- At least one chapter file exists.",
                "",
                "## Verification Commands",
                "- `find chapters -name \"*.html\" | wc -l`",
            ]
        ),
        task_statement="Create a thorough nginx guide.",
    )

    updated = artifacts.with_progress_context(
        touched_files=["/tmp/nginx/index.html"],
        completed_items=[
            "Create the guide scaffold",
            "Collect verification evidence",
        ],
    )

    assert "## Confirmed Progress" in updated.implementation_markdown
    assert "Already touched during execution: `/tmp/nginx/index.html`." in (
        updated.implementation_markdown
    )
    assert "Already completed during execution: Create the guide scaffold." in (
        updated.implementation_markdown
    )
    assert "Collect verification evidence" not in updated.implementation_markdown


def test_merge_refreshed_todos_with_existing_scope_keeps_grounded_progress() -> None:
    task = (
        "Create an equally thorough nginx guide with index.html plus chapter files "
        "covering getting started, installation, first website setup, configs, and "
        "advanced topics."
    )

    todos = merge_refreshed_todos_with_existing_scope(
        task,
        existing_pending_items=[
            "Create each chapter file in sequence, following the established pattern",
            "Collect verification evidence",
        ],
        existing_completed_items=["Create directory structure for the new nginx guide"],
        refreshed_steps=["Create sample chapter file to verify the structure works"],
    )

    assert todos[0]["content"] == "Create directory structure for the new nginx guide"
    assert todos[0]["status"] == "completed"
    assert any(
        item["content"] == "Create each chapter file in sequence, following the established pattern"
        and item["status"] == "pending"
        for item in todos
    )


def test_merge_refreshed_todos_with_existing_scope_filters_retro_refresh_noise() -> None:
    task = (
        "Create an equally thorough nginx guide with index.html plus chapter files "
        "covering getting started, installation, first website setup, configs, and "
        "advanced topics."
    )

    todos = merge_refreshed_todos_with_existing_scope(
        task,
        existing_pending_items=[
            "Create each chapter file in sequence, following the same structure as the Fortran guide",
            "Ensure all files are properly linked and formatted consistently",
        ],
        existing_completed_items=[
            "First, examine the existing Fortran guide structure to understand the format and cadence",
            "Create the directory structure for the new nginx guide",
            "Create the main index.html file",
        ],
        refreshed_steps=[
            "First examined the existing Fortran guide structure to understand format and cadence",
            "Created the main index.html file with navigation",
            "Created chapter files in sequence:",
            "01-getting-started.html",
            "02-installation.html",
            "03-first-website.html",
            "04-configuring.html",
            "All files properly linked with navigation between chapters",
            "Verify the final navigation links across the guide",
        ],
    )

    labels = {item["content"]: item["status"] for item in todos}
    assert (
        labels["Create each chapter file in sequence, following the same structure as the Fortran guide"]
        == "pending"
    )
    assert labels["Ensure all files are properly linked and formatted consistently"] == "pending"
    assert labels["Verify the final navigation links across the guide"] == "pending"
    assert "Created chapter files in sequence:" not in labels
    assert "04-configuring.html" not in labels


def test_merge_refreshed_todos_with_existing_scope_drops_unplanned_filename_expansion() -> None:
    task = (
        "Create an equally thorough nginx guide with index.html plus chapter files "
        "covering getting started, installation, configuration, usage, and troubleshooting."
    )

    todos = merge_refreshed_todos_with_existing_scope(
        task,
        existing_pending_items=[
            "Create chapter files with appropriate content structure",
        ],
        existing_completed_items=[
            "Create the nginx guide directory structure",
            "Create introduction.html",
        ],
        refreshed_steps=[
            "Create optimization.html",
            "Create security.html",
            "Ensure consistent chapter navigation",
        ],
        planned_files={
            "index.html",
            "introduction.html",
            "installation.html",
            "configuration.html",
            "usage.html",
            "troubleshooting.html",
        },
    )

    labels = {item["content"]: item["status"] for item in todos}
    assert "Create chapter files with appropriate content structure" in labels
    assert "Ensure consistent chapter navigation" in labels
    assert "Create optimization.html" not in labels
    assert "Create security.html" not in labels


def test_planning_artifacts_with_file_changes_replaces_file_change_section() -> None:
    artifacts = PlanningArtifacts(
        implementation_markdown="\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                "- `old.txt`",
                "",
                "## Execution Order",
                "- Do the work",
                "",
            ]
        )
        + "\n",
        verification_markdown="# Verification Plan\n",
        verification_commands=[],
        acceptance_criteria=["task"],
        implementation_steps=["Do the work"],
    )

    updated = artifacts.with_file_changes(
        ["`guides/nginx/index.html`", "`guides/nginx/chapters/`"]
    )

    assert "`old.txt`" not in updated.implementation_markdown
    assert "`guides/nginx/index.html`" in updated.implementation_markdown
    assert "`guides/nginx/chapters/`" in updated.implementation_markdown


def test_effective_pending_todo_items_filters_stale_discovery_after_artifacts_exist(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "First, examine the existing Fortran guide structure to understand the format and content organization",
        "Verify all guide files are linked and complete",
        "Complete the requested work",
    ]

    pending = effective_pending_todo_items(dod, project_root=temp_dir)

    assert "Verify all guide files are linked and complete" in pending
    assert "Complete the requested work" in pending
    assert not any("Fortran guide structure" in item for item in pending)


def test_effective_pending_todo_items_filters_completed_setup_before_build_finishes(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-introduction.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "Create the nginx directory structure",
        "Create each chapter file with appropriate content",
        "Complete the requested work",
    ]

    pending = effective_pending_todo_items(dod, project_root=temp_dir)

    assert "Create the nginx directory structure" not in pending
    assert "Create each chapter file with appropriate content" in pending
    assert "Complete the requested work" in pending


def test_effective_pending_todo_items_filters_stale_creation_steps_after_artifacts_exist(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "Create 01-getting-started.html",
        "Creating 02-installation.html",
        "Verify all guide files are linked and complete",
        "Complete the requested work",
    ]

    pending = effective_pending_todo_items(dod, project_root=temp_dir)

    assert "Verify all guide files are linked and complete" in pending
    assert "Complete the requested work" in pending
    assert "Create 01-getting-started.html" not in pending
    assert "Creating 02-installation.html" not in pending


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


def test_sync_todos_to_definition_of_done_keeps_completed_items_monotonic() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create 03-first-website.html",
                "active_form": "Creating 03-first-website.html",
                "status": "pending",
            },
            {
                "content": "Create 04-configuration-basics.html",
                "active_form": "Creating 04-configuration-basics.html",
                "status": "pending",
            },
        ],
    )

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="write-third-chapter",
            name="write",
            arguments={
                "file_path": "/tmp/nginx/chapters/03-first-website.html",
                "content": "<html></html>",
            },
        ),
    )
    assert "Create 03-first-website.html" in dod.completed_items

    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create 03-first-website.html",
                "active_form": "Creating 03-first-website.html",
                "status": "pending",
            },
            {
                "content": "Create 04-configuration-basics.html",
                "active_form": "Creating 04-configuration-basics.html",
                "status": "pending",
            },
        ],
    )

    assert "Create 03-first-website.html" in dod.completed_items
    assert "Create 03-first-website.html" not in dod.pending_items
    assert "Create 04-configuration-basics.html" in dod.pending_items


def test_advance_todos_from_tool_call_tracks_plan_progress() -> None:
    dod = create_definition_of_done("Fix the chapter links in index.html.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the current index.html file to understand its structure",
                "active_form": "Working on: First, examine the current index.html file to understand its structure",
                "status": "pending",
            },
            {
                "content": "List and read all HTML files in the chapters directory to extract chapter information",
                "active_form": "Working on: List and read all HTML files in the chapters directory to extract chapter information",
                "status": "pending",
            },
            {
                "content": "Parse chapter titles from each HTML file",
                "active_form": "Working on: Parse chapter titles from each HTML file",
                "status": "pending",
            },
            {
                "content": "Update index.html with correct chapter links and titles",
                "active_form": "Working on: Update index.html with correct chapter links and titles",
                "status": "pending",
            },
            {
                "content": "Verify the updated index.html file is properly formatted",
                "active_form": "Working on: Verify the updated index.html file is properly formatted",
                "status": "pending",
            },
        ],
    )

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="read-index",
            name="read",
            arguments={"file_path": "/tmp/fortran/index.html"},
        ),
    )
    assert (
        "First, examine the current index.html file to understand its structure"
        in dod.completed_items
    )

    assert not advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="glob-chapters",
            name="glob",
            arguments={"path": "/tmp/fortran/chapters", "pattern": "*.html"},
        ),
    )
    assert (
        "List and read all HTML files in the chapters directory to extract chapter information"
        in dod.pending_items
    )

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="read-chapter",
            name="read",
            arguments={"file_path": "/tmp/fortran/chapters/01-introduction.html"},
        ),
    )
    assert (
        "List and read all HTML files in the chapters directory to extract chapter information"
        in dod.completed_items
    )

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="read-second-chapter",
            name="read",
            arguments={"file_path": "/tmp/fortran/chapters/02-setup.html"},
        ),
    )
    assert "Parse chapter titles from each HTML file" in dod.completed_items

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="patch-index",
            name="patch",
            arguments={"file_path": "/tmp/fortran/index.html", "hunks": []},
        ),
    )
    assert "Update index.html with correct chapter links and titles" in dod.completed_items

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="verify-index",
            name="bash",
            arguments={"command": "grep -o 'href=\"[^\"]*\"' /tmp/fortran/index.html"},
        ),
    )
    assert "Verify the updated index.html file is properly formatted" in dod.completed_items


def test_advance_todos_from_tool_call_keeps_aggregate_mutation_steps_pending() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create each chapter file in sequence, following the same structure as the Fortran guide",
                "active_form": "Working on: Create each chapter file in sequence, following the same structure as the Fortran guide",
                "status": "pending",
            },
            {
                "content": "Ensure all files are properly linked and formatted consistently",
                "active_form": "Working on: Ensure all files are properly linked and formatted consistently",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="write-one-chapter",
                name="write",
                arguments={
                    "file_path": "/tmp/nginx/chapters/01-getting-started.html",
                    "content": "<html></html>",
                },
            ),
        )
        is False
    )
    assert (
        "Create each chapter file in sequence, following the same structure as the Fortran guide"
        in dod.pending_items
    )


def test_advance_todos_from_tool_call_keeps_plural_chapter_creation_step_pending() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create chapter files following the established pattern",
                "active_form": "Working on: Create chapter files following the established pattern",
                "status": "pending",
            },
            {
                "content": "Ensure consistency with existing guide formatting and content style",
                "active_form": "Working on: Ensure consistency with existing guide formatting and content style",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="write-one-chapter",
                name="write",
                arguments={
                    "file_path": "/tmp/nginx/chapters/01-overview.html",
                    "content": "<html></html>",
                },
            ),
        )
        is False
    )
    assert "Create chapter files following the established pattern" in dod.pending_items


def test_advance_todos_from_tool_call_tracks_bash_directory_creation_progress() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create the nginx directory structure",
                "active_form": "Working on: Create the nginx directory structure",
                "status": "pending",
            },
            {
                "content": "Create index.html for nginx guide",
                "active_form": "Working on: Create index.html for nginx guide",
                "status": "pending",
            },
        ],
    )

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="mkdir-nginx",
            name="bash",
            arguments={"command": "mkdir -p ~/Loader/guides/nginx/chapters"},
        ),
    )
    assert "Create the nginx directory structure" in dod.completed_items
    assert "Create index.html for nginx guide" in dod.pending_items


def test_infer_pending_todo_output_target_maps_broad_setup_to_planned_directory(
    tmp_path: Path,
) -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    nginx_root = tmp_path / "Loader" / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    implementation_plan = tmp_path / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{nginx_root / 'index.html'}`",
                "",
            ]
        )
    )
    dod.implementation_plan = str(implementation_plan)

    target = infer_pending_todo_output_target(
        dod,
        "Create the nginx directory structure",
        project_root=tmp_path,
    )

    assert target == chapters.resolve(strict=False)


def test_preferred_pending_todo_item_keeps_setup_step_when_missing_file_parent_absent(
    tmp_path: Path,
) -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    nginx_root = tmp_path / "Loader" / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    index_path = nginx_root / "index.html"
    implementation_plan = tmp_path / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                "",
            ]
        )
    )
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "Create the nginx directory structure",
        "Create the main index.html file for nginx guide",
        "Complete the requested work",
    ]

    preferred = preferred_pending_todo_item(
        dod,
        project_root=tmp_path,
        missing_artifact=(index_path.resolve(strict=False), False),
    )

    assert preferred == "Create the nginx directory structure"


def test_advance_todos_from_tool_call_does_not_complete_content_study_from_root_index_read() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the existing fortran guide structure and content",
                "active_form": "Working on: First, examine the existing fortran guide structure and content",
                "status": "pending",
            },
            {
                "content": "Develop the main index.html file for the nginx guide",
                "active_form": "Working on: Develop the main index.html file for the nginx guide",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="read-reference-index",
                name="read",
                arguments={"file_path": "~/Loader/guides/fortran/index.html"},
            ),
        )
        is False
    )
    assert (
        "First, examine the existing fortran guide structure and content"
        in dod.pending_items
    )
    assert "Develop the main index.html file for the nginx guide" in dod.pending_items


def test_advance_todos_from_tool_call_does_not_complete_content_examination_from_shallow_glob() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the existing fortran guide structure and content",
                "active_form": "Working on: First, examine the existing fortran guide structure and content",
                "status": "pending",
            },
            {
                "content": "Develop the main index.html file for the nginx guide",
                "active_form": "Working on: Develop the main index.html file for the nginx guide",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="glob-reference-root",
                name="glob",
                arguments={"path": "~/Loader/guides/fortran", "pattern": "**"},
            ),
        )
        is False
    )
    assert (
        "First, examine the existing fortran guide structure and content"
        in dod.pending_items
    )
    assert "Develop the main index.html file for the nginx guide" in dod.pending_items


def test_advance_todos_from_tool_call_does_not_complete_format_study_from_shallow_glob() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the existing fortran guide structure to understand the format",
                "active_form": "Working on: First, examine the existing fortran guide structure to understand the format",
                "status": "pending",
            },
            {
                "content": "Create the main index.html file for nginx guide",
                "active_form": "Working on: Create the main index.html file for nginx guide",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="glob-reference-root",
                name="glob",
                arguments={"path": "~/Loader/guides/fortran", "pattern": "**"},
            ),
        )
        is False
    )
    assert (
        "First, examine the existing fortran guide structure to understand the format"
        in dod.pending_items
    )
    assert "Create the main index.html file for nginx guide" in dod.pending_items


def test_advance_todos_from_tool_call_does_not_complete_deep_guide_study_from_root_index_read() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the existing fortran guide structure to understand the content organization and cadence",
                "active_form": "Working on: First, examine the existing fortran guide structure to understand the content organization and cadence",
                "status": "pending",
            },
            {
                "content": "Develop the main index.html file for the nginx guide",
                "active_form": "Working on: Develop the main index.html file for the nginx guide",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="read-reference-index",
                name="read",
                arguments={"file_path": "~/Loader/guides/fortran/index.html"},
            ),
        )
        is False
    )
    assert (
        "First, examine the existing fortran guide structure to understand the content organization and cadence"
        in dod.pending_items
    )
    assert "Develop the main index.html file for the nginx guide" in dod.pending_items


def test_advance_todos_from_tool_call_does_not_complete_populate_step_from_reference_read() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the existing fortran guide structure and content",
                "active_form": "Working on: First, examine the existing fortran guide structure and content",
                "status": "pending",
            },
            {
                "content": "Populate content for each chapter",
                "active_form": "Working on: Populate content for each chapter",
                "status": "pending",
            },
        ],
    )

    assert advance_todos_from_tool_call(
        dod,
        ToolCall(
            id="read-reference-chapter",
            name="read",
            arguments={"file_path": "~/Loader/guides/fortran/chapters/01-introduction.html"},
        ),
    )
    assert (
        "First, examine the existing fortran guide structure and content"
        in dod.completed_items
    )
    assert "Populate content for each chapter" in dod.pending_items


def test_advance_todos_from_tool_call_does_not_complete_linking_step_from_glob() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Link all chapters together properly in the index file",
                "active_form": "Working on: Link all chapters together properly in the index file",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="glob-reference-chapters",
                name="glob",
                arguments={"path": "~/Loader", "pattern": "**/fortran/chapters/*"},
            ),
        )
        is False
    )
    assert "Link all chapters together properly in the index file" in dod.pending_items


def test_advance_todos_from_tool_call_does_not_complete_aggregate_style_step_from_reference_read() -> None:
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create each chapter file with appropriate content",
                "active_form": "Working on: Create each chapter file with appropriate content",
                "status": "pending",
            },
            {
                "content": "Ensure all files follow the same structure and style as the Fortran guide",
                "active_form": "Working on: Ensure all files follow the same structure and style as the Fortran guide",
                "status": "pending",
            },
        ],
    )

    assert (
        advance_todos_from_tool_call(
            dod,
            ToolCall(
                id="read-reference-index",
                name="read",
                arguments={"file_path": "~/Loader/guides/fortran/index.html"},
            ),
        )
        is False
    )
    assert "Create each chapter file with appropriate content" in dod.pending_items
    assert (
        "Ensure all files follow the same structure and style as the Fortran guide"
        in dod.pending_items
    )


def test_sync_todos_to_definition_of_done_keeps_linking_step_pending_while_artifacts_missing(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create 01-getting-started.html chapter file",
                "active_form": "Creating 01-getting-started.html chapter file",
                "status": "completed",
            },
            {
                "content": "Link all chapters together properly in the index file",
                "active_form": "Linking chapters in the index file",
                "status": "completed",
            },
            {
                "content": "Create 02-installation.html chapter file",
                "active_form": "Creating 02-installation.html chapter file",
                "status": "pending",
            },
        ],
        project_root=temp_dir,
    )

    assert "Link all chapters together properly in the index file" in dod.pending_items
    assert "Link all chapters together properly in the index file" not in dod.completed_items


def test_sync_todos_to_definition_of_done_allows_linking_step_when_artifacts_exist(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Link all chapters together properly in the index file",
                "active_form": "Linking chapters in the index file",
                "status": "completed",
            },
        ],
        project_root=temp_dir,
    )

    assert "Link all chapters together properly in the index file" in dod.completed_items


def test_sync_todos_to_definition_of_done_reopens_directory_content_step_when_output_dir_is_empty(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    index_path.write_text("<html></html>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root / 'index.html'}`",
                f"- `{chapters}/` (directory for chapter files)",
                "",
                "## Execution Order",
                "- Create chapter files with appropriate content",
            ]
        )
    )

    dod = create_definition_of_done("Create an equally thorough nginx guide with chapters.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create chapter files with appropriate content",
                "active_form": "Creating chapter files with appropriate content",
                "status": "completed",
            },
        ],
        project_root=temp_dir,
    )

    assert "Create chapter files with appropriate content" in dod.pending_items
    assert "Create chapter files with appropriate content" not in dod.completed_items


def test_reconcile_aggregate_completion_steps_reopens_linking_step_when_artifacts_missing(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    chapter_three = chapters / "03-first-website.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                f"- `{chapter_three}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.completed_items.append("Link all chapters together properly")

    reconcile_aggregate_completion_steps(dod, project_root=temp_dir)

    assert "Link all chapters together properly" not in dod.completed_items
    assert "Link all chapters together properly" in dod.pending_items


def test_sync_todos_to_definition_of_done_drops_unplanned_artifact_expansion_after_plan_complete(
    temp_dir: Path,
) -> None:
    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create 01-getting-started.html",
                "active_form": "Creating 01-getting-started.html",
                "status": "completed",
            },
            {
                "content": "Create 02-installation.html",
                "active_form": "Creating 02-installation.html",
                "status": "completed",
            },
            {
                "content": "Create 07-performance-tuning.html",
                "active_form": "Creating 07-performance-tuning.html",
                "status": "in_progress",
            },
        ],
        project_root=temp_dir,
    )

    assert "Creating 07-performance-tuning.html" not in dod.pending_items
    assert "Create 01-getting-started.html" in dod.completed_items
    assert "Create 02-installation.html" in dod.completed_items
