"""Tests for definition-of-done state and persistence."""

from pathlib import Path

from loader.llm.base import ToolCall
from loader.runtime.dod import (
    DefinitionOfDoneStore,
    VerificationEvidence,
    all_planned_artifacts_exist,
    begin_new_verification_attempt,
    build_verification_summary,
    collect_planned_artifact_targets,
    create_definition_of_done,
    derive_verification_commands,
    determine_task_size,
    ensure_active_verification_attempt,
    record_successful_tool_call,
)


def test_determine_task_size_boundaries() -> None:
    assert determine_task_size(1, 10) == "small"
    assert determine_task_size(3, 99) == "small"
    assert determine_task_size(4, 99) == "standard"
    assert determine_task_size(15, 499) == "standard"
    assert determine_task_size(16, 499) == "large"
    assert determine_task_size(15, 500) == "large"


def test_definition_of_done_round_trip(tmp_path: Path) -> None:
    store = DefinitionOfDoneStore(tmp_path)
    dod = create_definition_of_done(
        "Create hello.py and verify it runs.",
        retry_budget=2,
    )
    dod.status = "fixing"
    dod.retry_count = 1
    dod.verification_commands = ["python hello.py"]
    dod.touched_files = [str(tmp_path / "hello.py")]
    attempt = begin_new_verification_attempt(dod)
    saved_path = store.save(dod)

    reloaded = store.load(saved_path)

    assert reloaded.task_statement == dod.task_statement
    assert reloaded.status == "fixing"
    assert reloaded.retry_count == 1
    assert reloaded.verification_commands == ["python hello.py"]
    assert reloaded.touched_files == [str(tmp_path / "hello.py")]
    assert reloaded.active_verification_attempt_id == attempt.attempt_id
    assert reloaded.active_verification_attempt_number == attempt.attempt_number


def test_ensure_active_verification_attempt_rehydrates_missing_active_attempt() -> None:
    dod = create_definition_of_done("Verify the runtime output.")
    dod.verification_attempt_counter = 2

    attempt = ensure_active_verification_attempt(dod)

    assert attempt.attempt_id == "verification-attempt-2"
    assert attempt.attempt_number == 2
    assert dod.active_verification_attempt_id == "verification-attempt-2"
    assert dod.active_verification_attempt_number == 2


def test_verification_command_derivation_prefers_runtime_evidence(tmp_path: Path) -> None:
    project_root = tmp_path
    dod = create_definition_of_done("Create hello.py and make sure it runs.")
    hello_path = project_root / "hello.py"
    record_successful_tool_call(
        dod,
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(hello_path), "content": "print('hi')\n"},
        ),
    )
    record_successful_tool_call(
        dod,
        ToolCall(
            id="bash-1",
            name="bash",
            arguments={"command": "python hello.py"},
        ),
    )

    commands = derive_verification_commands(
        dod,
        project_root=project_root,
        task_statement=dod.task_statement,
    )

    assert commands == ["python hello.py"]


def test_record_successful_tool_call_preserves_absolute_path_string(tmp_path: Path) -> None:
    dod = create_definition_of_done("Create hello.py and verify it exists.")
    absolute_path = tmp_path / "hello.py"

    record_successful_tool_call(
        dod,
        ToolCall(
            id="write-1",
            name="write",
            arguments={"file_path": str(absolute_path), "content": "print('hi')\n"},
        ),
    )

    assert dod.touched_files == [str(absolute_path)]


def test_derive_verification_commands_adds_semantic_html_toc_check(tmp_path: Path) -> None:
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    index = tmp_path / "index.html"
    index.write_text(
        "\n".join(
            [
                '<ul class="chapter-list">',
                '  <li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>',
                "</ul>",
            ]
        )
    )

    dod = create_definition_of_done(
        "Update index.html so the table of contents links and hrefs are correct."
    )
    dod.acceptance_criteria = [
        "All table of contents links in index.html point to existing chapter files.",
        "All link texts match the actual chapter titles.",
    ]
    dod.touched_files = [str(index)]

    commands = derive_verification_commands(
        dod,
        project_root=tmp_path,
        task_statement=dod.task_statement,
    )

    assert any(command.startswith("python3 - <<'PY'") for command in commands)
    assert not any(command == f"test -f {index}" for command in commands)


def test_derive_verification_commands_avoids_repo_defaults_for_external_artifacts(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='loader'\n")
    (tmp_path / "package.json").write_text("{}\n")
    external_root = tmp_path.parent / "external-guide"
    external_root.mkdir(exist_ok=True)
    external_index = external_root / "index.html"
    external_index.write_text("<html></html>\n")

    dod = create_definition_of_done("Create an external nginx guide.")
    dod.task_size = "standard"
    dod.touched_files = [str(external_index)]

    commands = derive_verification_commands(
        dod,
        project_root=tmp_path,
        task_statement=dod.task_statement,
    )

    assert commands == [f"test -f {external_index}"]


def test_derive_verification_commands_adds_generic_local_html_link_check(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    index = docs / "index.html"
    index.write_text('<a href="chapters/01-intro.html">Intro</a>\n')

    dod = create_definition_of_done("Create a small multi-page HTML guide.")
    dod.touched_files = [str(index)]

    commands = derive_verification_commands(
        dod,
        project_root=tmp_path,
        task_statement=dod.task_statement,
        supplement_existing=True,
    )

    assert any("Missing local HTML links:" in command for command in commands)


def test_derive_verification_commands_adds_planned_artifact_existence_checks(
    tmp_path: Path,
) -> None:
    implementation_plan = tmp_path / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                "- `docs/index.html`",
                "- `docs/chapters/01-intro.html`",
                "- `docs/chapters/02-installation.html`",
                "- `docs/chapters/`",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-page HTML guide.")
    dod.implementation_plan = str(implementation_plan)

    commands = derive_verification_commands(
        dod,
        project_root=tmp_path,
        task_statement=dod.task_statement,
        supplement_existing=True,
    )

    assert f"test -f {tmp_path / 'docs/index.html'}" in commands
    assert f"test -f {tmp_path / 'docs/chapters/01-intro.html'}" in commands
    assert f"test -f {tmp_path / 'docs/chapters/02-installation.html'}" in commands
    assert f"test -d {tmp_path / 'docs/chapters'}" in commands


def test_derive_verification_commands_adds_html_guide_quality_check_for_thorough_guides(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    chapters = docs / "chapters"
    chapters.mkdir(parents=True)
    implementation_plan = tmp_path / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{docs / 'index.html'}`",
                f"- `{chapters / '01-introduction.html'}`",
                f"- `{chapters / '02-installation.html'}`",
                f"- `{chapters / '03-configuration.html'}`",
                f"- `{chapters / '04-troubleshooting.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done(
        "Create an equally thorough multi-page HTML guide with chapter files."
    )
    dod.implementation_plan = str(implementation_plan)

    commands = derive_verification_commands(
        dod,
        project_root=tmp_path,
        task_statement=dod.task_statement,
        supplement_existing=True,
    )

    assert any("HTML guide content quality issues:" in command for command in commands)


def test_collect_planned_artifact_targets_ignores_prose_path_fragments_in_refreshed_plan(
    tmp_path: Path,
) -> None:
    implementation_plan = tmp_path / "implementation.md"
    touched_index = tmp_path / "external" / "guides" / "nginx" / "index.html"
    touched_index.parent.mkdir(parents=True)
    touched_index.write_text("<html></html>\n")
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                "- Created main index.html file with proper structure and navigation",
                "- Created the nginx guide directory structure (chapters/)",
                "- Created the first chapter file (01-introduction.html) with appropriate content",
                "",
                "## Confirmed Progress",
                f"- Already touched during execution: `{touched_index}`.",
            ]
        )
    )

    dod = create_definition_of_done("Create an external nginx guide.")
    dod.implementation_plan = str(implementation_plan)

    targets = collect_planned_artifact_targets(dod, project_root=tmp_path)

    assert (tmp_path / "chapters", True) not in targets
    assert (tmp_path / "01-introduction.html", False) not in targets
    assert targets == [(touched_index, False)]


def test_all_planned_artifacts_exist_requires_file_contents_for_planned_output_directory(
    tmp_path: Path,
) -> None:
    implementation_plan = tmp_path / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{tmp_path / 'guide' / 'index.html'}`",
                f"- `{tmp_path / 'guide' / 'chapters'}/` (directory for chapter files)",
                "",
                "## Execution Order",
                "- Create chapter files with appropriate content",
            ]
        )
    )

    guide_root = tmp_path / "guide"
    chapters = guide_root / "chapters"
    guide_root.mkdir()
    chapters.mkdir()
    (guide_root / "index.html").write_text("<html></html>\n")

    dod = create_definition_of_done("Create a multi-file guide with chapters.")
    dod.implementation_plan = str(implementation_plan)
    dod.completed_items = ["Create chapter files with appropriate content"]

    assert all_planned_artifacts_exist(dod, project_root=tmp_path) is False

    (chapters / "01-getting-started.html").write_text("<h1>Intro</h1>\n")

    assert all_planned_artifacts_exist(dod, project_root=tmp_path) is True


def test_all_planned_artifacts_exist_stays_false_while_touched_html_links_missing(
    tmp_path: Path,
) -> None:
    implementation_plan = tmp_path / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{tmp_path / 'guide' / 'index.html'}`",
                f"- `{tmp_path / 'guide' / 'chapters'}/` (directory for chapter files)",
                "",
                "## Execution Order",
                "- Create chapter files with appropriate content",
            ]
        )
    )

    guide_root = tmp_path / "guide"
    chapters = guide_root / "chapters"
    guide_root.mkdir()
    chapters.mkdir()
    index = guide_root / "index.html"
    index.write_text(
        '<a href="chapters/01-introduction.html">Intro</a>\n'
        '<a href="chapters/02-setup.html">Setup</a>\n'
    )
    (chapters / "01-introduction.html").write_text("<h1>Intro</h1>\n")

    dod = create_definition_of_done("Create a multi-file guide with chapters.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files = [str(index), str(chapters / "01-introduction.html")]
    dod.completed_items = ["Create chapter files with appropriate content"]

    assert all_planned_artifacts_exist(dod, project_root=tmp_path) is False

    (chapters / "02-setup.html").write_text("<h1>Setup</h1>\n")

    assert all_planned_artifacts_exist(dod, project_root=tmp_path) is True


def test_build_verification_summary_keeps_concrete_missing_link_details() -> None:
    summary = build_verification_summary(
        [
            VerificationEvidence(
                command="python3 - <<'PY' ... PY",
                passed=False,
                stderr=(
                    "Missing links:\n"
                    "chapters/05-control-structures.html -> missing\n"
                    "chapters/06-input-output.html -> missing\n"
                ),
            )
        ]
    )

    assert "Missing links:" in summary
    assert "chapters/05-control-structures.html -> missing" in summary
    assert "chapters/06-input-output.html -> missing" in summary
