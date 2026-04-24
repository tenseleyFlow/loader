"""Tests for response-repair helpers on RuntimeContext."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from loader.llm.base import Message, Role, ToolCall
from loader.runtime.context import RuntimeContext
from loader.runtime.dod import create_definition_of_done
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.repair import ResponseRepairer
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeSession:
    def __init__(self) -> None:
        self.messages = []

    def append(self, message) -> None:
        self.messages.append(message)


class FakeCodeFilter:
    def reset(self) -> None:
        return None


class FakeSafeguards:
    def __init__(self) -> None:
        self.action_tracker = object()
        self.validator = object()
        self.code_filter = FakeCodeFilter()

    def filter_stream_chunk(self, content: str) -> str:
        return content

    def filter_complete_content(self, content: str) -> str:
        return content

    def should_steer(self) -> bool:
        return False

    def get_steering_message(self) -> str | None:
        return None

    def record_response(self, content: str) -> None:
        return None

    def detect_text_loop(self, content: str) -> tuple[bool, str]:
        return False, ""

    def detect_loop(self) -> tuple[bool, str]:
        return False, ""


def build_context(
    *,
    temp_dir: Path,
    use_react: bool,
) -> RuntimeContext:
    registry = create_default_registry(temp_dir)
    registry.configure_workspace_root(temp_dir)
    rule_status = load_permission_rules(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
        rules=rule_status.rules,
    )
    session = FakeSession()
    return RuntimeContext(
        project_root=temp_dir,
        backend=ScriptedBackend(),
        registry=registry,
        session=session,  # type: ignore[arg-type]
        config=SimpleNamespace(force_react=use_react),
        capability_profile=SimpleNamespace(supports_native_tools=not use_react),  # type: ignore[arg-type]
        project_context=None,
        permission_policy=policy,
        permission_config_status=rule_status,
        workflow_mode="execute",
        safeguards=FakeSafeguards(),
    )


def test_response_repairer_uses_runtime_parser_for_bracket_tool_fallback(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    analysis = repairer.analyze_response(
        content="I need clarification.",
        response_content='[calls askuserquestion tool with: question="Which path?"]',
        tool_calls=[],
        extracted_iterations=0,
        max_extracted_iterations=3,
    )

    assert analysis.tool_calls == [
        ToolCall(
            id="call_0",
            name="AskUserQuestion",
            arguments={"question": "Which path?"},
        )
    ]
    assert analysis.tool_source == "raw_text"
    assert analysis.clear_stream is True


def test_response_repairer_recovers_todowrite_from_runtime_registry(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    analysis = repairer.analyze_response(
        content="I'll track the work first.",
        response_content=json.dumps(
            {
                "name": "TodoWrite",
                "arguments": {
                    "todos": [
                        {
                            "content": "Run tests",
                            "active_form": "Running tests",
                            "status": "in_progress",
                        }
                    ]
                },
            }
        ),
        tool_calls=[],
        extracted_iterations=0,
        max_extracted_iterations=3,
    )

    assert analysis.tool_source == "raw_text"
    assert analysis.clear_stream is True
    assert analysis.tool_calls == [
        ToolCall(
            id="call_0",
            name="TodoWrite",
            arguments={
                "todos": [
                    {
                        "content": "Run tests",
                        "active_form": "Running tests",
                        "status": "in_progress",
                    }
                ]
            },
        )
    ]


def test_response_repairer_fails_honestly_when_raw_tool_budget_is_exhausted(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    analysis = repairer.analyze_response(
        content=json.dumps(
            {
                "name": "read",
                "arguments": {"file_path": "README.md"},
            }
        ),
        response_content=json.dumps(
            {
                "name": "read",
                "arguments": {"file_path": "README.md"},
            }
        ),
        tool_calls=[],
        extracted_iterations=3,
        max_extracted_iterations=3,
    )

    assert analysis.should_stop is True
    assert analysis.final_response == (
        "I couldn't safely continue because the model kept emitting raw-text "
        "tool calls instead of proper tool invocations. Please try again or "
        "switch to a different backend/model."
    )
    assert analysis.failure == "raw-text tool recovery budget exhausted"
    assert "Let me know if you'd like me to continue" not in analysis.final_response


def test_empty_response_retry_message_surfaces_missing_planned_artifacts_and_working_note(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'guides' / 'nginx' / 'index.html'}`",
                f"- `{temp_dir / 'guides' / 'nginx' / 'chapters' / '01-getting-started.html'}`",
                f"- `{temp_dir / 'guides' / 'nginx' / 'chapters' / '02-installation.html'}`",
                "",
            ]
        )
    )
    first_artifact = temp_dir / "guides" / "nginx" / "index.html"
    first_artifact.parent.mkdir(parents=True)
    first_artifact.write_text("<html></html>\n")

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(first_artifact))
    dod.completed_items.append("Create the main index.html file")
    dod.pending_items.append("Create each chapter file in sequence")

    context.session.append(
        SimpleNamespace(
            role="tool",
            content=(
                "Observation [notepad_write_working]: Result: "
                "- [2026-04-21T19:17:34Z] Creating fifth chapter file: Advanced configurations"
            ),
        )
    )

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert "Latest working note: Creating fifth chapter file: Advanced configurations" in decision.retry_message
    assert "Next missing planned artifact: `01-getting-started.html`" in decision.retry_message
    assert "Remaining planned artifacts: `01-getting-started.html`, `02-installation.html`" in decision.retry_message
    assert "Resume with this exact next step: create `01-getting-started.html`." in decision.retry_message
    assert f"Prefer one `write` call for `{temp_dir / 'guides' / 'nginx' / 'chapters' / '01-getting-started.html'}` before any more reference reads." in decision.retry_message
    assert (
        "Shape the next response as one concrete `write(file_path=..., content=...)` "
        "tool call for that exact path."
        in decision.retry_message
    )
    assert (
        "Your next response should be the concrete mutation tool call itself, "
        "not TodoWrite alone, verification, or a completion summary."
        in decision.retry_message
    )
    assert "Do not restart discovery unless one specific missing fact blocks this step." in decision.retry_message


def test_empty_response_retry_mentions_write_can_create_missing_parent_directories(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    index_path = guide_root / "index.html"

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{index_path}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items.extend(
        [
            "Create nginx guide directory structure",
            "Write main index.html for nginx guide",
        ]
    )

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert (
        "Resume with this exact next step: continue `Write main index.html for nginx guide` "
        "by creating `index.html`."
        in decision.retry_message
    )
    assert (
        f"Prefer one `write(content=...)` call for `{index_path}` before more research."
        in decision.retry_message
    )
    assert (
        "Do not restart discovery unless one specific missing fact blocks that file write."
        in decision.retry_message
    )


def test_empty_response_retry_respects_discovery_first_pending_step(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'guides' / 'nginx' / 'index.html'}`",
                f"- `{temp_dir / 'guides' / 'nginx' / 'chapters'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items.extend(
        [
            "First, examine the existing fortran guide structure and content to understand the format",
            "Create the nginx directory structure",
            "Develop the main index.html file for the nginx guide",
        ]
    )

    context.session.append(
        SimpleNamespace(
            role="tool",
            content=(
                "Observation [notepad_write_working]: Result: "
                "- [2026-04-22T22:42:18Z] Analyzing the fortran guide structure before creating nginx guide"
            ),
        )
    )

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert (
        "Resume with this exact next step: advance `First, examine the existing fortran guide structure and content to understand the format`."
        in decision.retry_message
    )
    assert "one concrete evidence-gathering tool call" in decision.retry_message
    assert "Resume with this exact next step: create `index.html`." not in decision.retry_message


def test_empty_response_retry_budget_extends_for_late_stage_multi_artifact_progress(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    chapter_three = chapters / "03-first-website.html"
    chapter_four = chapters / "04-configuration-basics.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")
    chapter_three.write_text("<h1>Three</h1>\n")

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
                f"- `{chapter_four}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend(
        [str(index_path), str(chapter_one), str(chapter_two), str(chapter_three)]
    )
    dod.completed_items.extend(
        [
            "Create the directory structure for the new nginx guide",
            "Create the main index.html file with proper structure",
        ]
    )
    dod.pending_items.append("Create each chapter file in sequence")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=3,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert "retry 3/4" in decision.retry_message
    assert "Follow the same one-file-at-a-time mutation pattern" in decision.retry_message


def test_empty_response_retry_uses_compact_prompt_after_substantial_progress(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    context.session.messages.append(
        SimpleNamespace(
            content=(
                "Observation [notepad_write_working]: Result: "
                "- [2026-04-23T19:00:00Z] Creating fifth chapter file: Advanced features"
            )
        )
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    chapter_three = chapters / "03-first-website.html"
    chapter_four = chapters / "04-configuration-basics.html"
    chapter_five = chapters / "05-advanced-features.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")
    chapter_three.write_text("<h1>Three</h1>\n")
    chapter_four.write_text("<h1>Four</h1>\n")

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
                f"- `{chapter_four}`",
                f"- `{chapter_five}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend(
        [str(index_path), str(chapter_one), str(chapter_two), str(chapter_three)]
    )
    dod.completed_items.extend(
        [
            "Create the directory structure for the new nginx guide",
            "Create the main index.html file with proper structure",
        ]
    )
    dod.pending_items.append("Create each chapter file in sequence")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=3,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert "Continue from the exact next step below." in decision.retry_message
    assert "Latest working note:" not in decision.retry_message
    assert "Confirmed completed work:" not in decision.retry_message
    assert "Next pending item:" not in decision.retry_message


def test_empty_response_retry_points_at_next_output_file_when_planned_directory_is_empty(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    index_path.write_text("<html></html>\n")

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
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(index_path))
    dod.pending_items.append("Write the introduction chapter")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert "Next missing planned artifact: `chapters/`" in decision.retry_message
    assert (
        "Resume with this exact next step: continue `Write the introduction chapter` "
        "by creating the next output file under `chapters/`."
        in decision.retry_message
    )
    assert (
        f"Prefer one concrete `write` call for a file inside `{chapters}` before more research."
        in decision.retry_message
    )


def test_empty_response_retry_treats_develop_index_step_as_mutation_work(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    chapter_one = chapters / "01-introduction.html"
    index_path = guide_root / "index.html"

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{index_path}`",
                f"- `{chapters}/`",
                f"- `{chapter_one}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.completed_items.extend(
        [
            "First, examine the existing Fortran guide structure to understand the format and depth",
            "Create the new nginx guide directory structure",
        ]
    )
    dod.pending_items.append("Develop the main index.html file with proper structure")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=2,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert (
        "Resume with this exact next step: continue `Develop the main index.html file with proper structure`"
        in decision.retry_message
    )
    assert "Prefer one `write(content=...)` call" in decision.retry_message
    assert "Make the next response one concrete evidence-gathering tool call" not in decision.retry_message


def test_empty_response_retry_prefers_output_index_over_reference_index_with_same_name(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    nginx_root = temp_dir / "Loader" / "guides" / "nginx"
    fortran_root = temp_dir / "Loader" / "guides" / "fortran"
    nginx_root.mkdir(parents=True)
    fortran_root.mkdir(parents=True)
    reference_index = fortran_root / "index.html"
    reference_index.write_text("<html>fortran</html>\n")
    output_index = nginx_root / "index.html"

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{output_index}`",
                f"- `{nginx_root / 'chapters'}/`",
                f"- `{reference_index}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(reference_index))
    dod.completed_items.append(
        "First, examine the existing Fortran guide structure and content"
    )
    dod.pending_items.append("Develop the nginx index.html file")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=2,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert (
        f"Prefer one `write(content=...)` call for `{output_index}` before more research."
        in decision.retry_message
    )
    assert str(reference_index) not in decision.retry_message


def test_empty_response_retry_points_at_declared_child_file_within_incomplete_output_directory(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    index_path.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/introduction.html">Introduction</a>',
                '<a href="chapters/installation.html">Installation</a>',
                "</html>",
            ]
        )
        + "\n"
    )

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
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(index_path))
    dod.pending_items.append("Write the introduction chapter")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert "Next missing planned artifact: `chapters/`" in decision.retry_message
    assert "Next declared output under `chapters/`: `introduction.html`" in decision.retry_message
    assert (
        "Resume with this exact next step: continue `Write the introduction chapter` "
        "by creating `introduction.html`."
        in decision.retry_message
    )
    assert (
        f"Prefer one `write(content=...)` call for `{(chapters / 'introduction.html').resolve(strict=False)}` "
        "before more research."
        in decision.retry_message
    )


def test_empty_response_retry_infers_concrete_file_from_pending_todo_after_broad_artifacts_exist(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-introduction.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<html></html>\n")

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
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend([str(index_path), str(chapter_one)])
    dod.completed_items.extend(
        [
            "Create index.html for nginx guide",
            "Create first chapter file (01-introduction.html)",
        ]
    )
    dod.pending_items.append("Create second chapter file (02-installation.html)")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=2,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert (
        "Resume with this exact next step: continue `Create second chapter file "
        "(02-installation.html)` by creating `02-installation.html`."
        in decision.retry_message
    )
    assert (
        f"Prefer one `write(content=...)` call for `{chapters / '02-installation.html'}` "
        "before more research."
        in decision.retry_message
    )
    assert "Do not return another working note or empty response" in decision.retry_message


def test_empty_response_retry_maps_title_style_todo_to_html_graph_target(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-introduction.html"
    index_path.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/01-introduction.html">Chapter 1: Introduction to NGINX Tool</a>',
                '<a href="chapters/02-installation.html">Chapter 2: Installation and Setup</a>',
                "</html>",
            ]
        )
        + "\n"
    )
    chapter_one.write_text("<html></html>\n")

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
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend([str(index_path), str(chapter_one)])
    dod.completed_items.extend(
        [
            "Create index.html for nginx guide",
            "Create Chapter 1: Introduction to NGINX Tool",
        ]
    )
    dod.pending_items.append("Creating Chapter 2: Installation and Setup")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=2,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert (
        "Resume with this exact next step: continue `Creating Chapter 2: Installation and Setup` "
        "by creating `02-installation.html`."
        in decision.retry_message
    )
    assert (
        f"Prefer one `write(content=...)` call for `{(chapters / '02-installation.html').resolve(strict=False)}` "
        "before more research."
        in decision.retry_message
    )


def test_empty_response_retry_uses_compact_prompt_after_early_progress_with_concrete_next_file(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-introduction.html"
    index_path.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/01-introduction.html">Introduction</a>',
                '<a href="chapters/02-installation.html">Installation</a>',
                "</html>",
            ]
        )
        + "\n"
    )
    chapter_one.write_text("<html></html>\n")

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
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend([str(index_path), str(chapter_one)])
    dod.completed_items.extend(
        [
            "Create index.html for nginx guide",
            "Create first chapter file (01-introduction.html)",
        ]
    )
    dod.pending_items.append("Create second chapter file (02-installation.html)")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert "Continue from the exact next step below." in decision.retry_message
    assert "Confirmed completed work:" not in decision.retry_message
    assert "Next pending item:" not in decision.retry_message
    assert (
        "Resume with this exact next step: continue `Create second chapter file "
        "(02-installation.html)` by creating `02-installation.html`."
        in decision.retry_message
    )


def test_empty_response_retry_fails_after_extended_late_stage_budget_is_exhausted(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    chapter_three = chapters / "03-first-website.html"
    chapter_four = chapters / "04-configuration-basics.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")
    chapter_three.write_text("<h1>Three</h1>\n")

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
                f"- `{chapter_four}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend(
        [str(index_path), str(chapter_one), str(chapter_two), str(chapter_three)]
    )
    dod.completed_items.extend(
        [
            "Create the directory structure for the new nginx guide",
            "Create the main index.html file with proper structure",
        ]
    )
    dod.pending_items.append("Create each chapter file in sequence")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=5,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is False
    assert decision.final_response is not None
    assert "retrying 4 times" in decision.final_response


def test_empty_response_retry_mentions_todowrite_when_progress_has_outpaced_tracking(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root / 'index.html'}`",
                f"- `{chapters / '01-getting-started.html'}`",
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend(
        [
            str(guide_root / "index.html"),
            str(chapters / "01-getting-started.html"),
        ]
    )
    dod.completed_items.extend(
        [
            "Create the directory structure for the new nginx guide",
            "Create the main index.html file with proper structure",
        ]
    )
    dod.pending_items.append("Create each chapter file in sequence")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.retry_message is not None
    assert (
        "refresh `TodoWrite` alongside the next concrete mutation"
        in decision.retry_message
    )


def test_empty_response_retry_omits_stale_aggregate_completed_work_when_artifacts_missing(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
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
    dod.touched_files.extend([str(index_path), str(chapter_one), str(chapter_two)])
    dod.completed_items.extend(
        [
            "Create the main index.html file with proper structure",
            "Link all chapters together properly",
        ]
    )
    dod.pending_items.append("Create each chapter file in sequence")

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.retry_message is not None
    assert "Link all chapters together properly" not in decision.retry_message
    assert "Continue from the exact next step below." in decision.retry_message
    assert "Resume with this exact next step:" in decision.retry_message


def test_empty_response_retry_names_next_file_from_observed_sibling_directory(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
    )
    repairer = ResponseRepairer(context)

    reference_chapters = temp_dir / "fortran" / "chapters"
    reference_chapters.mkdir(parents=True)
    (reference_chapters / "01-introduction.html").write_text("<h1>Introduction</h1>\n")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    index_path.write_text("<html></html>\n")

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
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(index_path))
    dod.pending_items.append("Write the introduction chapter")
    context.session.append(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=[
                ToolCall(
                    id="read-ref-1",
                    name="read",
                    arguments={"file_path": str(reference_chapters / "01-introduction.html")},
                )
            ],
        )
    )

    decision = repairer.handle_empty_response(
        task="Create a multi-file nginx guide.",
        original_task=None,
        empty_retry_count=1,
        max_empty_retries=2,
        dod=dod,
    )

    assert decision.should_continue is True
    assert decision.retry_message is not None
    assert "Next missing planned artifact: `chapters/`" in decision.retry_message
    assert "Next observed output pattern under `chapters/`: `01-introduction.html`" in decision.retry_message
    assert (
        "Resume with this exact next step: continue `Write the introduction chapter` "
        "by creating `01-introduction.html`."
        in decision.retry_message
    )
    assert (
        "It mirrors the observed filename pattern from another `chapters/` directory "
        "you already inspected."
        in decision.retry_message
    )
