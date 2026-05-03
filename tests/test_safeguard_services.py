"""Tests for runtime-owned safeguard services."""

from __future__ import annotations

import tempfile
from pathlib import Path

import loader.agent.safeguards as agent_safeguards
from loader.agent.safeguards import RuntimeSafeguards as AgentRuntimeSafeguards
from loader.runtime.safeguard_services import (
    ActionTracker,
    PreActionValidator,
    ValidationResult,
)
from loader.runtime.safeguards import RuntimeSafeguards
from loader.runtime.semantic_rules.html_toc import (
    build_html_toc_edit_call_template,
    build_html_toc_replacement_block,
    format_html_inventory_entry,
    task_targets_html_toc,
    validate_html_toc,
)


def test_action_tracker_detects_duplicate_write_after_recording(tmp_path) -> None:
    tracker = ActionTracker()
    file_path = tmp_path / "notes.txt"
    arguments = {"file_path": str(file_path), "content": "alpha\n"}

    assert tracker.check_tool_call("write", arguments) == (False, "")

    tracker.record_tool_call("write", arguments)

    is_duplicate, reason = tracker.check_tool_call("write", arguments)

    assert is_duplicate is True
    assert str(file_path) in reason


def test_task_targets_html_toc_requires_explicit_repair_intent() -> None:
    prompt = (
        "Have a look at ~/Loader/guides/fortran and chapters/ within. Get a feel "
        "for the structure and cadence of the guide. We are going to make an all "
        "new equally thorough guide on how to use the nginx tool. It will live in "
        "~/Loader/guides/nginx/index.html and ~/Loader/guides/nginx/chapters/."
    )

    assert task_targets_html_toc(prompt) is False


def test_build_html_toc_replacement_block_uses_verified_inventory(tmp_path) -> None:
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    (chapters / "02-setup.html").write_text(
        "<h1>Chapter 2: Setting Up Your Environment</h1>\n"
    )
    index_path = tmp_path / "index.html"
    index_path.write_text(
        "<h2>Table of Contents</h2>\n"
        "<ul class=\"chapter-list\">\n"
        "    <li><a href=\"chapters/01-old.html\">Chapter 1: Old</a></li>\n"
        "</ul>\n"
    )

    replacement = build_html_toc_replacement_block(index_path)

    assert replacement is not None
    assert "<h2>Table of Contents</h2>" in replacement
    assert '<li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>' in replacement
    assert '<li><a href="chapters/02-setup.html">Chapter 2: Setting Up Your Environment</a></li>' in replacement


def test_build_html_toc_edit_call_template_uses_current_and_replacement_blocks(tmp_path) -> None:
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    index_path = tmp_path / "index.html"
    index_path.write_text(
        "<h2>Table of Contents</h2>\n"
        '<ul class="chapter-list">\n'
        '    <li><a href="chapters/01-old.html">Chapter 1: Old</a></li>\n'
        "</ul>\n"
    )

    template = build_html_toc_edit_call_template(index_path)

    assert template is not None
    assert template.startswith("edit(")
    assert f'file_path="{index_path}"' in template
    assert 'old_string="""' in template
    assert 'new_string="""' in template
    assert '<li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>' in template


def test_validate_html_toc_reports_missing_and_mismatched_links(tmp_path) -> None:
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    index_path = tmp_path / "index.html"
    index_path.write_text(
        '<ul class="chapter-list">\n'
        '    <li><a href="chapters/01-introduction.html">Chapter 1: Wrong Title</a></li>\n'
        '    <li><a href="chapters/02-missing.html">Chapter 2: Missing</a></li>\n'
        "</ul>\n"
    )

    result = validate_html_toc(index_path)

    assert result is not None
    assert result.valid is False
    assert result.link_count == 2
    assert result.missing == ("chapters/02-missing.html -> missing",)
    assert (
        result.mismatched
        == (
            "chapters/01-introduction.html -> Chapter 1: Wrong Title != Chapter 1: Introduction to Fortran",
        )
    )


def test_validate_html_toc_reports_success(tmp_path) -> None:
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    (chapters / "02-setup.html").write_text(
        "<h1>Chapter 2: Setting Up Your Environment</h1>\n"
    )
    index_path = tmp_path / "index.html"
    index_path.write_text(
        '<ul class="chapter-list">\n'
        '    <li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>\n'
        '    <li><a href="chapters/02-setup.html">Chapter 2: Setting Up Your Environment</a></li>\n'
        "</ul>\n"
    )

    result = validate_html_toc(index_path)

    assert result is not None
    assert result.valid is True
    assert result.link_count == 2
    assert result.missing == ()
    assert result.mismatched == ()


def test_action_tracker_preserves_loop_description_format() -> None:
    tracker = ActionTracker()

    tracker.record_tool_call("read", {"file_path": "a.txt"})
    tracker.record_tool_call("grep", {"pattern": "alpha"})
    tracker.record_tool_call("read", {"file_path": "b.txt"})
    tracker.record_tool_call("grep", {"pattern": "beta"})

    is_loop, description = tracker.detect_loop()

    assert is_loop is True
    assert description == "Repeating pattern detected (2x): read → grep"


def test_action_tracker_blocks_repeated_bash_observation_without_changes() -> None:
    tracker = ActionTracker()
    arguments = {"command": "ls -la ~/Loader/guides/fortran/chapters/"}

    tracker.record_tool_call("bash", arguments)

    is_duplicate, reason = tracker.check_tool_call("bash", arguments)

    assert is_duplicate is True
    assert "read-only shell probe" in reason


def test_action_tracker_allows_repeated_bash_observation_after_mutation() -> None:
    tracker = ActionTracker()
    bash_args = {"command": "ls -la ~/Loader/guides/fortran/chapters/"}
    patch_args = {
        "file_path": "index.html",
        "hunks": [
            {
                "old_start": 1,
                "old_lines": 1,
                "new_start": 1,
                "new_lines": 1,
                "lines": ["-old", "+new"],
            }
        ],
    }

    tracker.record_tool_call("bash", bash_args)
    tracker.record_tool_call("patch", patch_args)

    assert tracker.check_tool_call("bash", bash_args) == (False, "")


def test_action_tracker_blocks_repeated_read_without_changes(tmp_path) -> None:
    tracker = ActionTracker()
    file_path = tmp_path / "index.html"
    arguments = {"file_path": str(file_path)}

    tracker.record_tool_call("read", arguments)

    is_duplicate, reason = tracker.check_tool_call("read", arguments)

    assert is_duplicate is True
    assert str(file_path) in reason


def test_action_tracker_allows_one_interleaved_reread_without_changes(tmp_path) -> None:
    tracker = ActionTracker()
    index_path = tmp_path / "index.html"
    chapter_path = tmp_path / "chapter-1.html"

    tracker.record_tool_call("read", {"file_path": str(index_path)})
    tracker.record_tool_call("read", {"file_path": str(chapter_path)})

    assert tracker.check_tool_call("read", {"file_path": str(index_path)}) == (False, "")


def test_action_tracker_allows_reading_a_different_slice_of_the_same_file(tmp_path) -> None:
    tracker = ActionTracker()
    index_path = tmp_path / "index.html"

    tracker.record_tool_call("read", {"file_path": str(index_path)})

    assert tracker.check_tool_call(
        "read",
        {"file_path": str(index_path), "offset": 1, "limit": 50},
    ) == (False, "")


def test_action_tracker_blocks_fourth_interleaved_reread_without_changes(tmp_path) -> None:
    tracker = ActionTracker()
    index_path = tmp_path / "index.html"
    chapter_a = tmp_path / "chapter-1.html"
    chapter_b = tmp_path / "chapter-2.html"
    chapter_c = tmp_path / "chapter-3.html"

    tracker.record_tool_call("read", {"file_path": str(index_path)})
    tracker.record_tool_call("read", {"file_path": str(chapter_a)})
    tracker.record_tool_call("read", {"file_path": str(index_path)})
    tracker.record_tool_call("read", {"file_path": str(chapter_b)})
    tracker.record_tool_call("read", {"file_path": str(index_path)})
    tracker.record_tool_call("read", {"file_path": str(chapter_c)})

    is_duplicate, reason = tracker.check_tool_call("read", {"file_path": str(index_path)})

    assert is_duplicate is True
    assert str(index_path) in reason


def test_action_tracker_allows_one_target_index_reread_after_chapter_discovery(tmp_path) -> None:
    tracker = ActionTracker()
    index_path = tmp_path / "index.html"
    chapters = tmp_path / "chapters"
    chapter_a = chapters / "01-introduction.html"
    chapter_b = chapters / "02-setup.html"
    chapter_c = chapters / "03-basics.html"

    tracker.record_tool_call("read", {"file_path": str(index_path)})
    tracker.record_tool_call("read", {"file_path": str(chapter_a)})
    tracker.record_tool_call("read", {"file_path": str(chapter_b)})
    tracker.record_tool_call("read", {"file_path": str(chapter_c)})

    is_duplicate, reason = tracker.check_tool_call("read", {"file_path": str(index_path)})

    assert is_duplicate is False
    assert reason == ""


def test_action_tracker_blocks_second_target_index_reread_after_chapter_discovery(tmp_path) -> None:
    tracker = ActionTracker()
    index_path = tmp_path / "index.html"
    chapters = tmp_path / "chapters"

    tracker.record_tool_call("read", {"file_path": str(index_path)})
    tracker.record_tool_call("read", {"file_path": str(chapters / "01-introduction.html")})
    tracker.record_tool_call("read", {"file_path": str(chapters / "02-setup.html")})
    tracker.record_tool_call("read", {"file_path": str(chapters / "03-basics.html")})
    tracker.record_tool_call("read", {"file_path": str(index_path)})

    is_duplicate, reason = tracker.check_tool_call("read", {"file_path": str(index_path)})

    assert is_duplicate is True
    assert "reuse the earlier read result instead of rereading" in reason


def test_action_tracker_blocks_repeated_chapter_directory_search_once_titles_are_known(
    tmp_path,
) -> None:
    tracker = ActionTracker()
    chapters = tmp_path / "chapters"
    search_args = {"pattern": "*.html", "path": str(chapters)}

    tracker.record_tool_call("glob", search_args)
    tracker.record_tool_call("glob", search_args)

    is_duplicate, reason = tracker.check_tool_call("glob", search_args)

    assert is_duplicate is True
    assert "reuse the earlier search result instead of rerunning it" in reason


def test_action_tracker_allows_repeated_read_after_mutation(tmp_path) -> None:
    tracker = ActionTracker()
    file_path = tmp_path / "index.html"
    read_args = {"file_path": str(file_path)}
    edit_args = {
        "file_path": str(file_path),
        "old_string": "old",
        "new_string": "new",
    }

    tracker.record_tool_call("read", read_args)
    tracker.record_tool_call("edit", edit_args)

    assert tracker.check_tool_call("read", read_args) == (False, "")


def test_pre_action_validator_blocks_patch_without_hunks() -> None:
    validator = PreActionValidator()

    result = validator.validate(
        "patch",
        {"file_path": "notes.txt", "hunks": []},
    )

    assert result == ValidationResult(
        valid=False,
        reason="Patch hunks are missing",
        suggestion="Provide structured patch hunks or a unified diff patch string",
        severity="error",
    )


def test_pre_action_validator_allows_patch_string_without_hunks() -> None:
    validator = PreActionValidator()

    result = validator.validate(
        "patch",
        {
            "file_path": "notes.txt",
            "patch": "--- a/notes.txt\n+++ b/notes.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n",
        },
    )

    assert result == ValidationResult(valid=True)


def test_pre_action_validator_blocks_shell_text_rewrite_for_html_target() -> None:
    validator = PreActionValidator()

    result = validator.validate(
        "bash",
        {
            "command": (
                "cd /tmp/fortran-qwen-recovery-check && "
                "sed -i '1,3c\\<li>updated</li>' index.html"
            )
        },
    )

    assert result.valid is False
    assert result.reason == (
        "Shell-based text rewrites are brittle and bypass Loader's safer file tools"
    )
    assert "edit/patch/write" in result.suggestion
    assert "index.html" in result.suggestion


def test_pre_action_validator_allows_non_mutating_sed_probe() -> None:
    validator = PreActionValidator()

    result = validator.validate(
        "bash",
        {"command": "sed -n '1,20p' index.html"},
    )

    assert result == ValidationResult(valid=True)


def test_pre_action_validator_blocks_index_edit_with_missing_chapter_href(tmp_path) -> None:
    validator = PreActionValidator()
    index = tmp_path / "index.html"
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "05-input-output.html").write_text(
        "<h1>Chapter 5: Input and Output</h1>\n"
    )

    result = validator.validate(
        "edit",
        {
            "file_path": str(index),
            "old_string": '<li><a href="chapters/05-input-output.html">Chapter 5: Input and Output</a></li>',
            "new_string": '<li><a href="chapters/05-control-structures.html">Chapter 5: Control Structures</a></li>',
        },
    )

    assert result.valid is False
    assert result.reason == "Edited HTML links point to files that do not exist"
    assert "chapters/05-control-structures.html" in result.suggestion


def test_pre_action_validator_allows_incomplete_root_index_to_reshape_missing_child_target(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    index = guide / "index.html"
    index.write_text(
        "\n".join(
            [
                '<li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Nginx</a></li>',
                '<li><a href="chapters/02-installation.html">Chapter 2: Installation on POSIX Systems</a></li>',
                '<li><a href="chapters/03-configuration-basics.html">Chapter 3: Configuration Basics</a></li>',
                "",
            ]
        )
    )
    (chapters / "01-introduction.html").write_text("<html></html>\n")
    (chapters / "02-installation.html").write_text("<html></html>\n")

    result = validator.validate(
        "edit",
        {
            "file_path": str(index),
            "old_string": '<li><a href="chapters/03-configuration-basics.html">Chapter 3: Configuration Basics</a></li>',
            "new_string": '<li><a href="chapters/03-configuration.html">Chapter 3: Configuration Basics</a></li>',
        },
    )

    assert result.valid is True


def test_pre_action_validator_blocks_index_edit_with_title_mismatch(tmp_path) -> None:
    validator = PreActionValidator()
    index = tmp_path / "index.html"
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "12-troubleshooting-tips.html").write_text(
        "<h1>Chapter 12: Troubleshooting and Tips</h1>\n"
    )

    result = validator.validate(
        "edit",
        {
            "file_path": str(index),
            "old_string": '<li><a href="chapters/12-troubleshooting-tips.html">Chapter 12: Troubleshooting and Tips</a></li>',
            "new_string": '<li><a href="chapters/12-troubleshooting-tips.html">Chapter 12: Troubleshooting Tips</a></li>',
        },
    )

    assert result.valid is True


def test_pre_action_validator_allows_chapter_write_with_future_target_declared_by_index(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    (guide / "index.html").write_text(
        "\n".join(
            [
                '<a href="chapters/introduction.html">Introduction</a>',
                '<a href="chapters/installation.html">Installation</a>',
                "",
            ]
        )
    )

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "introduction.html"),
            "content": '<a href="installation.html">Next</a>\n',
        },
    )

    assert result.valid is True


def test_pre_action_validator_blocks_undeclared_chapter_file_creation_after_root_seed(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    (guide / "index.html").write_text(
        "\n".join(
            [
                '<a href="chapters/01-introduction.html">Introduction</a>',
                '<a href="chapters/02-installation.html">Installation</a>',
                '<a href="chapters/03-configuration.html">Configuration</a>',
                "",
            ]
        )
    )
    (chapters / "01-introduction.html").write_text("<html></html>\n")
    (chapters / "02-installation.html").write_text("<html></html>\n")

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "09-monitoring.html"),
            "content": "<html></html>\n",
        },
    )

    assert result.valid is False
    assert result.reason == "HTML file creation falls outside the current declared artifact set"
    assert "09-monitoring.html" in result.suggestion
    assert str((guide / "index.html").resolve(strict=False)) in result.suggestion


def test_pre_action_validator_allows_declared_missing_chapter_file_creation(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    (guide / "index.html").write_text(
        "\n".join(
            [
                '<a href="chapters/01-introduction.html">Introduction</a>',
                '<a href="chapters/02-installation.html">Installation</a>',
                '<a href="chapters/03-configuration.html">Configuration</a>',
                "",
            ]
        )
    )
    (chapters / "01-introduction.html").write_text("<html></html>\n")

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "02-installation.html"),
            "content": "<html></html>\n",
        },
    )

    assert result.valid is True


def test_pre_action_validator_blocks_chapter_write_with_undeclared_missing_sibling(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    (guide / "index.html").write_text(
        "\n".join(
            [
                '<a href="chapters/introduction.html">Introduction</a>',
                '<a href="chapters/installation.html">Installation</a>',
                '<a href="chapters/configuration.html">Configuration</a>',
                '<a href="chapters/usage.html">Usage</a>',
                '<a href="chapters/troubleshooting.html">Troubleshooting</a>',
                "",
            ]
        )
    )
    (chapters / "introduction.html").write_text('<a href="installation.html">Next</a>\n')
    (chapters / "installation.html").write_text('<a href="configuration.html">Next</a>\n')
    (chapters / "configuration.html").write_text('<a href="usage.html">Next</a>\n')

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "usage.html"),
            "content": '<a href="advanced.html">Next</a>\n',
        },
    )

    assert result.valid is False
    assert (
        result.reason
        == "HTML page introduces new local targets outside the current declared artifact set"
    )
    assert "advanced.html" in result.suggestion


def test_pre_action_validator_blocks_chapter_write_with_existing_but_undeclared_sibling(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    (guide / "index.html").write_text(
        "\n".join(
            [
                '<a href="chapters/01-introduction.html">Introduction</a>',
                '<a href="chapters/02-installation.html">Installation</a>',
                '<a href="chapters/03-basic-configuration.html">Basic Configuration</a>',
                '<a href="chapters/04-advanced-configuration.html">Advanced Configuration</a>',
                "",
            ]
        )
    )
    (chapters / "01-introduction.html").write_text('<a href="02-installation.html">Next</a>\n')
    (chapters / "02-installation.html").write_text(
        '<a href="03-basic-configuration.html">Next</a>\n'
    )
    (chapters / "04-locations-and-servers.html").write_text(
        '<a href="05-static-content.html">Next</a>\n'
    )

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "03-basic-configuration.html"),
            "content": '<a href="04-locations-and-servers.html">Next</a>\n',
        },
    )

    assert result.valid is False
    assert (
        result.reason
        == "HTML page introduces new local targets outside the current declared artifact set"
    )
    assert "04-locations-and-servers.html" in result.suggestion
    assert "04-advanced-configuration.html" in result.suggestion


def test_pre_action_validator_does_not_suggest_unrelated_declared_html_target(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    (guide / "index.html").write_text(
        "\n".join(
            [
                '<a href="chapters/introduction.html">Introduction</a>',
                '<a href="chapters/installation.html">Installation</a>',
                '<a href="chapters/configuration.html">Configuration</a>',
                '<a href="chapters/basic-usage.html">Basic Usage</a>',
                '<a href="chapters/advanced-topics.html">Advanced Topics</a>',
                "",
            ]
        )
    )

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "introduction.html"),
            "content": '<a href="troubleshooting.html">Troubleshooting</a>\n',
        },
    )

    assert result.valid is False
    assert "troubleshooting.html" in result.suggestion
    assert "Closest declared local targets include:" not in result.suggestion


def test_pre_action_validator_allows_chapter_write_with_root_declared_sibling_and_index_link(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    guide = tmp_path / "guide"
    chapters = guide / "chapters"
    chapters.mkdir(parents=True)
    (guide / "index.html").write_text(
        "\n".join(
            [
                '<a href="chapters/01-introduction.html">Introduction</a>',
                '<a href="chapters/02-installation.html">Installation</a>',
                '<a href="chapters/03-basic-configuration.html">Basic Configuration</a>',
                "",
            ]
        )
    )

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "02-installation.html"),
            "content": (
                '<a href="../index.html">Back to guide</a>\n'
                '<a href="03-basic-configuration.html">Next</a>\n'
            ),
        },
    )

    assert result.valid is True


def test_pre_action_validator_blocks_missing_numbered_read_with_existing_sibling(
    tmp_path: Path,
) -> None:
    validator = PreActionValidator()
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01-getting-started.html").write_text("<h1>Getting Started</h1>\n")

    result = validator.validate(
        "read",
        {"file_path": str(chapters / "01-introduction.html")},
    )

    assert result.valid is False
    assert result.reason == "Read target conflicts with an existing numbered sibling"
    assert "01-getting-started.html" in result.suggestion


def test_pre_action_validator_blocks_new_numbered_sibling_drift(tmp_path) -> None:
    validator = PreActionValidator()
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01-getting-started.html").write_text("<h1>Getting Started</h1>\n")

    result = validator.validate(
        "write",
        {
            "file_path": str(chapters / "01-intro.html"),
            "content": "<h1>Intro</h1>\n",
        },
    )

    assert result.valid is False
    assert result.reason == "New file conflicts with an existing numbered sibling"
    assert "01-getting-started.html" in result.suggestion


def test_format_html_inventory_entry_handles_tmp_alias_paths() -> None:
    root = Path(tempfile.mkdtemp(dir="/tmp"))
    chapters = root / "chapters"
    chapters.mkdir()
    candidate = chapters / "05-input-output.html"
    candidate.write_text("<h1>Chapter 5: Input and Output</h1>\n")

    entry = format_html_inventory_entry(root, candidate.resolve(strict=False))

    assert entry == "chapters/05-input-output.html = Chapter 5: Input and Output"


def test_runtime_safeguards_wrap_runtime_owned_services() -> None:
    safeguards = RuntimeSafeguards()

    assert isinstance(safeguards.action_tracker, ActionTracker)
    assert isinstance(safeguards.validator, PreActionValidator)


def test_agent_safeguards_reexport_runtime_safeguards() -> None:
    assert AgentRuntimeSafeguards is RuntimeSafeguards


def test_agent_safeguards_exports_curated_compatibility_surface() -> None:
    assert agent_safeguards.__all__ == [
        "ActionTracker",
        "CodeBlockFilter",
        "FilterResult",
        "PatternDetector",
        "PatternMatch",
        "PreActionValidator",
        "RuntimeSafeguards",
        "ValidationResult",
    ]
