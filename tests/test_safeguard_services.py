"""Tests for runtime-owned safeguard services."""

from __future__ import annotations

import loader.agent.safeguards as agent_safeguards
from loader.agent.safeguards import RuntimeSafeguards as AgentRuntimeSafeguards
from loader.runtime.safeguard_services import (
    ActionTracker,
    PreActionValidator,
    ValidationResult,
)
from loader.runtime.safeguards import RuntimeSafeguards


def test_action_tracker_detects_duplicate_write_after_recording(tmp_path) -> None:
    tracker = ActionTracker()
    file_path = tmp_path / "notes.txt"
    arguments = {"file_path": str(file_path), "content": "alpha\n"}

    assert tracker.check_tool_call("write", arguments) == (False, "")

    tracker.record_tool_call("write", arguments)

    is_duplicate, reason = tracker.check_tool_call("write", arguments)

    assert is_duplicate is True
    assert str(file_path) in reason


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
        suggestion="Provide one or more structured patch hunks",
        severity="error",
    )


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
