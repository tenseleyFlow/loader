"""Tests for runtime-owned safeguard services."""

from __future__ import annotations

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
