"""Tests for CLI resume argument rewriting."""

from __future__ import annotations

from loader.cli.options import inject_resume_target


def test_inject_resume_target_supports_flag_and_named_session() -> None:
    assert inject_resume_target([]) == []
    assert inject_resume_target(["--resume"]) == ["--resume-target", "__latest__"]
    assert inject_resume_target(["--resume", "session-123"]) == [
        "--resume-target",
        "session-123",
    ]
    assert inject_resume_target(["--resume", "session-123", "fix runtime"]) == [
        "--resume-target",
        "session-123",
        "fix runtime",
    ]
