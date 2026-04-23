"""Focused tests for workflow recovery priority rules."""

from __future__ import annotations

from pathlib import Path

from loader.runtime.workflow_recovery import _should_prioritize_missing_artifact


def test_workflow_recovery_prioritizes_missing_artifact_over_review_step() -> None:
    missing_artifact = (Path("/tmp/guide/06-ssl-configuration.html"), False)

    assert _should_prioritize_missing_artifact(
        next_pending="Ensure all files are properly linked and formatted consistently",
        missing_artifact=missing_artifact,
    )
    assert not _should_prioritize_missing_artifact(
        next_pending="Create the final chapter (06-ssl-configuration.html)",
        missing_artifact=missing_artifact,
    )
