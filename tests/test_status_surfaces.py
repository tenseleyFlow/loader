"""Tests for user-visible definition-of-done status formatting."""

from loader.cli.rendering import (
    format_dod_status,
    format_permission_mode,
    format_workflow_mode,
)
from loader.runtime.events import AgentEvent
from loader.ui.status_helpers import (
    format_definition_of_done_parts,
    format_permission_mode_part,
    format_workflow_mode_part,
)


def test_status_helper_formats_definition_of_done_parts() -> None:
    parts = format_definition_of_done_parts("verifying", 1, "failed")

    assert parts == [
        "[yellow]DoD: verifying[/yellow]",
        "[dim]1 pending[/dim]",
        "[red]verify failed[/red]",
    ]


def test_status_helper_omits_definition_of_done_when_absent() -> None:
    assert format_definition_of_done_parts("", 0, "") == []


def test_cli_dod_status_format_includes_pending_and_verification() -> None:
    event = AgentEvent(
        type="dod_status",
        dod_status="fixing",
        pending_items_count=2,
        last_verification_result="failed",
    )

    formatted = format_dod_status(event)

    assert formatted == "DoD: fixing | 2 pending | last verify: failed"


def test_permission_mode_helpers_use_expected_colors() -> None:
    assert format_permission_mode_part("read-only") == "[green]perm read-only[/green]"
    assert format_permission_mode("danger-full-access") == "[red]danger-full-access[/red]"


def test_workflow_mode_helpers_use_expected_colors() -> None:
    assert format_workflow_mode_part("plan") == "[cyan]flow plan[/cyan]"
    assert format_workflow_mode("verify") == "[green]verify[/green]"
