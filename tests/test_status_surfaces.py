"""Tests for user-visible definition-of-done status formatting."""

from loader.cli.rendering import format_dod_status
from loader.runtime.events import AgentEvent
from loader.ui.status_helpers import format_definition_of_done_parts


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
