"""Formatting helpers for user-visible runtime status."""

from __future__ import annotations


def format_definition_of_done_parts(
    status: str,
    pending_items_count: int,
    last_verification_result: str,
) -> list[str]:
    """Format definition-of-done state for the status line."""
    if not status:
        return []

    dod_color = {
        "draft": "cyan",
        "in_progress": "cyan",
        "verifying": "yellow",
        "fixing": "magenta",
        "done": "green",
        "failed": "red",
    }.get(status, "white")
    parts = [f"[{dod_color}]DoD: {status}[/{dod_color}]"]
    parts.append(f"[dim]{pending_items_count} pending[/dim]")

    if last_verification_result:
        verify_color = "green" if last_verification_result == "passed" else "red"
        if last_verification_result == "skipped":
            verify_color = "dim"
        parts.append(
            f"[{verify_color}]verify {last_verification_result}[/{verify_color}]"
        )

    return parts
