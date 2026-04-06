"""Lightweight CLI rendering helpers."""

from __future__ import annotations

from ..runtime.events import AgentEvent


def format_dod_status(event: AgentEvent) -> str:
    """Format a definition-of-done status event for CLI output."""
    parts = [f"DoD: {event.dod_status or 'unknown'}"]
    if event.pending_items_count is not None:
        parts.append(f"{event.pending_items_count} pending")
    if event.last_verification_result:
        parts.append(f"last verify: {event.last_verification_result}")
    return " | ".join(parts)
