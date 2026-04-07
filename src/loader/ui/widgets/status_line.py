"""Status line widget showing model, activity, time, tokens."""

from textual.reactive import reactive
from textual.widgets import Static

from ..status_helpers import format_definition_of_done_parts, format_permission_mode_part


class StatusLine(Static):
    """Status bar showing model info, activity, elapsed time, and tokens."""

    model: reactive[str] = reactive("")
    mode: reactive[str] = reactive("Native")
    permission_mode: reactive[str] = reactive("")
    activity: reactive[str] = reactive("")
    elapsed: reactive[float] = reactive(0.0)
    tokens: reactive[int] = reactive(0)
    dod_status: reactive[str] = reactive("")
    pending_items_count: reactive[int] = reactive(0)
    last_verification_result: reactive[str] = reactive("")

    def render(self) -> str:
        """Render the status line."""
        parts = []

        # Activity indicator (with spinner)
        if self.activity:
            parts.append(f"[yellow]*[/yellow] [bold]{self.activity}[/bold]")

        # Elapsed time (if active)
        if self.elapsed > 0:
            parts.append(f"[cyan]{self.elapsed:.1f}s[/cyan]")

        # Token count
        if self.tokens > 0:
            parts.append(f"[dim]{self.tokens} tokens[/dim]")

        parts.extend(
            format_definition_of_done_parts(
                self.dod_status,
                self.pending_items_count,
                self.last_verification_result,
            )
        )

        # Model info
        if self.model:
            parts.append(f"[blue]{self.model}[/blue]")

        # Mode
        if self.mode:
            parts.append(f"[dim]{self.mode}[/dim]")
        permission_mode = format_permission_mode_part(self.permission_mode)
        if permission_mode:
            parts.append(permission_mode)

        return " · ".join(parts) if parts else "[dim]Ready[/dim]"

    def watch_activity(self, activity: str) -> None:
        """React to activity changes."""
        self.refresh()

    def watch_elapsed(self, elapsed: float) -> None:
        """React to elapsed time changes."""
        self.refresh()

    def watch_tokens(self, tokens: int) -> None:
        """React to token count changes."""
        self.refresh()

    def watch_permission_mode(self, permission_mode: str) -> None:
        """React to permission mode changes."""
        self.refresh()

    def watch_dod_status(self, dod_status: str) -> None:
        """React to DoD status changes."""
        self.refresh()

    def watch_pending_items_count(self, pending_items_count: int) -> None:
        """React to DoD pending item changes."""
        self.refresh()

    def watch_last_verification_result(self, last_verification_result: str) -> None:
        """React to verification result changes."""
        self.refresh()

    def set_generating(self, is_generating: bool) -> None:
        """Set generating state."""
        if is_generating:
            self.activity = "Generating..."
        else:
            self.activity = ""
            self.elapsed = 0.0

    def update_elapsed(self, elapsed: float) -> None:
        """Update elapsed time."""
        self.elapsed = elapsed

    def update_tokens(self, tokens: int) -> None:
        """Update token count."""
        self.tokens = tokens

    def update_permission_mode(self, permission_mode: str) -> None:
        """Update the active permission mode."""
        self.permission_mode = permission_mode

    def update_definition_of_done(
        self,
        status: str,
        pending_items_count: int,
        last_verification_result: str | None,
    ) -> None:
        """Update definition-of-done status."""
        self.dod_status = status
        self.pending_items_count = pending_items_count
        self.last_verification_result = last_verification_result or ""

    def clear_definition_of_done(self) -> None:
        """Clear definition-of-done status."""
        self.dod_status = ""
        self.pending_items_count = 0
        self.last_verification_result = ""
