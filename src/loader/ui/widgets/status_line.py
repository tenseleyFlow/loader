"""Status line widget showing model, activity, time, tokens."""

from textual.reactive import reactive
from textual.widgets import Static


class StatusLine(Static):
    """Status bar showing model info, activity, elapsed time, and tokens."""

    model: reactive[str] = reactive("")
    mode: reactive[str] = reactive("Native")
    activity: reactive[str] = reactive("")
    elapsed: reactive[float] = reactive(0.0)
    tokens: reactive[int] = reactive(0)

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

        # Model info
        if self.model:
            parts.append(f"[blue]{self.model}[/blue]")

        # Mode
        if self.mode:
            parts.append(f"[dim]{self.mode}[/dim]")

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
