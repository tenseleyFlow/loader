"""Status line widget showing model, activity, time, tokens."""

from textual.reactive import reactive
from textual.widgets import Static

from ..status_helpers import (
    format_capability_part,
    format_definition_of_done_parts,
    format_permission_mode_part,
    format_runtime_owner_part,
    format_session_part,
    format_turn_phase_part,
    format_workflow_mode_part,
)


class StatusLine(Static):
    """Status bar showing model info, activity, elapsed time, and tokens."""

    model: reactive[str] = reactive("")
    mode: reactive[str] = reactive("Native")
    capability_profile: reactive[str] = reactive("")
    session_id: reactive[str] = reactive("")
    runtime_owner: reactive[str] = reactive("")
    workflow_mode: reactive[str] = reactive("")
    turn_phase: reactive[str] = reactive("")
    permission_mode: reactive[str] = reactive("")
    activity: reactive[str] = reactive("")
    elapsed: reactive[float] = reactive(0.0)
    tokens: reactive[int] = reactive(0)
    dod_status: reactive[str] = reactive("")
    pending_items_count: reactive[int] = reactive(0)
    last_verification_result: reactive[str] = reactive("")
    verification_attempt: reactive[str] = reactive("")

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
                self.verification_attempt,
            )
        )

        # Model info
        if self.model:
            parts.append(f"[blue]{self.model}[/blue]")

        # Mode
        if self.mode:
            parts.append(f"[dim]{self.mode}[/dim]")
        capability_profile = format_capability_part(self.capability_profile)
        if capability_profile:
            parts.append(capability_profile)
        workflow_mode = format_workflow_mode_part(self.workflow_mode)
        if workflow_mode:
            parts.append(workflow_mode)
        turn_phase = format_turn_phase_part(self.turn_phase)
        if turn_phase:
            parts.append(turn_phase)
        permission_mode = format_permission_mode_part(self.permission_mode)
        if permission_mode:
            parts.append(permission_mode)
        session_part = format_session_part(self.session_id)
        if session_part:
            parts.append(session_part)
        runtime_owner = format_runtime_owner_part(self.runtime_owner)
        if runtime_owner:
            parts.append(runtime_owner)

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

    def watch_capability_profile(self, capability_profile: str) -> None:
        """React to capability profile changes."""
        self.refresh()

    def watch_session_id(self, session_id: str) -> None:
        """React to session id changes."""
        self.refresh()

    def watch_runtime_owner(self, runtime_owner: str) -> None:
        """React to runtime owner changes."""
        self.refresh()

    def watch_workflow_mode(self, workflow_mode: str) -> None:
        """React to workflow mode changes."""
        self.refresh()

    def watch_turn_phase(self, turn_phase: str) -> None:
        """React to turn phase changes."""
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

    def watch_verification_attempt(self, verification_attempt: str) -> None:
        """React to verification attempt changes."""
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

    def update_capability_profile(self, capability_profile: str) -> None:
        """Update the active capability profile."""
        self.capability_profile = capability_profile

    def update_session_id(self, session_id: str) -> None:
        """Update the active session id."""
        self.session_id = session_id

    def update_runtime_owner(self, runtime_owner: str) -> None:
        """Update the active runtime owner path."""
        self.runtime_owner = runtime_owner

    def update_workflow_mode(self, workflow_mode: str) -> None:
        """Update the active workflow mode."""
        self.workflow_mode = workflow_mode

    def update_turn_phase(self, turn_phase: str) -> None:
        """Update the active turn phase."""
        self.turn_phase = turn_phase

    def update_definition_of_done(
        self,
        status: str,
        pending_items_count: int,
        last_verification_result: str | None,
        verification_attempt: str | None = None,
    ) -> None:
        """Update definition-of-done status."""
        self.dod_status = status
        self.pending_items_count = pending_items_count
        self.last_verification_result = last_verification_result or ""
        self.verification_attempt = verification_attempt or ""

    def clear_definition_of_done(self) -> None:
        """Clear definition-of-done status."""
        self.dod_status = ""
        self.pending_items_count = 0
        self.last_verification_result = ""
        self.verification_attempt = ""
