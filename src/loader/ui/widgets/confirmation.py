"""Confirmation modal for destructive tool operations."""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmationModal(ModalScreen[bool]):
    """Modal screen for confirming destructive operations."""

    BINDINGS = [
        Binding("y", "confirm", "Yes", show=True),
        Binding("n", "deny", "No", show=True),
        Binding("escape", "deny", "Cancel", show=False),
    ]

    CSS = """
    ConfirmationModal {
        align: center middle;
    }

    #confirmation-dialog {
        width: 60;
        height: auto;
        border: heavy $primary;
        background: $surface;
        padding: 1 2;
    }

    #confirmation-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #confirmation-message {
        margin-bottom: 1;
    }

    #confirmation-details {
        color: $text-muted;
        margin-bottom: 1;
        padding: 0 1;
        border-left: solid $primary-lighten-2;
    }

    #confirmation-buttons {
        align: center middle;
        height: auto;
        margin-top: 1;
    }

    #confirmation-buttons Button {
        margin: 0 1;
    }

    #btn-yes {
        background: $success;
    }

    #btn-no {
        background: $error;
    }
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        details: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.message = message
        self.details = details

    def compose(self) -> ComposeResult:
        with Vertical(id="confirmation-dialog"):
            yield Static(
                Text(f"⚠ Confirm {self.tool_name}", style="bold yellow"),
                id="confirmation-title",
                markup=False,
            )
            yield Static(Text(self.message), id="confirmation-message", markup=False)
            if self.details:
                yield Static(
                    Text(self.details, style="dim"),
                    id="confirmation-details",
                    markup=False,
                )
            with Horizontal(id="confirmation-buttons"):
                yield Button("Yes (y)", id="btn-yes", variant="success")
                yield Button("No (n)", id="btn-no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "btn-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_confirm(self) -> None:
        """Handle 'y' key."""
        self.dismiss(True)

    def action_deny(self) -> None:
        """Handle 'n' or escape."""
        self.dismiss(False)
