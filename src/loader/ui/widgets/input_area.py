"""Input area widget with '>' prompt."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Input, Static


class InputArea(Horizontal):
    """Fixed input area at bottom of screen with '>' prompt."""

    class Submitted(Message):
        """Message sent when user submits input."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("> ", classes="prompt")
        yield Input(placeholder="Type a message...", id="user-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if value:
            self.post_message(self.Submitted(value))
            event.input.clear()

    def focus_input(self) -> None:
        """Focus the input field."""
        self.query_one("#user-input", Input).focus()
