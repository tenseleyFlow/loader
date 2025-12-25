"""Input area widget with '>' prompt and history support."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.events import Key
from textual.message import Message
from textual.widgets import Input, Static


class InputArea(Horizontal):
    """Fixed input area at bottom of screen with '>' prompt and history."""

    class Submitted(Message):
        """Message sent when user submits input."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._history_index: int = -1  # -1 = new input, 0+ = history position
        self._current_input: str = ""  # Saves current input when navigating

    def compose(self) -> ComposeResult:
        yield Static("> ", classes="prompt")
        yield Input(placeholder="Type a message...", id="user-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if value:
            # Add to history (avoid duplicates of last entry)
            if not self._history or self._history[-1] != value:
                self._history.append(value)
            # Reset history navigation
            self._history_index = -1
            self._current_input = ""
            self.post_message(self.Submitted(value))
            event.input.clear()

    def on_key(self, event: Key) -> None:
        """Handle up/down arrow keys for history navigation."""
        if not self._history:
            return

        input_widget = self.query_one("#user-input", Input)

        if event.key == "up":
            event.prevent_default()
            event.stop()

            # Save current input when starting to navigate
            if self._history_index == -1:
                self._current_input = input_widget.value

            # Navigate up through history
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                # History is stored oldest-first, so we index from the end
                history_value = self._history[-(self._history_index + 1)]
                input_widget.value = history_value
                input_widget.cursor_position = len(history_value)

        elif event.key == "down":
            event.prevent_default()
            event.stop()

            if self._history_index > -1:
                self._history_index -= 1

                if self._history_index == -1:
                    # Back to current input
                    input_widget.value = self._current_input
                    input_widget.cursor_position = len(self._current_input)
                else:
                    history_value = self._history[-(self._history_index + 1)]
                    input_widget.value = history_value
                    input_widget.cursor_position = len(history_value)

    def focus_input(self) -> None:
        """Focus the input field."""
        self.query_one("#user-input", Input).focus()
