"""Input area widget with '>' prompt, history, and shadow text suggestions."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.events import Key
from textual.message import Message
from textual.widgets import Input, Static


# Available slash commands for suggestions
SLASH_COMMANDS = [
    "/help",
    "/model",
    "/models",
    "/clear",
    "/exit",
]


class InputArea(Horizontal):
    """Fixed input area at bottom of screen with '>' prompt, history, and shadow text."""

    DEFAULT_CSS = """
    InputArea {
        height: auto;
        width: 100%;
    }

    InputArea .prompt {
        width: auto;
        padding: 0 1 0 0;
    }

    InputArea #input-wrapper {
        width: 1fr;
    }

    InputArea #user-input {
        width: 100%;
        background: transparent;
    }

    InputArea #shadow-text {
        color: $text-disabled;
        layer: below;
        offset: 0 0;
    }
    """

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
        self._suggestion: str = ""  # Current shadow text suggestion

    def compose(self) -> ComposeResult:
        yield Static("> ", classes="prompt")
        yield Input(placeholder="Type a message...", id="user-input")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for shadow text suggestions."""
        value = event.value
        self._suggestion = ""

        # Suggest slash commands
        if value.startswith("/") and len(value) > 0:
            for cmd in SLASH_COMMANDS:
                if cmd.startswith(value) and cmd != value:
                    self._suggestion = cmd[len(value):]  # Just the completion part
                    break

        # Update placeholder to show suggestion
        input_widget = event.input
        if self._suggestion:
            # Show the suggestion as part of placeholder
            input_widget.placeholder = f"{self._suggestion}  (Tab to complete)"
        else:
            input_widget.placeholder = "Type a message..."

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
            self._suggestion = ""
            self.post_message(self.Submitted(value))
            event.input.clear()
            event.input.placeholder = "Type a message..."

    def on_key(self, event: Key) -> None:
        """Handle special keys: Tab for completion, Up/Down for history."""
        input_widget = self.query_one("#user-input", Input)

        # Tab to accept suggestion
        if event.key == "tab" and self._suggestion:
            event.prevent_default()
            event.stop()
            new_value = input_widget.value + self._suggestion
            input_widget.value = new_value
            input_widget.cursor_position = len(new_value)
            self._suggestion = ""
            input_widget.placeholder = "Type a message..."
            return

        if not self._history:
            return

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
