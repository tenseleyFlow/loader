"""Modal screen for structured AskUserQuestion prompts."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class QuestionModal(ModalScreen[str | None]):
    """Collect a single user answer for AskUserQuestion."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "submit", "Submit", show=False),
    ]

    CSS = """
    QuestionModal {
        align: center middle;
    }

    #question-dialog {
        width: 70;
        height: auto;
        border: heavy $primary;
        background: $surface;
        padding: 1 2;
    }

    #question-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #question-prompt {
        margin-bottom: 1;
    }

    #question-options {
        height: auto;
        margin-bottom: 1;
    }

    #question-options Button {
        margin: 0 1 1 0;
    }

    #question-input {
        margin-bottom: 1;
    }

    #question-actions {
        align: right middle;
        height: auto;
    }

    #question-actions Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        question: str,
        options: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.question = question
        self.options = list(options or [])

    def compose(self) -> ComposeResult:
        with Vertical(id="question-dialog"):
            yield Static(
                "[bold cyan]? Clarification Needed[/bold cyan]",
                id="question-title",
            )
            yield Static(self.question, id="question-prompt")
            if self.options:
                with Horizontal(id="question-options"):
                    for index, option in enumerate(self.options, start=1):
                        yield Button(f"{index}. {option}", id=f"option-{index}")
            yield Input(
                placeholder="Type your answer and press Enter",
                id="question-input",
            )
            with Horizontal(id="question-actions"):
                yield Button("Submit", id="submit", variant="success")
                yield Button("Cancel", id="cancel", variant="error")

    def on_mount(self) -> None:
        self.query_one("#question-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "submit":
            self._submit_current_input()
            return
        if button_id == "cancel":
            self.dismiss(None)
            return
        if button_id.startswith("option-"):
            index = int(button_id.split("-", maxsplit=1)[1]) - 1
            if 0 <= index < len(self.options):
                self.dismiss(self.options[index])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_submit(self) -> None:
        self._submit_current_input()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _submit_current_input(self) -> None:
        value = self.query_one("#question-input", Input).value.strip()
        self.dismiss(value or None)
