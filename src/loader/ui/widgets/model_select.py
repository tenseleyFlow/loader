"""Model selection modal with fuzzy filtering."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


class ModelSelectModal(ModalScreen[str | None]):
    """Modal screen for selecting an Ollama model with fuzzy filtering."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "select", "Select", show=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
    ]

    CSS = """
    ModelSelectModal {
        align: center middle;
    }

    #model-dialog {
        width: 70;
        height: 24;
        border: heavy $primary;
        background: $surface;
        padding: 1 2;
    }

    #model-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        text-align: center;
    }

    #model-filter {
        margin-bottom: 1;
    }

    #model-list {
        height: 14;
        border: round $primary-darken-2;
    }

    #model-list > .option-list--option-highlighted {
        background: $accent;
        color: $text;
    }

    #model-hint {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }

    .model-current {
        color: $success;
    }
    """

    def __init__(
        self,
        models: list[dict],
        current_model: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.models = models
        self.current_model = current_model
        self._all_options: list[tuple[str, str]] = []  # (display, value)

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Static(
                "[bold cyan]Select Model[/bold cyan]",
                id="model-title",
            )
            yield Input(placeholder="Type to filter...", id="model-filter")
            yield OptionList(id="model-list")
            yield Static(
                "[dim]↑↓/jk: navigate  Enter: select  Esc: cancel[/dim]",
                id="model-hint",
            )

    def on_mount(self) -> None:
        """Populate the model list on mount."""
        self._build_options()
        self._update_list("")
        # Focus the filter input
        self.query_one("#model-filter", Input).focus()

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable form."""
        if size_bytes >= 1_000_000_000:
            return f"{size_bytes / 1_000_000_000:.1f}GB"
        elif size_bytes >= 1_000_000:
            return f"{size_bytes / 1_000_000:.0f}MB"
        else:
            return f"{size_bytes / 1_000:.0f}KB"

    def _build_options(self) -> None:
        """Build the list of model options."""
        # Sort by size (largest first)
        sorted_models = sorted(self.models, key=lambda m: m.get("size", 0), reverse=True)

        for model in sorted_models:
            name = model.get("name", "unknown")
            size = self._format_size(model.get("size", 0))

            if name == self.current_model:
                display = f"[green]● {name}[/green] [dim]({size})[/dim] [yellow]← current[/yellow]"
            else:
                display = f"  {name} [dim]({size})[/dim]"

            self._all_options.append((display, name))

    def _update_list(self, filter_text: str) -> None:
        """Update the option list based on filter."""
        option_list = self.query_one("#model-list", OptionList)
        option_list.clear_options()

        filter_lower = filter_text.lower()
        matched = []

        for display, value in self._all_options:
            # Fuzzy match: all filter chars must appear in order
            if self._fuzzy_match(filter_lower, value.lower()):
                matched.append((display, value))

        for display, value in matched:
            option_list.add_option(Option(display, id=value))

        # Highlight current model if in list, otherwise first item
        if matched:
            # Try to find and highlight current model
            for i, (_, value) in enumerate(matched):
                if value == self.current_model:
                    option_list.highlighted = i
                    break
            else:
                option_list.highlighted = 0

    def _fuzzy_match(self, pattern: str, text: str) -> bool:
        """Simple fuzzy matching - all chars must appear in order."""
        if not pattern:
            return True
        pattern_idx = 0
        for char in text:
            if char == pattern[pattern_idx]:
                pattern_idx += 1
                if pattern_idx == len(pattern):
                    return True
        return False

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter the list as user types."""
        if event.input.id == "model-filter":
            self._update_list(event.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection."""
        if event.option.id:
            self.dismiss(str(event.option.id))

    def action_cancel(self) -> None:
        """Cancel selection."""
        self.dismiss(None)

    def action_select(self) -> None:
        """Select the highlighted option."""
        option_list = self.query_one("#model-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option and option.id:
                self.dismiss(str(option.id))

    def action_cursor_up(self) -> None:
        """Move cursor up in the list."""
        option_list = self.query_one("#model-list", OptionList)
        option_list.action_cursor_up()

    def action_cursor_down(self) -> None:
        """Move cursor down in the list."""
        option_list = self.query_one("#model-list", OptionList)
        option_list.action_cursor_down()
