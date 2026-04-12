"""Persistent todo list widget rendered above the input area."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

_STATUS_ICONS = {
    "pending": (" [ ] ", "dim"),
    "in_progress": (" [~] ", "bold yellow"),
    "completed": (" [x] ", "green"),
}


class TodoListWidget(Widget):
    """Renders the agent's current todo list with checkboxes and strikethrough."""

    DEFAULT_CSS = """
    TodoListWidget {
        height: auto;
        max-height: 10;
        display: none;
        padding: 0 1;
        border-top: solid $primary-darken-2;
    }

    TodoListWidget.has-items {
        display: block;
    }

    TodoListWidget #todo-content {
        width: 100%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._items: list[dict[str, str]] = []

    def compose(self) -> ComposeResult:
        yield Static("", id="todo-content")

    def update_todos(self, todos: list[dict[str, str]]) -> None:
        """Replace the displayed todo list."""
        self._items = list(todos)
        if not self._items:
            self.remove_class("has-items")
            return
        self.add_class("has-items")
        self._render()

    def _render(self) -> None:
        content = Text()
        content.append(" Tasks\n", style="bold")
        for item in self._items:
            status = item.get("status", "pending")
            icon, icon_style = _STATUS_ICONS.get(status, _STATUS_ICONS["pending"])
            label = item.get("content", "")

            content.append(icon, style=icon_style)
            if status == "completed":
                content.append(label, style="strike dim")
            elif status == "in_progress":
                active = item.get("active_form", label)
                content.append(active, style="bold")
            else:
                content.append(label)
            content.append("\n")

        self.query_one("#todo-content", Static).update(content)
