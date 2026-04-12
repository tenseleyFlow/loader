"""Tool call widget with collapsible result preview."""

from rich.markup import escape
from rich.text import Text

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static, Button


class ToolCallWidget(Vertical):
    """Widget for tool calls with expandable result preview."""

    TOOL_BULLETS = {
        "pending": "[yellow]○[/yellow]",
        "running": "[yellow]◐[/yellow]",
        "success": "[green]●[/green]",
        "error": "[red]●[/red]",
    }

    state: reactive[str] = reactive("pending")
    expanded: reactive[bool] = reactive(False)

    def __init__(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        preview_lines: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.preview_lines = preview_lines
        self._result: str = ""
        self._is_error: bool = False
        self._full_result: str = ""
        self._has_more: bool = False

    def compose(self) -> ComposeResult:
        # Format args for display
        args_str = self._format_args()

        yield Static(
            f"{self.TOOL_BULLETS['pending']} [bold cyan]{self.tool_name}[/bold cyan]({args_str})",
            id="tool-header",
            classes="tool-header",
        )
        yield Static("", id="tool-summary", classes="tool-summary")

        # Toggle button for expand/collapse (hidden by default until result has more lines)
        toggle = Button("▶ Show full output", id="tool-toggle", classes="tool-toggle", variant="default")
        toggle.display = False
        yield toggle

        full_result = Static("", id="tool-full-result", classes="tool-full-result")
        full_result.display = False
        yield full_result

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle toggle button press."""
        if event.button.id == "tool-toggle":
            if self.expanded:
                self.action_collapse()
            else:
                self.action_expand()
            event.stop()

    def _format_args(self) -> str:
        """Format tool arguments for display."""
        if not self.tool_args:
            return ""
        parts = []
        for k, v in self.tool_args.items():
            if isinstance(v, str):
                # Truncate long strings
                if len(v) > 40:
                    v = v[:37] + "..."
                parts.append(f'{k}="[dim]{v}[/dim]"')
            else:
                parts.append(f"{k}={v!r}")
        return ", ".join(parts)

    def set_running(self) -> None:
        """Mark as running."""
        self.state = "running"
        self._update_header()

    def set_result(self, result: str, is_error: bool = False) -> None:
        """Update widget with tool result."""
        self._result = result
        self._is_error = is_error
        self.state = "error" if is_error else "success"

        # Update styling
        self.remove_class("pending", "error", "success")
        self.add_class(self.state)

        # Update header
        self._update_header()

        # Build summary as a Rich Text object to avoid markup parsing errors
        # when tool results contain brackets or other Rich-like syntax
        summary = Text()
        if is_error:
            summary.append("✗ Failed\n", style="bold red")
        else:
            summary.append("✓ Success\n", style="bold green")

        lines = result.splitlines()

        toggle_widget = self.query_one("#tool-toggle", Button)
        full_result_widget = self.query_one("#tool-full-result", Static)

        if len(lines) <= self.preview_lines:
            summary.append(result)
            toggle_widget.display = False
            full_result_widget.display = False
            self._has_more = False
        else:
            preview = "\n".join(lines[: self.preview_lines])
            remaining = len(lines) - self.preview_lines
            summary.append(preview)
            summary.append(f"\n... ({remaining} more lines)", style="dim")

            self._full_result = escape(result)
            self._has_more = True
            self._update_toggle()
            toggle_widget.display = True
            full_result_widget.display = False

        self.query_one("#tool-summary", Static).update(summary)

    def _update_toggle(self) -> None:
        """Update the expand/collapse toggle button label."""
        toggle = self.query_one("#tool-toggle", Button)
        if self.expanded:
            toggle.label = "▼ Hide full output"
        else:
            toggle.label = "▶ Show full output"

    def action_expand(self) -> None:
        """Expand to show full output."""
        self.expanded = True
        self._update_toggle()
        full_result = self.query_one("#tool-full-result", Static)
        full_result.update(self._full_result)
        full_result.display = True

    def action_collapse(self) -> None:
        """Collapse to hide full output."""
        self.expanded = False
        self._update_toggle()
        self.query_one("#tool-full-result", Static).display = False

    def _update_header(self) -> None:
        """Update the header with current state."""
        args_str = self._format_args()
        bullet = self.TOOL_BULLETS.get(self.state, self.TOOL_BULLETS["pending"])
        color = "red" if self._is_error else "cyan"
        self.query_one("#tool-header", Static).update(
            f"{bullet} [bold {color}]{self.tool_name}[/bold {color}]({args_str})"
        )

    def watch_state(self, state: str) -> None:
        """React to state changes."""
        self.remove_class("pending", "running", "success", "error")
        self.add_class(state)
