"""Tool call widget with collapsible result preview."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Collapsible, Static


class ToolCallWidget(Vertical):
    """Collapsible widget for tool calls with result preview."""

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

    def compose(self) -> ComposeResult:
        # Format args for display
        args_str = self._format_args()

        yield Static(
            f"{self.TOOL_BULLETS['pending']} [bold cyan]{self.tool_name}[/bold cyan]({args_str})",
            id="tool-header",
            classes="tool-header",
        )
        yield Static("", id="tool-summary", classes="tool-summary")
        yield Collapsible(
            Static("", id="tool-full-result"),
            title="Show full output",
            collapsed=True,
            id="tool-collapsible",
        )

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

        # Update summary
        lines = result.splitlines()
        if len(lines) <= self.preview_lines:
            # Show all content in summary, hide collapsible
            summary = result
            self.query_one("#tool-collapsible", Collapsible).display = False
        else:
            # Show preview with expand hint
            preview = "\n".join(lines[: self.preview_lines])
            summary = f"{preview}\n[dim]... ({len(lines) - self.preview_lines} more lines)[/dim]"
            self.query_one("#tool-full-result", Static).update(result)

        self.query_one("#tool-summary", Static).update(summary)

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
