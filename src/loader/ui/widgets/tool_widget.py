"""Tool call widget with inline truncation (claw-code style)."""

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

# Display truncation limits
TOOL_RESULT_MAX_LINES = 60
TOOL_RESULT_MAX_CHARS = 6000
WRITE_PREVIEW_MAX_LINES = 80
READ_DISPLAY_MAX_LINES = 80
_TRUNCATION_NOTICE = "truncated for display; full result preserved in session"


class ToolCallWidget(Vertical):
    """Widget for tool calls with inline content display."""

    TOOL_BULLETS = {
        "pending": "[yellow]○[/yellow]",
        "running": "[yellow]◐[/yellow]",
        "success": "[green]●[/green]",
        "error": "[red]●[/red]",
    }

    state: reactive[str] = reactive("pending")

    def __init__(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self._result: str = ""
        self._is_error: bool = False

    def compose(self) -> ComposeResult:
        args_str = self._format_args()

        yield Static(
            f"{self.TOOL_BULLETS['pending']} [bold cyan]{self.tool_name}[/bold cyan]({args_str})",
            id="tool-header",
            classes="tool-header",
        )

        # For write/edit tools, show content as pre-approval preview
        initial_summary = Text()
        if self.tool_name in ("write", "edit", "patch"):
            content = self.tool_args.get("content", "")
            file_path = self.tool_args.get("file_path", "")
            if content and file_path:
                initial_summary.append(f"  ► {file_path}\n", style="bold")
                lines = content.splitlines()
                show = min(len(lines), WRITE_PREVIEW_MAX_LINES)
                for i, line in enumerate(lines[:show]):
                    initial_summary.append(f"  {i + 1:>4} ", style="dim")
                    initial_summary.append(f"{line}\n")
                if len(lines) > show:
                    initial_summary.append(
                        f"  … {len(lines) - show} more lines "
                        f"({_TRUNCATION_NOTICE})\n",
                        style="dim",
                    )
        yield Static(initial_summary, id="tool-summary", classes="tool-summary")

    def _format_args(self) -> str:
        """Format tool arguments for display."""
        if not self.tool_args:
            return ""
        parts = []
        for k, v in self.tool_args.items():
            if isinstance(v, str):
                limit = 200 if k in ("file_path", "path") else (80 if k == "content" else 40)
                if len(v) > limit:
                    v = v[: limit - 3] + "..."
                parts.append(f'{k}="[dim]{escape(v)}[/dim]"')
            else:
                parts.append(f"{k}={escape(repr(v))}")
        return ", ".join(parts)

    def set_running(self) -> None:
        """Mark as running."""
        self.state = "running"
        self._update_header()

    def set_result(self, result: str, is_error: bool = False) -> None:
        """Update widget with tool result using inline truncation."""
        self._result = result
        self._is_error = is_error
        self.state = "error" if is_error else "success"

        self.remove_class("pending", "error", "success")
        self.add_class(self.state)
        self._update_header()

        summary = Text()
        if is_error:
            summary.append("✗ Failed\n", style="bold red")
        else:
            summary.append("✓ Success\n", style="bold green")

        lines = result.splitlines()
        max_lines = (
            READ_DISPLAY_MAX_LINES if self.tool_name == "read"
            else TOOL_RESULT_MAX_LINES
        )

        if len(lines) <= max_lines and len(result) <= TOOL_RESULT_MAX_CHARS:
            summary.append(result)
        else:
            display = lines[:max_lines]
            text = "\n".join(display)
            if len(text) > TOOL_RESULT_MAX_CHARS:
                text = text[:TOOL_RESULT_MAX_CHARS]
            summary.append(text)
            remaining = len(lines) - max_lines
            if remaining > 0:
                summary.append(
                    f"\n… {remaining} more lines ({_TRUNCATION_NOTICE})",
                    style="dim",
                )

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
