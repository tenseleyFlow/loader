"""Tool call widget with bash-specific rich rendering."""

import json
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

from ...utils.file_mutations import (
    build_file_mutation_preview,
    is_file_mutation_tool,
    render_file_mutation_preview,
)

# Display truncation limits
TOOL_RESULT_MAX_LINES = 60
TOOL_RESULT_MAX_CHARS = 6000
WRITE_PREVIEW_MAX_LINES = 80
READ_DISPLAY_MAX_LINES = 80
_TRUNCATION_NOTICE = "truncated for display; full result preserved in session"


class ToolCallWidget(Vertical):
    """Widget for tool calls with inline content display."""

    TOOL_LABELS = {
        "write": "Write",
        "edit": "Edit",
        "patch": "Patch",
        "bash": "Bash",
        "bash_jobs": "Bash Jobs",
        "bash_wait": "Bash Wait",
        "bash_kill": "Bash Kill",
    }

    state: reactive[str] = reactive("pending")

    def __init__(
        self,
        tool_name: str,
        tool_args: dict | None = None,
        tool_call_id: str | None = None,
        phase: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.tool_call_id = tool_call_id
        self.phase = phase
        self._result: str = ""
        self._is_error: bool = False
        self._metadata: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Static(
            self._header_renderable(),
            id="tool-header",
            classes="tool-header",
        )
        yield Static(self._build_initial_summary(), id="tool-summary", classes="tool-summary")

    def _format_args(self) -> str:
        """Format tool arguments for display."""
        if self._is_bash_command_tool():
            return ""
        if self._is_file_mutation_tool():
            file_path = self.tool_args.get("file_path") or self.tool_args.get("path")
            if file_path:
                return f"file_path={self._format_arg_value('file_path', file_path)}"
            return ""
        if not self.tool_args:
            return ""
        parts = []
        for k, v in self.tool_args.items():
            parts.append(f"{k}={self._format_arg_value(k, v)}")
        return ", ".join(parts)

    def _format_arg_value(self, key: str, value: Any) -> str:
        """Format one argument value as plain text safe for header rendering."""
        if isinstance(value, str):
            limit = 200 if key in ("file_path", "path") else (80 if key == "content" else 40)
            if len(value) > limit:
                value = value[: limit - 3] + "..."
            return json.dumps(value)

        if key == "hunks" and isinstance(value, list):
            return f"{len(value)} hunk" if len(value) == 1 else f"{len(value)} hunks"
        if key == "todos" and isinstance(value, list):
            return f"{len(value)} todo" if len(value) == 1 else f"{len(value)} todos"
        if isinstance(value, list):
            return f"{len(value)} item" if len(value) == 1 else f"{len(value)} items"
        if isinstance(value, dict):
            keys = ", ".join(sorted(value.keys())[:4])
            suffix = "" if len(value) <= 4 else ", ..."
            return f"{{{keys}{suffix}}}"

        rendered = repr(value)
        if len(rendered) > 80:
            rendered = rendered[:77] + "..."
        return rendered

    def _display_name(self) -> str:
        base = self.TOOL_LABELS.get(self.tool_name, self.tool_name)
        if self.phase == "verification":
            return f"Verify {base}"
        return base

    def _is_bash_command_tool(self) -> bool:
        return self.tool_name == "bash"

    def _is_file_mutation_tool(self) -> bool:
        return is_file_mutation_tool(self.tool_name)

    def _header_renderable(self) -> Text:
        args_str = self._format_args()
        label = self._display_name()
        bullet_symbol = {
            "pending": "○",
            "running": "◐",
            "success": "●",
            "error": "●",
        }.get(self.state, "○")
        bullet_style = {
            "pending": "yellow",
            "running": "yellow",
            "success": "green",
            "error": "red",
        }.get(self.state, "yellow")
        label_style = "bold red" if self._is_error else "bold cyan"

        text = Text()
        text.append(bullet_symbol, style=bullet_style)
        text.append(" ")
        text.append(label, style=label_style)
        if args_str:
            text.append(f"({args_str})")
        return text

    def _build_initial_summary(self):
        if self._is_bash_command_tool():
            return Group(self._render_bash_command_panel())

        if self._is_file_mutation_tool():
            preview = build_file_mutation_preview(
                self.tool_name,
                tool_args=self.tool_args,
            )
            if preview is not None:
                return render_file_mutation_preview(
                    preview,
                    border_style=(
                        "magenta" if self.phase == "verification" else "cyan"
                    ),
                    title="Preview",
                    max_lines=WRITE_PREVIEW_MAX_LINES,
                    max_chars=6_000,
                )

        return Text()

    def _render_bash_command_panel(self) -> Panel:
        command = str(self.tool_args.get("command", "")).strip() or "(empty command)"
        return Panel(
            Text(command),
            title="Command",
            border_style="cyan",
            box=box.SQUARE,
            expand=True,
        )

    def _truncate_result(self, result: str, *, line_limit: int) -> tuple[str, bool]:
        lines = result.splitlines()
        if len(lines) <= line_limit and len(result) <= TOOL_RESULT_MAX_CHARS:
            return result, False

        display = lines[:line_limit]
        text = "\n".join(display)
        if len(text) > TOOL_RESULT_MAX_CHARS:
            text = text[:TOOL_RESULT_MAX_CHARS]
        return text, True

    def _build_bash_result(self, result: str):
        metadata = self._metadata
        renderables = [self._render_bash_command_panel()]
        status = Text()
        status.append(
            "✗ Failed\n" if self._is_error else "✓ Success\n",
            style="bold red" if self._is_error else "bold green",
        )

        detail_lines = []
        status_value = str(metadata.get("status", "failed" if self._is_error else "completed"))
        detail_lines.append(f"Status: {status_value.replace('_', ' ')}")
        job_id = metadata.get("job_id")
        if job_id:
            detail_lines.append(f"Job: {job_id}")
        pid = metadata.get("pid")
        if pid:
            detail_lines.append(f"PID: {pid}")
        if metadata.get("exit_code") is not None:
            detail_lines.append(f"Exit: {metadata['exit_code']}")
        if metadata.get("background") is not None:
            detail_lines.append(
                f"Mode: {'background' if metadata.get('background') else 'foreground'}"
            )
        if detail_lines:
            status.append("\n".join(detail_lines))

        stdout_text = str(metadata.get("stdout", "") or "")
        stderr_text = str(metadata.get("stderr", "") or "")
        show_summary_note = (
            (not stdout_text and not stderr_text and bool(result.strip()))
            or status_value not in {"completed", "running"}
        )
        if show_summary_note and result.strip():
            if status.plain:
                status.append("\n\n")
            preview, truncated = self._truncate_result(result, line_limit=24)
            status.append(preview)
            if truncated:
                status.append(f"\n… {_TRUNCATION_NOTICE}", style="dim")

        renderables.append(
            Panel(
                status,
                title="Status",
                border_style="red" if self._is_error else "green",
                box=box.SQUARE,
                expand=True,
            )
        )

        for stream_name, stream_text, truncated in (
            ("stdout", stdout_text, bool(metadata.get("stdout_truncated"))),
            ("stderr", stderr_text, bool(metadata.get("stderr_truncated"))),
        ):
            if not stream_text:
                continue
            preview, preview_truncated = self._truncate_result(stream_text, line_limit=40)
            stream_panel_text = Text(preview)
            if truncated or preview_truncated:
                stream_panel_text.append(f"\n… {_TRUNCATION_NOTICE}", style="dim")
            renderables.append(
                Panel(
                    stream_panel_text,
                    title=stream_name,
                    border_style="red" if stream_name == "stderr" else "dim",
                    box=box.SQUARE,
                    expand=True,
                )
            )

        return Group(*renderables)

    def set_running(self) -> None:
        """Mark as running."""
        self.state = "running"
        self._update_header()

    def set_result(
        self,
        result: str,
        is_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update widget with tool result using inline truncation."""
        self._result = result
        self._is_error = is_error
        self._metadata = metadata or {}
        self.state = "error" if is_error else "success"

        self.remove_class("pending", "running", "error", "success")
        self.add_class(self.state)
        self._update_header()

        if self._is_bash_command_tool():
            self._update_summary(self._build_bash_result(result))
            return

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

        self._update_summary(summary)

    def _update_header(self) -> None:
        """Update the header with current state."""
        try:
            self.query_one("#tool-header", Static).update(self._header_renderable())
        except NoMatches:
            self.call_after_refresh(self._update_header)

    def _update_summary(self, renderable) -> None:
        """Update the summary body once child widgets are mounted."""
        try:
            self.query_one("#tool-summary", Static).update(renderable)
        except NoMatches:
            self.call_after_refresh(lambda: self._update_summary(renderable))

    def watch_state(self, state: str) -> None:
        """React to state changes."""
        self.remove_class("pending", "running", "success", "error")
        self.add_class(state)
