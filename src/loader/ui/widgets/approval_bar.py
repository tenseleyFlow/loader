"""Approval bar widget for command confirmation (Claude Code style)."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static


class ApprovalBar(Widget, can_focus=True):
    """Inline approval bar that appears above the input when confirmation is needed.

    Shows: [tool_name] command_preview                    [Y]es [n]o [e]dit

    Similar to Claude Code's approval flow.
    """

    BINDINGS = [
        Binding("y", "approve", "Yes", show=False, priority=True),
        Binding("n", "reject", "No", show=False, priority=True),
        Binding("e", "edit", "Edit", show=False, priority=True),
        Binding("escape", "reject", "Cancel", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    ApprovalBar {
        height: auto;
        max-height: 4;
        display: none;
        padding: 0 1;
        background: $warning 15%;
        border-top: tall $warning;
        border-bottom: tall $warning;
    }

    ApprovalBar.visible {
        display: block;
    }

    ApprovalBar #approval-content {
        width: 100%;
    }
    """

    class Approved(Message):
        """Message sent when user approves the action."""
        pass

    class Rejected(Message):
        """Message sent when user rejects the action."""
        pass

    class EditRequested(Message):
        """Message sent when user wants to edit the command."""
        def __init__(self, command: str) -> None:
            self.command = command
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tool_name: str = ""
        self._command_preview: str = ""
        self._full_command: str = ""
        # Make this widget focusable from the start
        self.can_focus = True

    def compose(self) -> ComposeResult:
        yield Static("", id="approval-content")

    def show_approval(self, tool_name: str, message: str, details: str = "") -> None:
        """Show the approval bar with a pending action.

        Args:
            tool_name: Name of the tool (e.g., "bash", "write")
            message: Short message about the action
            details: Command or details to show (and potentially edit)
        """
        self._tool_name = tool_name
        self._full_command = details

        preview = details if details else message
        if len(preview) > 70:
            preview = preview[:67] + "..."
        content = self.query_one("#approval-content", Static)
        content.update(
            f"[bold $warning]\\[{tool_name}][/] {preview}  "
            f"[bold green]\\[Y][/]es  [bold red]\\[n][/]o  [bold]\\[e][/]dit"
        )

        # Show the bar
        self.add_class("visible")

        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(
                    f"[approval-bar] show_approval: tool={tool_name}, "
                    f"visible=True, scheduling focus...\n"
                )
        except Exception:
            pass

        # Use a short timer to let Textual complete the layout pass
        # before attempting focus. call_after_refresh is too early.
        self.set_timer(0.15, self._deferred_focus)

    def _deferred_focus(self) -> None:
        """Attempt focus after layout has settled."""
        self.focus()
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"[approval-bar] deferred focus: has_focus={self.has_focus}\n")
        except Exception:
            pass
        if not self.has_focus:
            # Last resort: try scrolling into view and focusing again
            self.scroll_visible()
            self.focus()

    def hide_approval(self) -> None:
        """Hide the approval bar."""
        self.remove_class("visible")
        self._tool_name = ""
        self._command_preview = ""
        self._full_command = ""

    def action_approve(self) -> None:
        """Handle 'y' key - approve the action."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write("[approval-bar] action_approve called, posting Approved message\n")
        except Exception:
            pass
        self.post_message(self.Approved())
        self.hide_approval()

    def action_reject(self) -> None:
        """Handle 'n' or escape - reject the action."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write("[approval-bar] action_reject called, posting Rejected message\n")
        except Exception:
            pass
        self.post_message(self.Rejected())
        self.hide_approval()

    def action_edit(self) -> None:
        """Handle 'e' key - edit the command."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write("[approval-bar] action_edit called, posting EditRequested message\n")
        except Exception:
            pass
        self.post_message(self.EditRequested(self._full_command))
        self.hide_approval()
