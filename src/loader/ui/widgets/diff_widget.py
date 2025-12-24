"""Diff widget for file edit operations."""

import difflib
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class DiffWidget(Vertical):
    """Display file diffs with syntax highlighting."""

    def __init__(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        context_lines: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.file_path = file_path
        self.old_string = old_string
        self.new_string = new_string
        self.context_lines = context_lines

    def compose(self) -> ComposeResult:
        # Calculate stats
        old_lines = self.old_string.splitlines()
        new_lines = self.new_string.splitlines()
        added = sum(1 for line in new_lines if line not in old_lines)
        removed = sum(1 for line in old_lines if line not in new_lines)

        # Header
        filename = Path(self.file_path).name
        yield Static(
            f"[green]●[/green] [bold cyan]Update[/bold cyan]({filename})",
            classes="diff-header",
        )

        # Stats
        stats_parts = []
        if added > 0:
            stats_parts.append(f"[green]+{added}[/green]")
        if removed > 0:
            stats_parts.append(f"[red]-{removed}[/red]")
        if stats_parts:
            yield Static(
                f"└ {', '.join(stats_parts)} lines",
                classes="diff-stats",
            )

        # Diff content
        yield Static(
            self._render_diff(),
            classes="diff-content",
        )

    def _render_diff(self) -> str:
        """Render the diff with colors."""
        diff = difflib.unified_diff(
            self.old_string.splitlines(keepends=True),
            self.new_string.splitlines(keepends=True),
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}",
            n=self.context_lines,
        )

        lines = []
        line_num = 0

        for line in diff:
            line = line.rstrip("\n")

            if line.startswith("+++") or line.startswith("---"):
                # File headers - skip
                continue
            elif line.startswith("@@"):
                # Hunk header - extract line number
                parts = line.split(" ")
                if len(parts) >= 3:
                    # Parse @@ -start,count +start,count @@
                    try:
                        new_range = parts[2]
                        line_num = int(new_range.split(",")[0].lstrip("+")) - 1
                    except (ValueError, IndexError):
                        pass
                lines.append(f"[dim]{line}[/dim]")
            elif line.startswith("+"):
                line_num += 1
                # Added line - green
                content = line[1:]  # Remove the + prefix
                lines.append(
                    f"[dim]{line_num:4d}[/dim] [green on #1a3d1a]+ {content}[/green on #1a3d1a]"
                )
            elif line.startswith("-"):
                # Removed line - red (no line number increment)
                content = line[1:]  # Remove the - prefix
                lines.append(
                    f"[dim]    [/dim] [red on #3d1a1a]- {content}[/red on #3d1a1a]"
                )
            else:
                # Context line
                line_num += 1
                content = line[1:] if line.startswith(" ") else line
                lines.append(f"[dim]{line_num:4d}[/dim]   {content}")

        return "\n".join(lines) if lines else "[dim]No changes[/dim]"
