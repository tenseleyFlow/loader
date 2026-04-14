"""Diff widget for file edit and write operations."""

import difflib
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class DiffWidget(Vertical):
    """Display file diffs with syntax highlighting using Rich Text objects."""

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
        self.is_new_file = not old_string

    def compose(self) -> ComposeResult:
        old_lines = self.old_string.splitlines() if self.old_string else []
        new_lines = self.new_string.splitlines()

        filename = Path(self.file_path).name
        if self.is_new_file:
            yield Static(
                f"[green]●[/green] [bold green]Create[/bold green]({filename})\n"
                f"└ [green]+{len(new_lines)}[/green] lines",
                classes="diff-header",
            )
        else:
            added = sum(1 for line in new_lines if line not in old_lines)
            removed = sum(1 for line in old_lines if line not in new_lines)
            stats_parts = []
            if added > 0:
                stats_parts.append(f"[green]+{added}[/green]")
            if removed > 0:
                stats_parts.append(f"[red]-{removed}[/red]")
            stats = f"└ {', '.join(stats_parts)} lines" if stats_parts else ""
            yield Static(
                f"[green]●[/green] [bold cyan]Update[/bold cyan]({filename})\n{stats}",
                classes="diff-header",
            )

        # Full diff content rendered with Rich Text (no markup parsing errors)
        diff_content = Static(classes="diff-content")
        diff_content.update(self._render_diff())
        yield diff_content

    def _render_diff(self) -> Text:
        """Render the diff with colors using Rich Text objects."""
        diff = difflib.unified_diff(
            self.old_string.splitlines(keepends=True),
            self.new_string.splitlines(keepends=True),
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}",
            n=self.context_lines,
        )

        result = Text()
        line_num = 0

        for line in diff:
            line = line.rstrip("\n")

            if line.startswith("+++") or line.startswith("---"):
                continue
            elif line.startswith("@@"):
                parts = line.split(" ")
                if len(parts) >= 3:
                    try:
                        new_range = parts[2]
                        line_num = int(new_range.split(",")[0].lstrip("+")) - 1
                    except (ValueError, IndexError):
                        pass
                result.append(f"{line}\n", style="dim")
            elif line.startswith("+"):
                line_num += 1
                content = line[1:]
                result.append(f"{line_num:4d} ", style="dim")
                result.append(f"+ {content}\n", style="green")
            elif line.startswith("-"):
                content = line[1:]
                result.append("     ", style="dim")
                result.append(f"- {content}\n", style="red")
            else:
                line_num += 1
                content = line[1:] if line.startswith(" ") else line
                result.append(f"{line_num:4d}", style="dim")
                result.append(f"   {content}\n")

        if not result.plain:
            result.append("No changes", style="dim")

        return result
