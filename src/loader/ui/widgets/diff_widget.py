"""Shared diff widget for file-mutation tool previews."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ...utils.file_mutations import render_file_mutation_preview


class DiffWidget(Vertical):
    """Display normalized file-mutation previews with inline diff rendering."""

    def __init__(self, preview: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.preview = preview

    def compose(self) -> ComposeResult:
        yield Static(
            render_file_mutation_preview(
                self.preview,
                border_style="cyan",
                title="Diff",
                max_lines=60,
                max_chars=6_000,
            ),
            classes="diff-content",
        )
