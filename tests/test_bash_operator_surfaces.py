"""Tests for bash job metadata and operator-facing surfaces."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.console import Console
from textual.app import App, ComposeResult
from textual.widgets import Static

import loader.cli.main as cli_main_module
from loader.runtime.events import AgentEvent
from loader.tools import BashTool
from loader.ui.adapter import (
    EventAdapter,
    ResponseComplete,
    ToolCallCompleted,
    ToolCallStarted,
)
from loader.ui.app import LoaderApp
from loader.ui.widgets import ApprovalBar, DiffWidget
from loader.ui.widgets.tool_widget import ToolCallWidget
from loader.utils.file_mutations import (
    build_file_mutation_preview,
    build_file_mutation_preview_dict,
    render_file_mutation_preview,
)


class _FakeApp:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def post_message(self, message: object) -> None:
        self.messages.append(message)


class _FakeShellOwner:
    def __init__(self) -> None:
        self.session = SimpleNamespace(runtime_owner_path="")
        self.last_turn_summary = None
        self.registry = SimpleNamespace(get=lambda name: None)
        self.safeguards = SimpleNamespace(filter_stream_chunk=lambda chunk: chunk)


class _ApprovalHost(App[None]):
    def compose(self) -> ComposeResult:
        yield ApprovalBar(id="approval")


def _patch_tool_args() -> dict[str, object]:
    return {
        "file_path": "~/Loader/animals/index.html",
        "hunks": [
            {
                "lines": [
                    '-            <a href="cat.html">Learn about Cats</a>',
                    '+            <a href="cat.html">Learn about Big Cats</a>',
                    '             <a href="dog.html">Learn about Dogs</a>',
                ],
                "new_lines": 2,
                "new_start": 18,
                "old_lines": 2,
                "old_start": 18,
            }
        ],
    }


def _raw_patch_tool_args() -> dict[str, object]:
    return {
        "file_path": "animals/index.html",
        "hunks": [
            {
                "old_start": 53,
                "old_lines": 1,
                "new_start": 53,
                "new_lines": 5,
                "lines": [
                    "</body>",
                    "</html>",
                    "",
                    "<!-- New animal entries -->",
                    '<div class="animal-card">',
                    '<h2><a href="wolf.html">Wolf</a></h2>',
                    (
                        "<p>Wolves are wild canines that live in packs and are known "
                        "for their intelligence and social behavior.</p>"
                    ),
                    "</div>",
                    "",
                    '<div class="animal-card">',
                    '<h2><a href="bear.html">Bear</a></h2>',
                    (
                        "<p>Bears are large mammals that are found in various parts "
                        "of the world, known for their strength and omnivorous diet.</p>"
                    ),
                    "</div>",
                    "",
                    '<div class="animal-card">',
                    '<h2><a href="penguin.html">Penguin</a></h2>',
                    (
                        "<p>Penguins are flightless birds that live in the Southern "
                        "Hemisphere, known for their distinctive waddle and swimming "
                        "abilities.</p>"
                    ),
                    "</div>",
                ],
            }
        ],
    }


def _render_text(renderable, *, width: int = 100) -> str:
    console = Console(record=True, width=width)
    console.print(renderable)
    return console.export_text(styles=False)


def test_event_adapter_preserves_tool_metadata_on_completion() -> None:
    app = _FakeApp()
    adapter = EventAdapter(app)  # type: ignore[arg-type]
    metadata = {"job_id": "bash-3", "status": "running", "background": True}

    adapter.handle_event(
        AgentEvent(
            type="tool_call",
            tool_name="bash",
            tool_call_id="call-bash-3",
            tool_args={"command": "python -m http.server 8000", "background": True},
            phase="assistant",
        )
    )
    adapter.handle_event(
        AgentEvent(
            type="tool_result",
            tool_name="bash",
            tool_call_id="call-bash-3",
            content="Started bash job bash-3",
            tool_metadata=metadata,
            phase="assistant",
        )
    )

    completed = next(message for message in app.messages if isinstance(message, ToolCallCompleted))
    assert completed.tool_call_id == "call-bash-3"
    assert completed.metadata == metadata


def test_event_adapter_adds_mutation_preview_for_patch_completion() -> None:
    app = _FakeApp()
    adapter = EventAdapter(app)  # type: ignore[arg-type]
    tool_args = _patch_tool_args()

    adapter.handle_event(
        AgentEvent(
            type="tool_call",
            tool_name="patch",
            tool_call_id="call-patch-1",
            tool_args=tool_args,
            phase="assistant",
        )
    )
    adapter.handle_event(
        AgentEvent(
            type="tool_result",
            tool_name="patch",
            tool_call_id="call-patch-1",
            content="Successfully patched ~/Loader/animals/index.html",
            tool_metadata={
                "file_path": "~/Loader/animals/index.html",
                "structured_patch": tool_args["hunks"],
            },
            phase="assistant",
        )
    )

    completed = next(message for message in app.messages if isinstance(message, ToolCallCompleted))
    assert completed.mutation_preview is not None
    assert completed.mutation_preview["operation"] == "patch"
    assert completed.mutation_preview["file_path"] == "~/Loader/animals/index.html"


def test_tool_call_widget_renders_full_bash_command_in_box() -> None:
    command = "python -m http.server 8000 --directory /tmp/preview-pages"
    widget = ToolCallWidget("bash", {"command": command})

    header = widget._header_renderable().plain
    rendered = _render_text(widget._build_initial_summary(), width=120)

    assert "Bash" in header
    assert "command=" not in header
    assert "Command" in rendered
    assert command in rendered


def test_build_file_mutation_preview_uses_structured_patch_metadata() -> None:
    preview = build_file_mutation_preview(
        "write",
        metadata={
            "kind": "update",
            "file_path": "/tmp/animals/index.html",
            "original_file": "<h1>Animals</h1>\n",
            "content": "<h1>Animals</h1>\n<p>Updated</p>\n",
            "structured_patch": [
                {
                    "old_start": 1,
                    "old_lines": 1,
                    "new_start": 1,
                    "new_lines": 2,
                    "lines": [
                        " <h1>Animals</h1>",
                        "+<p>Updated</p>",
                    ],
                }
            ],
        },
    )

    assert preview is not None
    assert preview.operation == "update"
    assert preview.added_lines == 1
    assert preview.removed_lines == 0


def test_render_file_mutation_preview_truncates_large_diff() -> None:
    preview = build_file_mutation_preview(
        "write",
        tool_args={
            "file_path": "/tmp/generated.txt",
            "content": "\n".join(f"line {idx}" for idx in range(120)),
        },
    )
    assert preview is not None

    rendered = _render_text(
        render_file_mutation_preview(preview, max_lines=6, max_chars=1_000),
        width=120,
    )

    assert "Create(generated.txt)" in rendered
    assert "truncated for display" in rendered


def test_cli_print_tool_call_renders_bash_panel_without_truncating(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli_main_module, "console", console)

    command = "python -m http.server 8000 --directory /tmp/preview-pages"
    cli_main_module._print_tool_call("bash", {"command": command})

    rendered = console.export_text(styles=False)
    assert "Bash" in rendered
    assert "Command" in rendered
    assert command in rendered
    assert "command=" not in rendered


def test_tool_call_widget_renders_patch_preview_instead_of_raw_tool_call() -> None:
    widget = ToolCallWidget("patch", _patch_tool_args())

    header = widget._header_renderable().plain
    rendered = _render_text(widget._build_initial_summary(), width=120)

    assert "Patch" in header
    assert 'file_path="~/Loader/animals/index.html"' in header
    assert "Preview" in rendered
    assert "Patch(index.html)" in rendered
    assert "<a href=\"cat.html\">Learn about Big Cats</a>" in rendered
    assert "hunks=1 hunk" not in rendered


def test_cli_print_tool_call_renders_patch_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli_main_module, "console", console)

    cli_main_module._print_tool_call("patch", _patch_tool_args())

    rendered = console.export_text(styles=False)
    assert "Patch" in rendered
    assert "Preview" in rendered
    assert "Patch(index.html)" in rendered
    assert "<a href=\"cat.html\">Learn about Big Cats</a>" in rendered
    assert "hunks=1 hunk" not in rendered


def test_cli_print_tool_result_renders_edit_diff_from_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli_main_module, "console", console)

    cli_main_module._print_tool_result(
        "edit",
        "Successfully edited index.html",
        metadata={
            "file_path": "/tmp/index.html",
            "original_file": "<p>Old</p>\n",
            "new_string": "<p>New</p>\n",
            "structured_patch": [
                {
                    "old_start": 1,
                    "old_lines": 1,
                    "new_start": 1,
                    "new_lines": 1,
                    "lines": [
                        "-<p>Old</p>",
                        "+<p>New</p>",
                    ],
                }
            ],
        },
    )

    rendered = console.export_text(styles=False)
    assert "Edit" in rendered
    assert "Diff" in rendered
    assert "+ <p>New</p>" in rendered
    assert "- <p>Old</p>" in rendered


@pytest.mark.asyncio
async def test_approval_bar_renders_file_mutation_preview() -> None:
    app = _ApprovalHost()
    preview = build_file_mutation_preview_dict("patch", tool_args=_patch_tool_args())
    assert preview is not None

    async with app.run_test() as pilot:
        bar = app.query_one(ApprovalBar)
        bar.show_approval(
            "patch",
            "Patch file: ~/Loader/animals/index.html",
            "apply structured patch hunks",
            preview=preview,
        )
        await pilot.pause()

        content = bar.query_one("#approval-content", Static)
        rendered = _render_text(content.content, width=120)

        assert "Approve Patch" in rendered
        assert "Preview" in rendered
        assert "Patch(index.html)" in rendered


@pytest.mark.asyncio
async def test_approval_bar_fallback_handles_raw_patch_details() -> None:
    app = _ApprovalHost()
    raw_details = (
        "patch(file_path=\"animals/index.html\", hunks="
        f"{_raw_patch_tool_args()['hunks']})"
    )

    async with app.run_test() as pilot:
        bar = app.query_one(ApprovalBar)
        bar.show_approval(
            "patch",
            "Patch file: animals/index.html",
            raw_details,
            preview=None,
        )
        await pilot.pause()

        content = bar.query_one("#approval-content", Static)
        rendered = _render_text(content.content, width=120)

        assert "Approve Patch" in rendered
        assert "Details" in rendered
        assert "wolf.html" in rendered


@pytest.mark.asyncio
async def test_loader_app_replaces_patch_tool_widget_with_diff_widget() -> None:
    tool_args = _patch_tool_args()
    preview = build_file_mutation_preview_dict("patch", tool_args=tool_args)
    assert preview is not None

    app = LoaderApp(shell_owner=_FakeShellOwner())
    async with app.run_test() as pilot:
        app.post_message(
            ToolCallStarted(
                tool_name="patch",
                tool_args=tool_args,
                tool_call_id="patch-call-1",
                phase="assistant",
            )
        )
        await pilot.pause()
        assert len(list(app.query(ToolCallWidget))) == 1

        app.post_message(
            ToolCallCompleted(
                tool_name="patch",
                content="Successfully patched ~/Loader/animals/index.html",
                is_error=False,
                phase="assistant",
                tool_call_id="patch-call-1",
                metadata={
                    "file_path": "~/Loader/animals/index.html",
                    "structured_patch": tool_args["hunks"],
                },
                mutation_preview=preview,
            )
        )
        await pilot.pause()

        assert len(list(app.query(DiffWidget))) == 1
        assert len(list(app.query(ToolCallWidget))) == 0


@pytest.mark.asyncio
async def test_loader_app_mounts_raw_patch_preview_without_markup_crash() -> None:
    app = LoaderApp(shell_owner=_FakeShellOwner())

    async with app.run_test() as pilot:
        app.post_message(
            ToolCallStarted(
                tool_name="patch",
                tool_args=_raw_patch_tool_args(),
                tool_call_id="patch-call-raw",
                phase="assistant",
            )
        )
        await pilot.pause()

        widget = next(iter(app.query(ToolCallWidget)))
        summary = widget.query_one("#tool-summary", Static)
        rendered = _render_text(summary.content, width=120)

        assert "Preview" in rendered
        assert "<h2><a href=\"wolf.html\">Wolf</a></h2>" in rendered
        assert "wolf.html" in rendered
        assert "penguin.html" in rendered
        assert "< /body>" not in rendered


@pytest.mark.asyncio
async def test_loader_app_matches_repeated_tool_results_by_tool_call_id() -> None:
    app = LoaderApp(shell_owner=_FakeShellOwner())
    async with app.run_test() as pilot:
        app.post_message(
            ToolCallStarted(
                tool_name="read",
                tool_args={"file_path": "/tmp/cats.html"},
                tool_call_id="read-call-1",
                phase="assistant",
            )
        )
        app.post_message(
            ToolCallStarted(
                tool_name="read",
                tool_args={"file_path": "/tmp/penguins.html"},
                tool_call_id="read-call-2",
                phase="assistant",
            )
        )
        await pilot.pause()

        app.post_message(
            ToolCallCompleted(
                tool_name="read",
                tool_call_id="read-call-2",
                content="<h1>Penguins</h1>",
                is_error=False,
                phase="assistant",
            )
        )
        await pilot.pause()

        widgets = list(app.query(ToolCallWidget))
        first = next(widget for widget in widgets if widget.tool_call_id == "read-call-1")
        second = next(widget for widget in widgets if widget.tool_call_id == "read-call-2")

        assert first.state == "running"
        assert second.state == "success"
        assert "/tmp/cats.html" in first._header_renderable().plain
        assert "/tmp/penguins.html" in second._header_renderable().plain


@pytest.mark.asyncio
async def test_loader_app_renders_plain_response_without_markup_parsing() -> None:
    app = LoaderApp(shell_owner=_FakeShellOwner())

    async with app.run_test() as pilot:
        app.post_message(
            ResponseComplete(
                content=(
                    "patch(file_path=\"animals/index.html\", hunks="
                    f"{_raw_patch_tool_args()['hunks']})"
                )
            )
        )
        await pilot.pause()

        message_area = app.query_one("#message-area")
        last_widget = list(message_area.children)[-1]
        rendered = _render_text(last_widget.render(), width=120)
        assert "patch(file_path=" in rendered
        assert "hunks=" in rendered


def test_cli_parse_local_bash_commands_supports_slash_aliases() -> None:
    assert cli_main_module._parse_local_bash_command("/jobs 5") == ("bash_jobs", {"limit": 5})
    assert cli_main_module._parse_local_bash_command("/wait bash-7 2.5") == (
        "bash_wait",
        {"job_id": "bash-7", "timeout": 2.5},
    )
    assert cli_main_module._parse_local_bash_command("kill bash-2 50") == (
        "bash_kill",
        {"job_id": "bash-2", "force_after_ms": 50},
    )


@pytest.mark.asyncio
async def test_cli_interrupt_active_foreground_bash_prints_interrupted_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli_main_module, "console", console)

    bash_tool = BashTool(timeout=10.0)
    job = await bash_tool.manager.start(
        command='python -c "import time; time.sleep(30)"',
        cwd=None,
        timeout=10.0,
        background=False,
        mutability="workspace-write",
    )
    owner = SimpleNamespace(
        registry=SimpleNamespace(get=lambda name: bash_tool if name == "bash" else None)
    )

    try:
        interrupted = await cli_main_module._interrupt_active_foreground_bash(owner)
        assert interrupted is True
        assert bash_tool.manager.active_foreground_job_id is None
        rendered = console.export_text(styles=False)
        assert "Status: interrupted" in rendered
    finally:
        if job.is_running:
            await bash_tool.manager.kill_job(job.job_id, interrupted=True)
