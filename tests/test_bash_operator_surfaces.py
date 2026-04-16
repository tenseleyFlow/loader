"""Tests for bash job metadata and operator-facing surfaces."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.console import Console

import loader.cli.main as cli_main_module
from loader.runtime.events import AgentEvent
from loader.tools import BashTool
from loader.ui.adapter import EventAdapter, ToolCallCompleted
from loader.ui.widgets.tool_widget import ToolCallWidget


class _FakeApp:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def post_message(self, message: object) -> None:
        self.messages.append(message)


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
            tool_args={"command": "python -m http.server 8000", "background": True},
            phase="assistant",
        )
    )
    adapter.handle_event(
        AgentEvent(
            type="tool_result",
            tool_name="bash",
            content="Started bash job bash-3",
            tool_metadata=metadata,
            phase="assistant",
        )
    )

    completed = next(message for message in app.messages if isinstance(message, ToolCallCompleted))
    assert completed.metadata == metadata


def test_tool_call_widget_renders_full_bash_command_in_box() -> None:
    command = "python -m http.server 8000 --directory /tmp/preview-pages"
    widget = ToolCallWidget("bash", {"command": command})

    header = widget._header_renderable().plain
    rendered = _render_text(widget._build_initial_summary(), width=120)

    assert "Bash" in header
    assert "command=" not in header
    assert "Command" in rendered
    assert command in rendered


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


def test_tool_call_widget_summarizes_patch_hunks_safely() -> None:
    widget = ToolCallWidget(
        "patch",
        {
            "file_path": "~/Loader/animals/index.html",
            "hunks": [
                {
                    "lines": [
                        '            <a href="cat.html">Learn about Cats</a>',
                        '            <a href="dog.html">Learn about Dogs</a>',
                    ],
                    "new_lines": 2,
                    "new_start": 18,
                    "old_lines": 2,
                    "old_start": 18,
                }
            ],
        },
    )

    header = widget._header_renderable().plain

    assert "patch" in header
    assert 'file_path="~/Loader/animals/index.html"' in header
    assert "hunks=1 hunk" in header
    assert "<a href=" not in header


def test_cli_print_tool_call_summarizes_patch_hunks_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli_main_module, "console", console)

    cli_main_module._print_tool_call(
        "patch",
        {
            "file_path": "~/Loader/animals/index.html",
            "hunks": [
                {
                    "lines": [
                        '            <a href="cat.html">Learn about Cats</a>',
                        '            <a href="dog.html">Learn about Dogs</a>',
                    ],
                    "new_lines": 2,
                    "new_start": 18,
                    "old_lines": 2,
                    "old_start": 18,
                }
            ],
        },
    )

    rendered = console.export_text(styles=False)
    assert 'file_path="~/Loader/animals/index.html"' in rendered
    assert "hunks=1 hunk" in rendered
    assert "<a href=" not in rendered


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
