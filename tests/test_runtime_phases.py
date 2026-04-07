"""Tests for explicit runtime turn-phase tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, ToolCall
from tests.helpers.runtime_harness import ScriptedBackend, run_scenario


def _config(*, completion_check: bool = False) -> AgentConfig:
    config = AgentConfig(auto_context=False, stream=False, max_iterations=8)
    config.reasoning.completion_check = completion_check
    return config


def _turn_phases(run) -> list[str]:
    return [
        event.turn_phase
        for event in run.events
        if event.type == "turn_phase" and event.turn_phase
    ]


@pytest.mark.asyncio
async def test_empty_output_enters_repair_phase(temp_dir: Path) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=""),
            CompletionResponse(content="I can help with that now."),
        ]
    )

    run = await run_scenario(
        "Inspect the workspace and summarize it briefly.",
        backend,
        config=_config(completion_check=False),
        project_root=temp_dir,
    )

    phases = _turn_phases(run)
    assert "repair" in phases
    assert phases[:3] == ["prepare", "assistant", "repair"]
    assert phases[-2:] == ["completion", "finalize"]
    assert run.agent.session.active_turn_phase is None


@pytest.mark.asyncio
async def test_completion_nudge_and_tool_batch_emit_named_phases(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(temp_dir)
    target = temp_dir / "hello.py"
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content="Done."),
            CompletionResponse(
                content="You're right, I'll create the file first.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(target),
                            "content": "print('hello from loader')\n",
                        },
                    )
                ],
            ),
            CompletionResponse(
                content="Now I'll run the script.",
                tool_calls=[
                    ToolCall(
                        id="bash-1",
                        name="bash",
                        arguments={"command": f"python {target.name}"},
                    )
                ],
            ),
            CompletionResponse(content="Successfully created and ran hello.py."),
        ]
    )

    run = await run_scenario(
        "Create a hello.py file and run it.",
        backend,
        config=_config(completion_check=True),
        project_root=temp_dir,
    )

    phases = _turn_phases(run)
    assert "completion" in phases
    assert "tools" in phases
    assert phases[0] == "prepare"
    assert phases[-1] == "finalize"
    assert any(event.type == "completion_check" for event in run.events)
