"""Tests for the runtime-owned decomposition lane."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import CompletionResponse
from loader.runtime.decomposition_lane import DecompositionTurnRunner
from loader.runtime.deliberation import DECOMPOSITION_PROMPT
from tests.helpers.runtime_harness import ScriptedBackend


def _decomposition_json(*, subtasks: list[dict[str, object]]) -> CompletionResponse:
    return CompletionResponse(content=json.dumps({"subtasks": subtasks}))


@pytest.mark.asyncio
async def test_decomposition_turn_runner_executes_subtasks_and_summarizes(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            _decomposition_json(
                subtasks=[
                    {
                        "id": "1",
                        "description": "Read the spec",
                        "verification": "Spec is understood",
                    },
                    {
                        "id": "2",
                        "description": "Implement the feature",
                        "dependencies": ["1"],
                        "verification": "Tests pass",
                    },
                ]
            )
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    events = []
    calls: list[tuple[str, str | None, str | None]] = []

    async def emit(event) -> None:
        events.append(event)

    async def run_task(
        task: str,
        _emit,
        _on_confirmation,
        _on_user_question,
        requested_mode: str | None,
        original_task: str | None,
    ) -> str:
        calls.append((task, requested_mode, original_task))
        if task == "Read the spec":
            return "Spec reviewed."
        if task == "Implement the feature":
            return "Feature implemented."
        return "Feature shipped."

    runner = DecompositionTurnRunner(agent, run_task=run_task)
    response = await runner.run(
        "Read the spec and implement the feature",
        emit,
        original_task="Read the spec and implement the feature",
    )

    assert response == "Feature shipped."
    assert [call[0] for call in calls] == [
        "Read the spec",
        "Implement the feature",
        (
            "All subtasks completed for: Read the spec and implement the feature\n\n"
            "Task: Read the spec and implement the feature\n\n"
            "Subtasks:\n"
            "  ● 1. Read the spec\n"
            "      Verify: Spec is understood\n"
            "  ● 2. Implement the feature (after: 1)\n"
            "      Verify: Tests pass\n\n"
            "Provide a brief summary of what was accomplished."
        ),
    ]
    assert all(call[1] is None for call in calls)
    assert all(
        call[2] == "Read the spec and implement the feature"
        for call in calls
    )
    assert [event.type for event in events] == [
        "thinking",
        "decomposition",
        "subtask",
        "subtask",
    ]
    assert backend.invocations[0].messages[1].content == DECOMPOSITION_PROMPT.format(
        task="Read the spec and implement the feature"
    )
    assert [
        message.content for message in agent.session.messages
    ] == [
        "Execute this subtask: Read the spec\n\nVerification: Spec is understood",
        "Execute this subtask: Implement the feature\n\nVerification: Tests pass",
        (
            "All subtasks completed for: Read the spec and implement the feature\n\n"
            "Task: Read the spec and implement the feature\n\n"
            "Subtasks:\n"
            "  ● 1. Read the spec\n"
            "      Verify: Spec is understood\n"
            "  ● 2. Implement the feature (after: 1)\n"
            "      Verify: Tests pass\n\n"
            "Provide a brief summary of what was accomplished."
        ),
    ]


@pytest.mark.asyncio
async def test_decomposition_turn_runner_returns_partial_completion_after_failed_retries(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            _decomposition_json(
                subtasks=[
                    {
                        "id": "1",
                        "description": "Patch the file",
                        "verification": "File is updated",
                    },
                    {
                        "id": "2",
                        "description": "Run the tests",
                        "dependencies": ["1"],
                        "verification": "Tests are green",
                    },
                ]
            )
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    events = []
    responses = iter(["failed once", "failed again"])
    calls: list[str] = []

    async def emit(event) -> None:
        events.append(event)

    async def run_task(
        task: str,
        _emit,
        _on_confirmation,
        _on_user_question,
        _requested_mode,
        _original_task,
    ) -> str:
        calls.append(task)
        return next(responses)

    runner = DecompositionTurnRunner(agent, run_task=run_task)
    response = await runner.run("Patch the file and run the tests", emit)

    assert response.startswith("Task partially completed. Task: Patch the file and run the tests")
    assert calls == ["Patch the file", "Patch the file"]
    assert [event.content for event in events if event.type == "subtask"] == [
        "[0/2] Patch the file",
        "Retrying subtask: Patch the file",
        "[0/2] Patch the file",
    ]

