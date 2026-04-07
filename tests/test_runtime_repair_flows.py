"""Deterministic coverage for current runtime repair behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, Role, ToolCall
from tests.helpers.runtime_harness import ScriptedBackend, run_scenario


def non_streaming_config() -> AgentConfig:
    """Shared deterministic config for repair-flow tests."""

    config = AgentConfig(auto_context=False, stream=False, max_iterations=8)
    config.reasoning.completion_check = False
    return config


def tool_event_names(run) -> list[str]:
    """Return non-verification tool events in order."""

    return [
        event.tool_name
        for event in run.events
        if event.type == "tool_call" and event.tool_name and event.phase != "verification"
    ]


@pytest.mark.asyncio
async def test_first_turn_action_prompt_does_not_inject_prefill_message(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[CompletionResponse(content="I can help with that.")]
    )

    await run_scenario(
        "Create allowed.txt with a greeting.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert not any(
        message.role == Role.ASSISTANT and message.content == "["
        for message in backend.invocations[0].messages
    )


@pytest.mark.asyncio
async def test_empty_response_retry_injects_honest_user_reminder_and_recovers(
    temp_dir: Path,
) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("repair baseline\n")
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=""),
            CompletionResponse(
                content="I'll inspect the file now.",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        name="read",
                        arguments={"file_path": str(fixture)},
                    )
                ],
            ),
            CompletionResponse(content="Recovered after the empty response."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read"]
    assert "Recovered after the empty response." in run.response
    assert any(
        message.role == Role.USER
        and "[EMPTY ASSISTANT RESPONSE]" in message.content
        for message in backend.invocations[1].messages
    )


@pytest.mark.asyncio
async def test_repeated_empty_responses_fail_honestly_after_one_retry(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=""),
            CompletionResponse(content=""),
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == []
    assert run.response == (
        "I didn't get a usable response from the model after retrying once. "
        "Please try again or switch to a different backend/model."
    )
    assert len(backend.invocations) == 2
