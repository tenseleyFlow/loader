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
    policy_entries = [
        entry
        for entry in run.agent.last_turn_summary.workflow_timeline
        if entry.kind.startswith(("repair_", "completion_"))
    ]
    assert [entry.kind for entry in policy_entries] == [
        "repair_retry",
        "completion_complete",
    ]
    assert policy_entries[0].policy_stage == "empty_response"
    assert any(
        message.role == Role.USER
        and "[EMPTY ASSISTANT RESPONSE]" in message.content
        for message in backend.invocations[1].messages
    )


@pytest.mark.asyncio
async def test_empty_response_retry_carries_forward_confirmed_progress(
    temp_dir: Path,
) -> None:
    target = temp_dir / "hello.py"
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll create the file now.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(target),
                            "content": "print('hello')\n",
                        },
                    )
                ],
            ),
            CompletionResponse(content=""),
            CompletionResponse(content="Recovered after the empty response."),
        ]
    )

    run = await run_scenario(
        "Create hello.py with a greeting.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert "Recovered after the empty response." in run.response
    retry_messages = [
        message.content
        for message in backend.invocations[2].messages
        if message.role == Role.USER and "[EMPTY ASSISTANT RESPONSE]" in message.content
    ]
    assert retry_messages
    assert "retry 1/2" in retry_messages[0]
    assert "Continue from the confirmed progress below instead of restarting." in retry_messages[0]
    assert "hello.py" in retry_messages[0]


@pytest.mark.asyncio
async def test_empty_response_retry_budget_resets_after_successful_turn(
    temp_dir: Path,
) -> None:
    first = temp_dir / "one.txt"
    second = temp_dir / "two.txt"
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=""),
            CompletionResponse(
                content="I'll create the first file now.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(first),
                            "content": "one\n",
                        },
                    )
                ],
            ),
            CompletionResponse(content=""),
            CompletionResponse(
                content="I'll create the second file now.",
                tool_calls=[
                    ToolCall(
                        id="write-2",
                        name="write",
                        arguments={
                            "file_path": str(second),
                            "content": "two\n",
                        },
                    )
                ],
            ),
            CompletionResponse(content="Both files are created."),
        ]
    )

    run = await run_scenario(
        "Create one.txt and two.txt.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert run.response.startswith("Both files are created.")
    retry_messages: list[str] = []
    for invocation in backend.invocations:
        for message in invocation.messages:
            if message.role != Role.USER or "[EMPTY ASSISTANT RESPONSE]" not in message.content:
                continue
            if retry_messages and retry_messages[-1] == message.content:
                continue
            retry_messages.append(message.content)
    assert len(retry_messages) >= 2
    assert all("retry 2/2" not in message for message in retry_messages)
    assert sum("retry 1/2" in message for message in retry_messages) >= 2


@pytest.mark.asyncio
async def test_repeated_empty_responses_fail_honestly_after_one_retry(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=""),
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
        "I didn't get a usable response from the model after retrying 2 times. "
        "Please try again or switch to a different backend/model."
    )
    assert len(backend.invocations) == 3
    assert [entry.kind for entry in run.agent.last_turn_summary.workflow_timeline[-3:]] == [
        "repair_retry",
        "repair_retry",
        "repair_fail",
    ]
    assert run.agent.last_turn_summary.workflow_timeline[-1].reason_code == (
        "empty_response_retry_exhausted"
    )
    assert run.agent.session.last_turn_transition_kind == "terminal"
    assert run.agent.session.last_turn_transition_reason_code == (
        "empty_response_retry_exhausted"
    )


@pytest.mark.asyncio
async def test_raw_text_tool_recovery_budget_fails_honestly(
    temp_dir: Path,
) -> None:
    for name in ("one.txt", "two.txt", "three.txt", "four.txt"):
        (temp_dir / name).write_text(f"{name}\n")

    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content='{"name": "read", "arguments": {"file_path": "one.txt"}}'
            ),
            CompletionResponse(
                content='{"name": "read", "arguments": {"file_path": "two.txt"}}'
            ),
            CompletionResponse(
                content='{"name": "read", "arguments": {"file_path": "three.txt"}}'
            ),
            CompletionResponse(
                content='{"name": "read", "arguments": {"file_path": "four.txt"}}'
            ),
        ]
    )

    run = await run_scenario(
        "Inspect the text fixtures.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read", "read", "read"]
    assert run.response == (
        "I couldn't safely continue because the model kept emitting raw-text "
        "tool calls instead of proper tool invocations. Please try again or "
        "switch to a different backend/model."
    )
    assert [entry.kind for entry in run.agent.last_turn_summary.workflow_timeline[-4:]] == [
        "repair_retry",
        "repair_retry",
        "repair_retry",
        "repair_fail",
    ]
    assert run.agent.last_turn_summary.workflow_timeline[-1].reason_code == (
        "raw_text_tool_recovery_exhausted"
    )
    assert "Let me know if you'd like me to continue" not in run.response
