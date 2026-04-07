"""Deterministic coverage for current runtime repair heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, Message, Role, ToolCall
from tests.helpers.runtime_harness import ScriptedBackend, run_scenario


def non_streaming_config() -> AgentConfig:
    """Shared deterministic config for repair-flow tests."""

    config = AgentConfig(auto_context=False, stream=False, max_iterations=8)
    config.reasoning.completion_check = False
    return config


def completion_check_config() -> AgentConfig:
    """Shared config for continuation-driven repair tests."""

    config = AgentConfig(auto_context=False, stream=False, max_iterations=8)
    config.reasoning.completion_check = True
    return config


def self_critique_config() -> AgentConfig:
    """Shared config for self-critique characterization."""

    config = AgentConfig(auto_context=False, stream=False, max_iterations=8)
    config.reasoning.completion_check = False
    config.reasoning.self_critique = True
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


@pytest.mark.asyncio
async def test_fake_tool_narration_no_longer_injects_scolding_prompt(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content="I ran the command and created the file."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == []
    assert run.response == "I ran the command and created the file."
    assert len(backend.invocations) == 1


@pytest.mark.asyncio
async def test_deflection_response_no_longer_injects_use_your_tools_prompt(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content="You can read the file to inspect its contents."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == []
    assert run.response == "You can read the file to inspect its contents."
    assert len(backend.invocations) == 1


@pytest.mark.asyncio
async def test_non_mutating_completion_returns_directly_without_text_bailout(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content="Done."),
        ]
    )

    run = await run_scenario(
        "Create a hello.py file and run it.",
        backend,
        config=completion_check_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == []
    assert run.response == "Done."
    assert len(backend.invocations) == 1
    assert not any(
        event.type == "error" and "Text loop detected" in event.content
        for event in run.events
    )


@pytest.mark.asyncio
async def test_post_action_follow_up_suffix_is_not_appended_to_final_response(
    temp_dir: Path,
) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("repair baseline\n")
    backend = ScriptedBackend(
        completions=[
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
            CompletionResponse(content="Inspected the file successfully."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read"]
    assert run.response == "Inspected the file successfully."


@pytest.mark.asyncio
async def test_action_loop_bailout_stops_repeating_tool_pattern(
    temp_dir: Path,
) -> None:
    first = temp_dir / "first.txt"
    second = temp_dir / "second.txt"
    first.write_text("first\n")
    second.write_text("second\n")
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll inspect both files.",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        name="read",
                        arguments={"file_path": str(first)},
                    ),
                    ToolCall(
                        id="read-2",
                        name="read",
                        arguments={"file_path": str(second)},
                    ),
                ],
            ),
            CompletionResponse(
                content="I'll inspect them again.",
                tool_calls=[
                    ToolCall(
                        id="read-3",
                        name="read",
                        arguments={"file_path": str(first)},
                    ),
                    ToolCall(
                        id="read-4",
                        name="read",
                        arguments={"file_path": str(second)},
                    ),
                ],
            ),
        ]
    )

    run = await run_scenario(
        "Read both fixture files.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read", "read", "read", "read"]
    assert run.response == (
        "I noticed I was repeating the same actions. "
        "Let me know what you'd like me to do differently."
    )
    assert any(
        event.type == "error" and "Loop detected: Repeating pattern detected" in event.content
        for event in run.events
    )


@pytest.mark.asyncio
async def test_long_code_response_no_longer_reroutes_for_self_critique(
    temp_dir: Path,
) -> None:
    initial_response = (
        "def build_report():\n"
        "    summary = []\n"
        "    for item in range(20):\n"
        "        summary.append(f'result-{item}')\n"
        "    return '\\n'.join(summary)\n\n"
    ) * 4
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=initial_response),
        ]
    )

    run = await run_scenario(
        "Explain the code changes clearly.",
        backend,
        config=self_critique_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == []
    assert run.response == initial_response.strip()
    assert not any(event.type == "critique" for event in run.events)
    assert len(backend.invocations) == 1
