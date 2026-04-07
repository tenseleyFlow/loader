"""Deterministic coverage for current runtime repair heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
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


def test_fresh_agent_messages_are_disconnected_from_session_history(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    agent.session.append(
        Message(role=Role.USER, content="Create allowed.txt with a greeting.")
    )

    assert agent.messages == []
    assert [message.content for message in agent.session.messages] == [
        "Create allowed.txt with a greeting."
    ]


@pytest.mark.asyncio
async def test_empty_response_repair_injects_retry_prompt_and_recovers(
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
        message.role == Role.ASSISTANT
        and "Great! Now let me proceed with the task." in message.content
        for message in backend.invocations[1].messages
    )


@pytest.mark.asyncio
async def test_fake_tool_narration_repair_injects_scolding_prompt(
    temp_dir: Path,
) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("repair baseline\n")
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I ran the command and created the file."
            ),
            CompletionResponse(
                content="I'll inspect the real tool result now.",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        name="read",
                        arguments={"file_path": str(fixture)},
                    )
                ],
            ),
            CompletionResponse(content="Recovered after fake tool narration."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read"]
    assert "Recovered after fake tool narration." in run.response
    assert any(
        message.role == Role.USER
        and "CRITICAL ERROR: You are PRETENDING to use tools" in message.content
        for message in backend.invocations[1].messages
    )


@pytest.mark.asyncio
async def test_deflection_repair_injects_use_your_tools_prompt(
    temp_dir: Path,
) -> None:
    fixture = temp_dir / "fixture.txt"
    fixture.write_text("repair baseline\n")
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="You can read the file to inspect its contents."
            ),
            CompletionResponse(
                content="I'll inspect the real tool result now.",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        name="read",
                        arguments={"file_path": str(fixture)},
                    )
                ],
            ),
            CompletionResponse(content="Recovered after deflection."),
        ]
    )

    run = await run_scenario(
        "Read the fixture file.",
        backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == ["read"]
    assert "Recovered after deflection." in run.response
    assert any(
        message.role == Role.USER
        and "Please use your tools to execute the task" in message.content
        for message in backend.invocations[1].messages
    )


@pytest.mark.asyncio
async def test_text_loop_bailout_stops_after_repeated_continuation_response(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content="Done."),
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
    assert run.response == (
        "I seem to be repeating myself. "
        "Let me know if you'd like me to try a different approach."
    )
    assert any(
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
async def test_self_critique_reroutes_long_code_response_for_revision(
    temp_dir: Path,
) -> None:
    initial_response = (
        "def build_report():\n"
        "    summary = []\n"
        "    for item in range(20):\n"
        "        summary.append(f'result-{item}')\n"
        "    return '\\n'.join(summary)\n\n"
    ) * 4
    critique_json = (
        '{"issues": ["Missing explanation"], '
        '"suggestions": ["Summarize the code changes more clearly"], '
        '"should_revise": true, "severity": "moderate"}'
    )
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(content=initial_response),
            CompletionResponse(content=critique_json),
            CompletionResponse(content="Revised answer with a concise explanation."),
        ]
    )

    run = await run_scenario(
        "Explain the code changes clearly.",
        backend,
        config=self_critique_config(),
        project_root=temp_dir,
    )

    assert tool_event_names(run) == []
    assert run.response == "Revised answer with a concise explanation."
    assert any(event.type == "critique" for event in run.events)
    assert any(
        "Review your response and identify potential issues."
        in invocation.messages[0].content
        for invocation in backend.invocations
        if invocation.mode == "complete" and invocation.messages
    )
    assert any(
        message.role == Role.USER and "[SELF-CRITIQUE] Review your response:" in message.content
        for message in backend.invocations[-1].messages
    )
