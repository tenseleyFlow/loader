"""Runtime coverage for Sprint 04 workflow tools."""

from __future__ import annotations

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, ToolCall
from tests.helpers.runtime_harness import ScriptedBackend, run_scenario


def non_streaming_config() -> AgentConfig:
    """Shared deterministic config for runtime tool tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=4)


async def _answer(question: str, options: list[str] | None) -> str:
    assert "Which path" in question
    assert options == ["Plan first", "Execute now"]
    return "1"


@pytest.mark.asyncio
async def test_ask_user_question_round_trips_through_runtime() -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I need one clarification.",
                tool_calls=[
                    ToolCall(
                        id="ask-1",
                        name="AskUserQuestion",
                        arguments={
                            "question": "Which path should we take?",
                            "options": ["Plan first", "Execute now"],
                        },
                    )
                ],
            ),
            CompletionResponse(content="We'll plan first."),
        ]
    )

    run = await run_scenario(
        "Implement the task, but ask me which path to take first.",
        backend,
        config=non_streaming_config(),
        on_user_question=_answer,
    )

    tool_events = [event for event in run.events if event.type == "tool_call"]
    tool_results = [event for event in run.events if event.type == "tool_result"]

    assert "We'll plan first." in run.response
    assert [event.tool_name for event in tool_events] == ["AskUserQuestion"]
    assert any("Plan first" in event.content for event in tool_results)
