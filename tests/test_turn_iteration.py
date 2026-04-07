"""Tests for assistant-cycle iteration orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import CompletionResponse, ToolCall
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.turn_iteration import TurnIterationAction
from tests.helpers.runtime_harness import ScriptedBackend


def non_streaming_config() -> AgentConfig:
    """Shared config for direct turn-iteration tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=8)


async def _run_iteration(
    runtime: ConversationRuntime,
    *,
    task: str,
    requested_mode: str = "execute",
    original_task: str | None = None,
) -> tuple:
    events = []

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task=task,
        emit=capture,
        requested_mode=requested_mode,
        original_task=original_task,
        on_user_question=None,
    )
    decision = await runtime.turn_iteration.run_iteration(
        task=prepared.task,
        effective_task=prepared.effective_task,
        original_task=original_task,
        effective_max_tokens=prepared.effective_max_tokens,
        iterations=1,
        max_iterations=runtime.agent.config.max_iterations,
        actions_taken=[],
        continuation_count=0,
        empty_retry_count=0,
        max_empty_retries=5,
        extracted_iterations=0,
        max_extracted_iterations=3,
        consecutive_errors=0,
        dod=prepared.definition_of_done,
        emit=capture,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=runtime._emit_confirmation(capture),
    )
    return prepared, decision, events


@pytest.mark.asyncio
async def test_turn_iteration_completes_react_final_answer(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="Thought: I have enough context.\nFinal Answer: All set.",
            )
        ],
        supports_native_tools=False,
    )
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)

    prepared, decision, events = await _run_iteration(
        runtime,
        task="Explain whether the turn iteration controller works.",
    )

    assert agent.use_react
    assert decision.action == TurnIterationAction.COMPLETE
    assert prepared.summary.final_response == "All set."
    assert prepared.summary.assistant_messages[-1].content.endswith("Final Answer: All set.")
    assert any(event.type == "response" and event.content == "All set." for event in events)


@pytest.mark.asyncio
async def test_turn_iteration_executes_native_tool_batch_and_continues(
    temp_dir: Path,
) -> None:
    readme = temp_dir / "README.md"
    readme.write_text("Loader runtime notes\n")

    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll inspect the README first.",
                tool_calls=[
                    ToolCall(
                        id="read-1",
                        name="read",
                        arguments={"file_path": "README.md"},
                    )
                ],
            )
        ]
    )
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)

    prepared, decision, events = await _run_iteration(
        runtime,
        task="Inspect the README before continuing.",
    )

    assert decision.action == TurnIterationAction.CONTINUE
    assert decision.new_actions_taken == ["read: {'file_path': 'README.md'}"]
    assert prepared.summary.assistant_messages[-1].tool_calls[0].name == "read"
    assert len(prepared.summary.tool_result_messages) == 1
    assert "Loader runtime notes" in prepared.summary.tool_result_messages[0].content
    assert any(event.type == "tool_call" and event.tool_name == "read" for event in events)
    assert any(
        event.type == "tool_result" and "Loader runtime notes" in event.content
        for event in events
    )
