"""Direct tests for dedicated assistant response route handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import ToolCall
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.phases import TurnPhase
from loader.runtime.repair import ToolCallAnalysis
from loader.runtime.response_route_types import ResponseRouteAction, ResponseRouteContext
from loader.runtime.turn_completion import TurnCompletionAction, TurnCompletionDecision
from tests.helpers.runtime_harness import ScriptedBackend


def non_streaming_config() -> AgentConfig:
    """Shared config for direct route-handler tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=8)


async def _prepare_context(
    runtime: ConversationRuntime,
    *,
    task: str,
    continuation_count: int = 0,
    consecutive_errors: int = 0,
) -> tuple[ResponseRouteContext, list]:
    events = []

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task=task,
        emit=capture,
        requested_mode="execute",
        original_task=None,
        on_user_question=None,
    )
    await runtime.phase_tracker.enter(
        TurnPhase.ASSISTANT,
        capture,
        detail="Requesting assistant response",
        reason_code="request_assistant_response",
    )
    context = ResponseRouteContext(
        task=prepared.task,
        effective_task=prepared.effective_task,
        iterations=1,
        max_iterations=runtime.context.config.max_iterations,
        actions_taken=[],
        continuation_count=continuation_count,
        consecutive_errors=consecutive_errors,
        dod=prepared.definition_of_done,
        summary=prepared.summary,
        executor=prepared.executor,
        rollback_plan=prepared.rollback_plan,
    )
    return context, events


@pytest.mark.asyncio
async def test_final_answer_route_handler_completes_response(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(completions=[]),
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    context, events = await _prepare_context(
        runtime,
        task="Explain whether final answers route correctly.",
        continuation_count=1,
        consecutive_errors=2,
    )

    async def capture(event) -> None:
        events.append(event)

    decision = await runtime.response_router.final_answer_handler.handle(
        analysis=ToolCallAnalysis(
            content="All set.",
            response_content="Final Answer: All set.",
            is_final_answer=True,
            final_response="All set.",
        ),
        context=context,
        emit=capture,
    )

    assert decision.action == ResponseRouteAction.COMPLETE
    assert context.summary.final_response == "All set."
    assert context.summary.assistant_messages[-1].content == "Final Answer: All set."
    assert any(event.type == "response" and event.content == "All set." for event in events)


@pytest.mark.asyncio
async def test_tool_batch_route_handler_finalizes_halted_batch(
    temp_dir: Path,
) -> None:
    config = non_streaming_config()
    config.auto_recover = False
    agent = Agent(
        backend=ScriptedBackend(completions=[]),
        config=config,
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    context, events = await _prepare_context(
        runtime,
        task="Inspect the missing file and recover honestly.",
        consecutive_errors=2,
    )

    async def capture(event) -> None:
        events.append(event)

    decision = await runtime.response_router.tool_batch_handler.handle(
        analysis=ToolCallAnalysis(
            content="I'll inspect the file first.",
            response_content="I'll inspect the file first.",
            tool_calls=[
                ToolCall(
                    id="read-missing",
                    name="read",
                    arguments={"file_path": "missing.md"},
                )
            ],
            tool_source="native",
        ),
        pending_tool_calls_seen=set(),
        context=context,
        emit=capture,
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=runtime._emit_confirmation(capture),
    )

    assert decision.action == ResponseRouteAction.FINALIZE
    assert decision.finalize_reason_code == "tool_batch_halted"
    assert decision.new_actions_taken == ["read: {'file_path': 'missing.md'}"]
    assert any(event.type == "tool_call" and event.tool_name == "read" for event in events)


@pytest.mark.asyncio
async def test_text_completion_route_handler_maps_continue_decision(
    temp_dir: Path,
) -> None:
    agent = Agent(
        backend=ScriptedBackend(completions=[]),
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    context, events = await _prepare_context(
        runtime,
        task="Continue the investigation.",
        continuation_count=2,
        consecutive_errors=1,
    )

    async def capture(event) -> None:
        events.append(event)

    async def fake_handle_text_response(**kwargs) -> TurnCompletionDecision:
        return TurnCompletionDecision(
            action=TurnCompletionAction.CONTINUE,
            continuation_count=3,
        )

    runtime.response_router.text_completion_handler.turn_completion.handle_text_response = (  # type: ignore[method-assign]
        fake_handle_text_response
    )

    decision = await runtime.response_router.text_completion_handler.handle(
        analysis=ToolCallAnalysis(
            content="I looked into it.",
            response_content="I looked into it.",
        ),
        context=context,
        emit=capture,
    )

    assert decision.action == ResponseRouteAction.CONTINUE
    assert decision.continuation_count == 3
    assert decision.consecutive_errors == 1
