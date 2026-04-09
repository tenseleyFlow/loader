"""Tests for turn bootstrap and workflow preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import CompletionResponse, ToolCall
from loader.runtime.completion_trace import CompletionTraceEntry
from loader.runtime.conversation import ConversationRuntime
from tests.helpers.runtime_harness import ScriptedBackend


def non_streaming_config() -> AgentConfig:
    """Shared config for direct turn-preparation tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=8)


@pytest.mark.asyncio
async def test_turn_preparation_bootstraps_execute_turn_state(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    events = []
    task = "Update README.md heading."
    agent.session.last_completion_decision_code = "verification_passed"
    agent.session.last_completion_decision_summary = (
        "accepted the response after verification evidence passed"
    )
    agent.session.append_completion_trace_entry(
        CompletionTraceEntry(
            stage="definition_of_done",
            outcome="complete",
            decision_code="verification_passed",
            decision_summary="accepted the response after verification evidence passed",
        )
    )

    async def capture(event) -> None:
        events.append(event)

    prepared = await runtime.turn_preparation.prepare(
        task=task,
        emit=capture,
        requested_mode="execute",
        original_task=None,
        on_user_question=None,
    )

    assert prepared.task == task
    assert prepared.effective_task == task
    assert prepared.executor is not None
    assert prepared.rollback_plan is not None
    assert prepared.effective_max_tokens >= 512
    assert prepared.summary.workflow_mode == "execute"
    assert prepared.definition_of_done.current_mode == "execute"
    assert prepared.definition_of_done.storage_path is not None
    assert agent.session.current_task == task
    assert agent.session.active_dod_path == prepared.definition_of_done.storage_path
    assert agent.session.workflow_mode == "execute"
    assert agent.session.last_completion_decision_code is None
    assert agent.session.completion_trace == []
    assert any(event.type == "dod_status" for event in events)
    assert any(
        event.type == "workflow_mode" and event.workflow_mode == "execute"
        for event in events
    )


@pytest.mark.asyncio
async def test_turn_preparation_can_bootstrap_clarify_handoff(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I need one clarification before I proceed.",
                tool_calls=[
                    ToolCall(
                        id="ask-1",
                        name="AskUserQuestion",
                        arguments={
                            "question": (
                                "Should I keep the runtime change inside "
                                "src/loader/runtime/workflow_lanes.py?"
                            ),
                        },
                    )
                ],
            ),
            CompletionResponse(
                content="\n".join(
                    [
                        "## Task Statement",
                        "Tighten Loader runtime clarify behavior.",
                        "",
                        "## Desired Outcome",
                        "- Keep the clarify handoff grounded in one runtime seam.",
                        "",
                        "## Non Goals",
                        "- Do not broaden into unrelated CLI changes.",
                        "",
                        "## Decision Boundaries",
                        "- Stop and confirm before crossing into other runtime modules.",
                        "",
                        "## Constraints",
                        "- Stay within the current repository.",
                        "",
                        "## Likely Touchpoints",
                        "- src/loader/runtime/workflow_lanes.py",
                        "",
                        "## Acceptance Criteria",
                        "- workflow_lanes.py remains the primary touchpoint.",
                    ]
                )
            ),
        ]
    )
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    events = []
    asked_questions: list[str] = []

    async def capture(event) -> None:
        events.append(event)

    async def answer(question: str, _: list[str] | None) -> str:
        asked_questions.append(question)
        return "Yes, keep it inside workflow_lanes.py."

    prepared = await runtime.turn_preparation.prepare(
        task="Tighten Loader runtime clarify behavior.",
        emit=capture,
        requested_mode="clarify",
        original_task=None,
        on_user_question=answer,
    )

    assert asked_questions == [
        "Should I keep the runtime change inside src/loader/runtime/workflow_lanes.py?"
    ]
    assert prepared.summary.workflow_mode == "execute"
    assert prepared.definition_of_done.current_mode == "execute"
    assert prepared.definition_of_done.clarify_brief is not None
    assert Path(prepared.definition_of_done.clarify_brief).exists()
    assert agent.session.workflow_mode == "execute"
    assert any(
        entry.kind == "clarify_exit" for entry in prepared.summary.workflow_timeline
    )
    assert any(
        entry.reason_code.startswith("post_clarify_")
        for entry in prepared.summary.workflow_timeline
    )
    assert [
        event.workflow_mode
        for event in events
        if event.type == "workflow_mode" and event.workflow_mode
    ] == ["clarify", "execute"]
