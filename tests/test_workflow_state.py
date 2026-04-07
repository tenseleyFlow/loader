"""Tests for workflow-state coordination helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig
from loader.runtime.conversation import ConversationRuntime
from loader.runtime.dod import DefinitionOfDoneStore
from loader.runtime.events import TurnSummary
from loader.runtime.workflow import WorkflowDecisionKind, WorkflowMode
from loader.runtime.workflow_policy import ModeDecision
from tests.helpers.runtime_harness import ScriptedBackend


def non_streaming_config() -> AgentConfig:
    """Shared config for direct workflow-state tests."""

    return AgentConfig(auto_context=False, stream=False, max_iterations=8)


@pytest.mark.asyncio
async def test_workflow_state_applies_mode_to_session_dod_and_timeline(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    agent.prompt_format = "native"
    agent.prompt_sections = ["Workflow Context", "Runtime Config"]
    runtime = ConversationRuntime(agent)
    dod = runtime.dod_store.create_or_resume("Tighten the workflow state controller.")
    brief_path = temp_dir / "brief.md"
    brief_path.write_text("# Task Brief\n\n## Likely Touchpoints\n- src/loader/runtime\n")
    dod.clarify_brief = str(brief_path)
    summary = TurnSummary(final_response="")
    events = []

    async def capture(event) -> None:
        events.append(event)

    decision = ModeDecision.transition(
        WorkflowMode.PLAN,
        reason_code="task_is_complex",
        reason_summary="task complexity favors planning first",
        decision_kind=WorkflowDecisionKind.HANDOFF,
        scheduled_next_mode=WorkflowMode.EXECUTE,
    )

    await runtime.workflow_state.set_workflow_mode(
        decision,
        dod=dod,
        emit=capture,
        summary=summary,
    )

    reloaded = DefinitionOfDoneStore(temp_dir).load(Path(dod.storage_path))

    assert agent.workflow_mode == "plan"
    assert agent.session.workflow_mode == "plan"
    assert summary.workflow_mode == "plan"
    assert summary.workflow_reason_code == "task_is_complex"
    assert summary.workflow_decision_kind == "handoff"
    assert dod.current_mode == "plan"
    assert dod.mode_history[-1] == "plan"
    assert reloaded.current_mode == "plan"
    assert summary.definition_of_done is dod
    assert summary.workflow_timeline[-1].mode == "plan"
    assert summary.workflow_timeline[-1].kind == "handoff"
    assert summary.workflow_timeline[-1].artifact_paths == [str(brief_path)]
    assert summary.workflow_timeline[-1].prompt_format == "native"
    assert any(
        event.type == "workflow_mode" and event.workflow_mode == "plan"
        for event in events
    )


def test_workflow_state_appends_execute_bridge_once(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend()
    agent = Agent(
        backend=backend,
        config=non_streaming_config(),
        project_root=temp_dir,
    )
    runtime = ConversationRuntime(agent)
    dod = runtime.dod_store.create_or_resume("Use the persisted workflow artifacts.")

    brief_path = temp_dir / "brief.md"
    brief_path.write_text("# Task Brief\n\n## In Scope\n- Runtime workflow\n")
    plan_path = temp_dir / "plan.md"
    plan_path.write_text("# Implementation Plan\n\n- Extract the controller\n")
    verify_path = temp_dir / "verify.md"
    verify_path.write_text("# Verification Plan\n\n- Run uv run pytest -q\n")
    dod.clarify_brief = str(brief_path)
    dod.implementation_plan = str(plan_path)
    dod.verification_plan = str(verify_path)

    runtime.workflow_state.maybe_append_execute_bridge(dod)
    runtime.workflow_state.maybe_append_execute_bridge(dod)

    bridge_messages = [
        message.content
        for message in agent.session.messages
        if "[WORKFLOW BRIDGE]" in message.content
    ]

    assert len(bridge_messages) == 1
    assert "Use the clarify brief below as the requirements source of truth." in bridge_messages[0]
    assert "# Implementation Plan" in bridge_messages[0]
    assert "# Verification Plan" in bridge_messages[0]
