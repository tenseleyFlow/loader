"""Direct tests for context-owned workflow and phase controllers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import Message
from loader.runtime.context import RuntimeContext
from loader.runtime.dod import DefinitionOfDoneStore
from loader.runtime.events import TurnSummary
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.phases import TurnPhase, TurnPhaseTracker
from loader.runtime.tracing import RuntimeTracer
from loader.runtime.workflow import WorkflowDecisionKind, WorkflowMode
from loader.runtime.workflow_policy import ModeDecision
from loader.runtime.workflow_state import WorkflowStateController
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeCodeFilter:
    def reset(self) -> None:
        return None


class FakeSafeguards:
    def __init__(self) -> None:
        self.action_tracker = object()
        self.validator = object()
        self.code_filter = FakeCodeFilter()

    def filter_stream_chunk(self, content: str) -> str:
        return content

    def filter_complete_content(self, content: str) -> str:
        return content

    def should_steer(self) -> bool:
        return False

    def get_steering_message(self) -> str | None:
        return None

    def record_response(self, content: str) -> None:
        return None


class FakeSession:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.workflow_timeline = []
        self.workflow_mode: str | None = None
        self.workflow_reason_code: str | None = None
        self.active_turn_phase: str | None = None
        self.last_turn_transition_summary: str | None = None
        self.last_turn_transition_kind: str | None = None
        self.last_turn_transition_reason_code: str | None = None

    def update_runtime_state(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def append_workflow_timeline_entry(self, entry) -> None:
        self.workflow_timeline.append(entry)

    def append(self, message: Message) -> None:
        self.messages.append(message)


def build_context(temp_dir: Path) -> tuple[RuntimeContext, list[str], FakeSession]:
    registry = create_default_registry(temp_dir)
    registry.configure_workspace_root(temp_dir)
    rule_status = load_permission_rules(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
        rules=rule_status.rules,
    )
    workflow_modes: list[str] = []
    session = FakeSession()
    context = RuntimeContext(
        project_root=temp_dir,
        backend=ScriptedBackend(),
        registry=registry,
        session=session,  # type: ignore[arg-type]
        config=SimpleNamespace(force_react=False, stream=False),
        capability_profile=SimpleNamespace(supports_native_tools=True),  # type: ignore[arg-type]
        project_context=None,
        permission_policy=policy,
        permission_config_status=rule_status,
        workflow_mode="execute",
        safeguards=FakeSafeguards(),
        prompt_format="native",
        prompt_sections=["Workflow Context"],
        set_workflow_mode_callback=workflow_modes.append,
    )
    return context, workflow_modes, session


@pytest.mark.asyncio
async def test_workflow_state_controller_runs_on_runtime_context(
    temp_dir: Path,
) -> None:
    context, workflow_modes, session = build_context(temp_dir)
    controller = WorkflowStateController(
        context,
        dod_store=DefinitionOfDoneStore(temp_dir),
    )
    dod = controller.dod_store.create_or_resume("Plan the runtime context migration.")
    summary = TurnSummary(final_response="")
    events = []

    async def emit(event) -> None:
        events.append(event)

    decision = ModeDecision.transition(
        WorkflowMode.PLAN,
        reason_code="task_is_complex",
        reason_summary="task complexity favors a plan first",
        decision_kind=WorkflowDecisionKind.HANDOFF,
    )

    await controller.set_workflow_mode(
        decision,
        dod=dod,
        emit=emit,
        summary=summary,
    )

    assert workflow_modes == ["plan"]
    assert context.workflow_mode == "plan"
    assert session.workflow_mode == "plan"
    assert summary.workflow_timeline[-1].prompt_format == "native"
    assert any(event.type == "workflow_mode" for event in events)


@pytest.mark.asyncio
async def test_turn_phase_tracker_runs_on_runtime_context(
    temp_dir: Path,
) -> None:
    context, _workflow_modes, session = build_context(temp_dir)
    tracker = TurnPhaseTracker(context, RuntimeTracer())
    events = []

    async def emit(event) -> None:
        events.append(event)

    await tracker.enter(
        TurnPhase.PREPARE,
        emit,
        detail="Preparing the turn",
        reason_code="prepare_turn",
    )
    tracker.clear()

    assert session.last_turn_transition_summary == "start -> prepare [normal] Preparing the turn"
    assert session.last_turn_transition_reason_code == "prepare_turn"
    assert session.active_turn_phase is None
    assert events[0].turn_phase == "prepare"
