"""Tests for tool-batch execution on RuntimeContext."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.agent.reasoning import ActionVerification, ConfidenceAssessment, ConfidenceLevel
from loader.llm.base import Message, Role, ToolCall
from loader.runtime.context import RuntimeContext, RuntimeLegacyServices
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.events import AgentEvent, TurnSummary
from loader.runtime.executor import ToolExecutionOutcome, ToolExecutionState
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.recovery import RecoveryContext
from loader.runtime.tool_batches import ToolBatchRunner
from loader.runtime.tracing import RuntimeTracer
from loader.tools.base import ToolResult as RegistryToolResult
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeSession:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = list(messages)

    def append(self, message: Message) -> None:
        self.messages.append(message)


class FakeCodeFilter:
    def reset(self) -> None:
        return None


class FakeSafeguards:
    def __init__(self, *, detect_loop_result: tuple[bool, str] = (False, "")) -> None:
        self.action_tracker = object()
        self.validator = object()
        self.code_filter = FakeCodeFilter()
        self._detect_loop_result = detect_loop_result

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

    def detect_text_loop(self, content: str) -> tuple[bool, str]:
        return False, ""

    def detect_loop(self) -> tuple[bool, str]:
        return self._detect_loop_result


class FakeExecutor:
    def __init__(self, outcomes: list[ToolExecutionOutcome]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[ToolCall] = []

    async def execute_tool_call(self, tool_call: ToolCall, **_: object) -> ToolExecutionOutcome:
        self.calls.append(tool_call)
        if not self._outcomes:
            raise AssertionError("No fake tool outcome queued")
        return self._outcomes.pop(0)


def build_context(
    *,
    temp_dir: Path,
    messages: list[Message],
    safeguards: FakeSafeguards,
    assess_confidence,
    verify_action,
    recovery_context: RecoveryContext | None = None,
    confidence_scoring: bool = False,
    verification: bool = False,
    auto_recover: bool = True,
    min_confidence_for_action: int = 3,
) -> tuple[RuntimeContext, dict[str, RecoveryContext | None]]:
    registry = create_default_registry(temp_dir)
    registry.configure_workspace_root(temp_dir)
    rule_status = load_permission_rules(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
        rules=rule_status.rules,
    )
    recovery_holder = {"value": recovery_context}
    context = RuntimeContext(
        project_root=temp_dir,
        backend=ScriptedBackend(),
        registry=registry,
        session=FakeSession(messages),  # type: ignore[arg-type]
        config=SimpleNamespace(
            force_react=False,
            max_recovery_attempts=2,
            auto_recover=auto_recover,
            reasoning=SimpleNamespace(
                rollback=False,
                show_rollback_plan=False,
                completion_check=True,
                max_continuation_prompts=5,
                self_critique=False,
                confidence_scoring=confidence_scoring,
                min_confidence_for_action=min_confidence_for_action,
                verification=verification,
            ),
        ),
        capability_profile=SimpleNamespace(supports_native_tools=True),  # type: ignore[arg-type]
        project_context=None,
        permission_policy=policy,
        permission_config_status=rule_status,
        workflow_mode="execute",
        safeguards=safeguards,
        legacy=RuntimeLegacyServices(
            message_history=lambda: messages,
            drain_steering_queue=lambda: [],
            queue_steering_message=lambda message: None,
            set_workflow_mode=lambda mode: None,
            refresh_capability_profile=lambda: None,
            assess_confidence=assess_confidence,
            verify_action=verify_action,
            get_recovery_context=lambda: recovery_holder["value"],
            set_recovery_context=lambda value: recovery_holder.__setitem__("value", value),
        ),
    )
    return context, recovery_holder


def tool_outcome(
    *,
    tool_call: ToolCall,
    output: str,
    is_error: bool,
) -> ToolExecutionOutcome:
    return ToolExecutionOutcome(
        tool_call=tool_call,
        state=ToolExecutionState.EXECUTED,
        message=Message.tool_result_message(
            tool_call_id=tool_call.id,
            display_content=output,
            result_content=output,
            is_error=is_error,
        ),
        event_content=output,
        is_error=is_error,
        result_output=output,
        registry_result=RegistryToolResult(output=output, is_error=is_error),
    )


@pytest.mark.asyncio
async def test_tool_batch_runner_uses_context_for_confidence_gate(temp_dir: Path) -> None:
    captured: dict[str, str] = {}

    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        captured["context"] = context
        return ConfidenceAssessment(
            action=f"{tool_name} with {tool_args}",
            tool_name=tool_name,
            tool_args=tool_args,
            level=ConfidenceLevel.LOW,
            reasoning="Need to inspect the target first.",
            risks=["Unknown target file"],
        )

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run for skipped actions")

    context, _ = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(role=Role.USER, content="Please inspect the project."),
            Message(role=Role.ASSISTANT, content="I will read the file next."),
        ],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        confidence_scoring=True,
        min_confidence_for_action=3,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(id="read-1", name="read", arguments={"file_path": "README.md"})
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    executor = FakeExecutor([tool_outcome(tool_call=tool_call, output="unused", is_error=False)])
    result = await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=emit,
        summary=TurnSummary(final_response=""),
        dod=create_definition_of_done("Read the docs"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert result.actions_taken == []
    assert executor.calls == []
    assert "Please inspect the project." in captured["context"]
    assert context.session.messages[-1].role == Role.USER
    assert "[LOW CONFIDENCE WARNING]" in context.session.messages[-1].content
    assert [event.type for event in events] == ["confidence"]


@pytest.mark.asyncio
async def test_tool_batch_runner_tracks_recovery_with_legacy_context(temp_dir: Path) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run for failed actions")

    context, recovery_holder = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=True,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(id="bash-1", name="bash", arguments={"command": "pytest"})
    executor = FakeExecutor([tool_outcome(tool_call=tool_call, output="command failed", is_error=True)])
    summary = TurnSummary(final_response="")
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=emit,
        summary=summary,
        dod=create_definition_of_done("Run tests"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert recovery_holder["value"] is not None
    assert summary.tool_result_messages
    assert context.session.messages[-1] == summary.tool_result_messages[-1]
    assert any(event.type == "recovery" for event in events)


@pytest.mark.asyncio
async def test_tool_batch_runner_verifies_with_context_services(temp_dir: Path) -> None:
    verification_calls: list[str] = []

    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        verification_calls.append(result)
        return ActionVerification(
            tool_name=tool_name,
            tool_args=tool_args,
            expected_outcome="Success",
            actual_result=result,
            verified=False,
            discrepancies=["File contents did not match"],
            needs_correction=True,
            correction_suggestion="Read the file before editing again.",
        )

    existing_recovery = RecoveryContext(
        original_tool="edit",
        original_args={"file_path": "README.md"},
    )
    context, recovery_holder = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        recovery_context=existing_recovery,
        verification=True,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(id="read-1", name="read", arguments={"file_path": "README.md"})
    executor = FakeExecutor([tool_outcome(tool_call=tool_call, output="file contents", is_error=False)])
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=emit,
        summary=TurnSummary(final_response=""),
        dod=create_definition_of_done("Read the docs"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert verification_calls == ["file contents"]
    assert recovery_holder["value"] is None
    assert context.session.messages[-1].role == Role.TOOL
    assert context.session.messages[-1].content == "file contents"
    assert any(event.type == "verification" for event in events)
