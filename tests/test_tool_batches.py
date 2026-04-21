"""Tests for tool-batch execution on RuntimeContext."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import Message, Role, ToolCall
from loader.runtime.context import RuntimeContext
from loader.runtime.dod import (
    DefinitionOfDoneStore,
    VerificationEvidence,
    create_definition_of_done,
)
from loader.runtime.events import AgentEvent, TurnSummary
from loader.runtime.executor import ToolExecutionOutcome, ToolExecutionState
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.reasoning_types import (
    ActionVerification,
    ConfidenceAssessment,
    ConfidenceLevel,
)
from loader.runtime.recovery import RecoveryContext
from loader.runtime.tool_batches import ToolBatchRunner
from loader.tools.base import ToolResult as RegistryToolResult
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeSession:
    def __init__(self, messages: list[Message]) -> None:
        self.messages = list(messages)
        self.workflow_timeline = []

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def append_workflow_timeline_entry(self, entry) -> None:
        self.workflow_timeline.append(entry)


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
) -> RuntimeContext:
    registry = create_default_registry(temp_dir)
    registry.configure_workspace_root(temp_dir)
    rule_status = load_permission_rules(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
        rules=rule_status.rules,
    )
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
        reasoning=SimpleNamespace(
            assess_confidence=assess_confidence,
            verify_action=verify_action,
        ),
        recovery_context=recovery_context,
    )
    return context


def tool_outcome(
    *,
    tool_call: ToolCall,
    output: str,
    is_error: bool,
    metadata: dict[str, object] | None = None,
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
        registry_result=RegistryToolResult(
            output=output,
            is_error=is_error,
            metadata=metadata or {},
        ),
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

    context = build_context(
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
    event_types = [event.type for event in events]
    assert "confidence" in event_types


@pytest.mark.asyncio
async def test_tool_batch_runner_tracks_recovery_with_legacy_context(temp_dir: Path) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run for failed actions")

    context = build_context(
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

    assert context.recovery_context is not None
    assert summary.tool_result_messages
    assert context.session.messages[-1] == summary.tool_result_messages[-1]
    assert any(event.type == "recovery" for event in events)


@pytest.mark.asyncio
async def test_tool_batch_runner_emits_tool_metadata(temp_dir: Path) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="bash-1",
        name="bash",
        arguments={"command": "python -m http.server 8000", "background": True},
    )
    metadata = {
        "job_id": "bash-1",
        "status": "running",
        "background": True,
    }
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Started bash job bash-1",
                is_error=False,
                metadata=metadata,
            )
        ]
    )
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=emit,
        summary=TurnSummary(final_response=""),
        dod=create_definition_of_done("Launch a preview server"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    tool_result = next(event for event in events if event.type == "tool_result")
    assert tool_result.tool_metadata == metadata


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
    context = build_context(
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
    assert context.recovery_context is existing_recovery
    assert existing_recovery.successful_steps == [
        ("read", {"file_path": "README.md"})
    ]
    assert context.session.messages[-1].role == Role.TOOL
    assert context.session.messages[-1].content == "file contents"
    assert any(event.type == "verification" for event in events)


@pytest.mark.asyncio
async def test_tool_batch_runner_preserves_recovery_context_across_diagnostic_success(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    existing_recovery = RecoveryContext(
        original_tool="read",
        original_args={"file_path": "chapters/04-data-types.html"},
    )
    existing_recovery.add_attempt(
        "read",
        {"file_path": "chapters/04-data-types.html"},
        "File not found",
    )
    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        recovery_context=existing_recovery,
        auto_recover=False,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="bash-1",
        name="bash",
        arguments={"command": "ls chapters"},
    )
    executor = FakeExecutor(
        [tool_outcome(tool_call=tool_call, output="01-introduction.html", is_error=False)]
    )

    summary = TurnSummary(final_response="")
    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=create_definition_of_done("Fix the chapter links"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert context.recovery_context is existing_recovery
    assert existing_recovery.successful_steps == [
        ("bash", {"command": "ls chapters"})
    ]


@pytest.mark.asyncio
async def test_tool_batch_runner_clears_recovery_context_after_successful_mutation(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    existing_recovery = RecoveryContext(
        original_tool="read",
        original_args={"file_path": "chapters/04-data-types.html"},
    )
    existing_recovery.add_attempt(
        "read",
        {"file_path": "chapters/04-data-types.html"},
        "File not found",
    )
    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        recovery_context=existing_recovery,
        auto_recover=False,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="patch-1",
        name="patch",
        arguments={
            "file_path": "index.html",
            "hunks": [{"old_start": 1, "old_lines": 1, "new_start": 1, "new_lines": 1, "lines": ["-a", "+b"]}],
        },
    )
    executor = FakeExecutor(
        [tool_outcome(tool_call=tool_call, output="Patched index.html", is_error=False)]
    )

    summary = TurnSummary(final_response="")
    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=create_definition_of_done("Fix the chapter links"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert context.recovery_context is None


@pytest.mark.asyncio
async def test_tool_batch_runner_queues_duplicate_observation_nudge(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    messages = [
        Message(
            role=Role.TOOL,
            content=(
                "Observation [glob]: Result: "
                f"{temp_dir}/chapters/01-introduction.html\n"
                f"{temp_dir}/chapters/02-setup.html\n"
                f"{temp_dir}/chapters/03-basics.html"
            ),
            tool_results=[],
        ),
        Message(
            role=Role.ASSISTANT,
            content="I already inspected the first chapter title.",
            tool_calls=[
                ToolCall(
                    id="read-ch1",
                    name="read",
                    arguments={"file_path": str(temp_dir / 'chapters' / '01-introduction.html')},
                )
            ],
        ),
        Message.tool_result_message(
            tool_call_id="read-ch1",
            display_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
            result_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
        ),
        Message(
            role=Role.ASSISTANT,
            content="I should update the index now.",
            tool_calls=[
                ToolCall(
                    id="read-index",
                    name="read",
                    arguments={"file_path": str(temp_dir / 'index.html')},
                )
            ],
        ),
    ]
    context = build_context(
        temp_dir=temp_dir,
        messages=messages,
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.session.current_task = (
        f"Update {temp_dir / 'index.html'} with the right chapter links."
    )
    queued_messages: list[str] = []
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="read-dup",
        name="read",
        arguments={"file_path": str(temp_dir / "index.html")},
    )
    duplicate_message = (
        "[Skipped - duplicate action: Already read "
        f"{temp_dir / 'index.html'} recently without any intervening changes; "
        "reuse the earlier read result instead of rereading]"
    )
    executor = FakeExecutor(
        [
            ToolExecutionOutcome(
                tool_call=tool_call,
                state=ToolExecutionState.DUPLICATE,
                message=Message.tool_result_message(
                    tool_call_id=tool_call.id,
                    display_content=duplicate_message,
                    result_content=duplicate_message,
                ),
                event_content=duplicate_message,
                is_error=False,
                result_output=duplicate_message,
            )
        ]
    )

    summary = TurnSummary(final_response="")
    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=create_definition_of_done("Fix the chapter links"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert len(queued_messages) == 1
    assert "Reuse the earlier observation instead of repeating it." in queued_messages[0]
    assert "01-introduction.html = Chapter 1: Introduction to Fortran" in queued_messages[0]
    assert "index.html" in queued_messages[0]


@pytest.mark.asyncio
async def test_tool_batch_runner_proactively_queues_verified_html_inventory(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    chapters = temp_dir / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    (chapters / "02-setup.html").write_text(
        "<h1>Chapter 2: Setting Up Your Environment</h1>\n"
    )
    (temp_dir / "index.html").write_text("<ul></ul>\n")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.session.current_task = (
        f"Update {temp_dir / 'index.html'} so the chapter links match the sibling files."
    )
    queued_messages: list[str] = []
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="glob-1",
        name="glob",
        arguments={"path": str(chapters), "pattern": "*.html"},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="\n".join(
                    [
                        str(chapters / "01-introduction.html"),
                        str(chapters / "02-setup.html"),
                    ]
                ),
                is_error=False,
            )
        ]
    )

    summary = TurnSummary(final_response="")
    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=create_definition_of_done("Fix the chapter links"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert len(queued_messages) == 1
    assert "verified sibling inventory" in queued_messages[0]
    assert "chapters/01-introduction.html = Chapter 1: Introduction to Fortran" in queued_messages[0]
    assert str(temp_dir / "index.html") in queued_messages[0]
    assert len(summary.tool_result_messages) == 1
    assert (
        "Verified chapter inventory: chapters/01-introduction.html = Chapter 1: Introduction to Fortran"
        in summary.tool_result_messages[0].content
    )


@pytest.mark.asyncio
async def test_tool_batch_runner_marks_validated_html_toc_completion_after_successful_edit(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    chapters = temp_dir / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    (chapters / "02-setup.html").write_text(
        "<h1>Chapter 2: Setting Up Your Environment</h1>\n"
    )
    index_path = temp_dir / "index.html"
    old_block = (
        '<ul class="chapter-list">\n'
        '    <li><a href="chapters/01-old.html">Chapter 1: Old</a></li>\n'
        '    <li><a href="chapters/02-old.html">Chapter 2: Old</a></li>\n'
        "</ul>\n"
    )
    new_block = (
        '<ul class="chapter-list">\n'
        '    <li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>\n'
        '    <li><a href="chapters/02-setup.html">Chapter 2: Setting Up Your Environment</a></li>\n'
        "</ul>\n"
    )
    index_path.write_text(new_block)

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    queued_messages: list[str] = []
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="edit-1",
        name="edit",
        arguments={
            "file_path": str(index_path),
            "old_string": old_block,
            "new_string": new_block,
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=f"Successfully edited {index_path}",
                is_error=False,
            )
        ]
    )

    summary = TurnSummary(final_response="")
    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=create_definition_of_done("Fix the chapter links"),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert any(
        "Semantic verification preview: validated 2 toc links in index.html"
        in message.content
        for message in summary.tool_result_messages
    )
    assert len(queued_messages) == 1
    assert "already satisfies the verified chapter-link constraints" in queued_messages[0]
    assert "Do not reread `index.html` or files in `chapters/`" in queued_messages[0]


async def _noop_emit(event: AgentEvent) -> None:
    return None


@pytest.mark.asyncio
async def test_tool_batch_runner_marks_verification_planned_after_new_mutation(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="write-1",
        name="write",
        arguments={"file_path": str(temp_dir / "README.md"), "content": "updated\n"},
    )
    executor = FakeExecutor(
        [tool_outcome(tool_call=tool_call, output="wrote file", is_error=False)]
    )
    summary = TurnSummary(final_response="")
    dod = create_definition_of_done("Update README and verify it still works.")
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=emit,
        summary=summary,
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert dod.last_verification_result == "planned"
    assert dod.verification_commands
    assert "Collect verification evidence" in dod.pending_items
    assert dod.active_verification_attempt_id == "verification-attempt-1"
    assert dod.active_verification_attempt_number == 1
    assert summary.workflow_timeline[-1].reason_code == "verification_planned"
    assert summary.workflow_timeline[-1].policy_outcome == "planned"
    assert summary.workflow_timeline[-1].verification_observations[0].status == "planned"
    assert (
        summary.workflow_timeline[-1].verification_observations[0].attempt_id
        == "verification-attempt-1"
    )
    assert (
        summary.workflow_timeline[-1].verification_observations[0].attempt_number == 1
    )


@pytest.mark.asyncio
async def test_tool_batch_runner_marks_passed_verification_stale_after_new_mutation(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should be disabled in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="write-1",
        name="write",
        arguments={"file_path": str(temp_dir / "README.md"), "content": "updated\n"},
    )
    executor = FakeExecutor(
        [tool_outcome(tool_call=tool_call, output="wrote file", is_error=False)]
    )
    summary = TurnSummary(final_response="")
    dod = create_definition_of_done("Update README and verify it still works.")
    dod.verification_commands = ["uv run pytest -q"]
    dod.last_verification_result = "passed"
    dod.verification_attempt_counter = 1
    dod.active_verification_attempt_id = "verification-attempt-1"
    dod.active_verification_attempt_number = 1
    dod.evidence = [
        VerificationEvidence(
            command="uv run pytest -q",
            passed=True,
            stdout="401 passed",
            kind="test",
        )
    ]
    dod.completed_items.append("Collect verification evidence")
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=emit,
        summary=summary,
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert dod.last_verification_result == "stale"
    assert dod.evidence == []
    assert "Collect verification evidence" in dod.pending_items
    assert "Collect verification evidence" not in dod.completed_items
    assert dod.active_verification_attempt_id == "verification-attempt-2"
    assert dod.active_verification_attempt_number == 2
    assert summary.workflow_timeline[-1].reason_code == "verification_stale"
    assert summary.workflow_timeline[-1].policy_outcome == "stale"
    assert summary.workflow_timeline[-1].verification_observations[0].status == "stale"
    assert (
        summary.workflow_timeline[-1].verification_observations[0].attempt_id
        == "verification-attempt-1"
    )
    assert (
        summary.workflow_timeline[-1].verification_observations[0].attempt_number == 1
    )
    assert (
        summary.workflow_timeline[-1].verification_observations[0].supersedes_attempt_id
        == "verification-attempt-2"
    )
    assert (
        summary.workflow_timeline[-1].verification_observations[0].command
        == "uv run pytest -q"
    )
