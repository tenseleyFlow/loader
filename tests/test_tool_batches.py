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
from loader.runtime.tool_batches import (
    ToolBatchRunner,
)
from loader.runtime.tool_batches import (
    _should_prioritize_missing_artifact as tool_batches_should_prioritize_missing_artifact,
)
from loader.runtime.workflow import sync_todos_to_definition_of_done
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
    state: ToolExecutionState = ToolExecutionState.EXECUTED,
    metadata: dict[str, object] | None = None,
) -> ToolExecutionOutcome:
    return ToolExecutionOutcome(
        tool_call=tool_call,
        state=state,
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
    (temp_dir / "chapters").mkdir()
    (temp_dir / "index.html").write_text("<ul></ul>\n")
    (temp_dir / "chapters" / "01-introduction.html").write_text("<h1>Intro</h1>\n")
    (temp_dir / "chapters" / "02-setup.html").write_text("<h1>Setup</h1>\n")
    (temp_dir / "chapters" / "03-basics.html").write_text("<h1>Basics</h1>\n")
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'index.html'}`",
                f"- `{temp_dir / 'chapters' / '01-introduction.html'}`",
                f"- `{temp_dir / 'chapters' / '02-setup.html'}`",
                f"- `{temp_dir / 'chapters' / '03-basics.html'}`",
                f"- `{temp_dir / 'chapters' / '04-variables.html'}`",
            ]
        )
    )
    context.session.current_task = (
        f"Update {temp_dir / 'index.html'} with the right chapter links."
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
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
    dod = create_definition_of_done("Fix the chapter links")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items.append("Create the remaining chapter files")
    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert len(persistent_messages) == 1
    assert "Reuse the earlier observation instead of repeating it." in persistent_messages[0]
    assert "A declared output artifact is still missing." in persistent_messages[0]
    assert "Resume by creating `04-variables.html` now." in persistent_messages[0]
    assert (
        f"Prefer one `write` call for `{temp_dir / 'chapters' / '04-variables.html'}` instead of more rereads."
        in persistent_messages[0]
    )
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_todo_write_does_not_regress_completed_file_todo(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run for this scenario")

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
        auto_recover=False,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create 03-first-website.html",
                "active_form": "Creating 03-first-website.html",
                "status": "pending",
            },
            {
                "content": "Create 04-configuration-basics.html",
                "active_form": "Creating 04-configuration-basics.html",
                "status": "pending",
            },
        ],
    )

    chapter_path = temp_dir / "guides" / "nginx" / "chapters" / "03-first-website.html"
    chapter_path.parent.mkdir(parents=True)
    write_call = ToolCall(
        id="write-ch3",
        name="write",
        arguments={"file_path": str(chapter_path), "content": "<html></html>\n"},
    )
    stale_todo_call = ToolCall(
        id="todo-stale",
        name="TodoWrite",
        arguments={
            "todos": [
                {
                    "content": "Create 03-first-website.html",
                    "active_form": "Creating 03-first-website.html",
                    "status": "pending",
                },
                {
                    "content": "Create 04-configuration-basics.html",
                    "active_form": "Creating 04-configuration-basics.html",
                    "status": "pending",
                },
            ]
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=write_call,
                output=f"Successfully wrote {chapter_path}",
                is_error=False,
            ),
            tool_outcome(
                tool_call=stale_todo_call,
                output="Todos updated",
                is_error=False,
                metadata={
                    "new_todos": [
                        {
                            "content": "Create 03-first-website.html",
                            "active_form": "Creating 03-first-website.html",
                            "status": "pending",
                        },
                        {
                            "content": "Create 04-configuration-basics.html",
                            "active_form": "Creating 04-configuration-basics.html",
                            "status": "pending",
                        },
                    ]
                },
            ),
        ]
    )

    summary = TurnSummary(final_response="")
    await runner.execute_batch(
        tool_calls=[write_call, stale_todo_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert "Create 03-first-website.html" in dod.completed_items
    assert "Create 03-first-website.html" not in dod.pending_items
    assert "Create 04-configuration-basics.html" in dod.pending_items


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
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
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

    assert persistent_messages == []
    assert ephemeral_messages == []
    assert len(summary.tool_result_messages) == 1
    assert "Verified chapter inventory:" not in summary.tool_result_messages[0].content


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
    context.session.current_task = (
        "Update index.html so every chapter link and title matches the real HTML files in chapters/."
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
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
        dod=create_definition_of_done(
            "Update index.html so every chapter link and title matches the real HTML files in chapters/."
        ),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert all(
        "Semantic verification preview:" not in message.content
        for message in summary.tool_result_messages
    )
    assert persistent_messages == []
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_does_not_apply_html_toc_handoff_to_reference_read(
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
    index_path.write_text(
        "<h2>Table of Contents</h2>\n"
        '<ul class="chapter-list">\n'
        '    <li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>\n'
        '    <li><a href="chapters/02-setup.html">Chapter 2: Setting Up Your Environment</a></li>\n'
        "</ul>\n"
    )

    prompt = (
        "Have a look at ~/Loader/guides/fortran and chapters/ within. Get a feel "
        "for the structure and cadence of the guide. We are going to make an all "
        "new equally thorough guide on how to use the nginx tool."
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.session.current_task = prompt  # type: ignore[attr-defined]
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="read-index",
        name="read",
        arguments={"file_path": str(index_path)},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=index_path.read_text(),
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
        dod=create_definition_of_done(prompt),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages == []
    assert ephemeral_messages == []
    assert all(
        "Semantic verification preview:" not in message.content
        for message in summary.tool_result_messages
    )


@pytest.mark.asyncio
async def test_tool_batch_runner_queues_next_pending_todo_after_discovery_progress(
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

    reference = temp_dir / "fortran" / "chapters" / "01-introduction.html"
    reference.parent.mkdir(parents=True)
    reference.write_text("<h1>Introduction</h1>\n<p>Guide cadence.</p>\n")
    nginx_root = temp_dir / "Loader" / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{nginx_root / 'index.html'}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create an equally thorough nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Examine the existing Fortran guide structure to understand the cadence and format",
                "active_form": "Working on: Examine the existing Fortran guide structure to understand the cadence and format",
                "status": "pending",
            },
            {
                "content": "Create the nginx directory structure",
                "active_form": "Working on: Create the nginx directory structure",
                "status": "pending",
            },
            {
                "content": "Create the nginx index.html file",
                "active_form": "Working on: Create the nginx index.html file",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="read-reference",
        name="read",
        arguments={"file_path": str(reference)},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="<h1>Introduction</h1>\n<p>Guide cadence.</p>\n",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert (
        "Examine the existing Fortran guide structure to understand the cadence and format"
        in dod.completed_items
    )
    assert any(
        "Continue with the next pending item: `Create the nginx directory structure`"
        in message
        for message in persistent_messages
    )
    assert any(
        "Resume by creating `chapters/` now." in message
        for message in persistent_messages
    )
    assert all("01-introduction.html" not in message for message in persistent_messages)
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_queues_setup_directory_before_file_when_plan_lists_index_first(
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

    reference = temp_dir / "fortran" / "chapters" / "01-introduction.html"
    reference.parent.mkdir(parents=True)
    reference.write_text("<h1>Introduction</h1>\n<p>Guide cadence.</p>\n")
    nginx_root = temp_dir / "Loader" / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{nginx_root / 'index.html'}`",
                f"- `{chapters}/`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create an equally thorough nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Examine the existing Fortran guide structure to understand the cadence and format",
                "active_form": "Working on: Examine the existing Fortran guide structure to understand the cadence and format",
                "status": "pending",
            },
            {
                "content": "Create the nginx directory structure",
                "active_form": "Working on: Create the nginx directory structure",
                "status": "pending",
            },
            {
                "content": "Create the nginx index.html file",
                "active_form": "Working on: Create the nginx index.html file",
                "status": "pending",
            },
        ],
        project_root=temp_dir,
    )
    tool_call = ToolCall(
        id="read-reference-index-first",
        name="read",
        arguments={"file_path": str(reference)},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="<h1>Introduction</h1>\n<p>Guide cadence.</p>\n",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages
    assert any(
        "Continue with the next pending item: `Create the nginx directory structure`"
        in message
        for message in persistent_messages
    )
    assert any(
        "Resume by creating `chapters/` now." in message
        for message in persistent_messages
    )
    assert all(
        "Next step: create `index.html`." not in message
        for message in persistent_messages
    )
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_duplicate_reference_read_prefers_next_pending_todo(
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

    reference = temp_dir / "fortran" / "index.html"
    reference.parent.mkdir(parents=True)
    reference.write_text("<h1>Fortran Beginner's Guide</h1>\n")

    messages = [
        Message(
            role=Role.TOOL,
            content=(
                "Observation [read]: Result: "
                "<h1>Fortran Beginner's Guide</h1>\n"
            ),
        )
    ]
    context = build_context(
        temp_dir=temp_dir,
        messages=messages,
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    prompt = (
        "Have a look at ~/Loader/guides/fortran and chapters/ within. Get a feel "
        "for the structure and cadence of the guide. We are going to make an all "
        "new equally thorough guide on how to use the nginx tool."
    )
    context.session.current_task = prompt
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done(prompt)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Examine the existing Fortran guide structure to understand the cadence and format",
                "active_form": "Working on: Examine the existing Fortran guide structure to understand the cadence and format",
                "status": "completed",
            },
            {
                "content": "Create the nginx directory structure",
                "active_form": "Working on: Create the nginx directory structure",
                "status": "pending",
            },
            {
                "content": "Create the nginx index.html file",
                "active_form": "Working on: Create the nginx index.html file",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="read-dup",
        name="read",
        arguments={"file_path": str(reference)},
    )
    duplicate_message = (
        "[Skipped - duplicate action: Already read "
        f"{reference} recently without any intervening changes; "
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert len(persistent_messages) == 1
    assert "Reuse the earlier observation instead of repeating it." in persistent_messages[0]
    assert (
        "Continue with the next pending item: `Create the nginx directory structure`"
        in persistent_messages[0]
    )
    assert "Update `" not in persistent_messages[0]
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_successful_reference_read_prioritizes_concrete_missing_artifact(
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

    guide_root = temp_dir / "Loader" / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    chapter_one = chapters / "01-introduction.html"
    chapter_one.write_text("<html></html>\n")
    index_path = guide_root / "index.html"

    reference = temp_dir / "Loader" / "guides" / "fortran" / "chapters" / "01-introduction.html"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("<h1>Introduction</h1>\n<p>Guide cadence.</p>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapters / '02-installation.html'}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(chapter_one))
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Examine the existing Fortran guide structure to understand the format and cadence",
                "active_form": "Working on: Examine the existing Fortran guide structure to understand the format and cadence",
                "status": "pending",
            },
            {
                "content": "Create each chapter file with appropriate content",
                "active_form": "Working on: Create each chapter file with appropriate content",
                "status": "pending",
            },
            {
                "content": "Ensure all files follow the same structure and style as the Fortran guide",
                "active_form": "Working on: Ensure all files follow the same structure and style as the Fortran guide",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="read-reference-chapter",
        name="read",
        arguments={"file_path": str(reference)},
    )
    read_output = "Observation [read]: Result: <h1>Introduction</h1>\n<p>Guide cadence.</p>\n"
    executor = FakeExecutor(
        [
            ToolExecutionOutcome(
                tool_call=tool_call,
                state=ToolExecutionState.EXECUTED,
                message=Message.tool_result_message(
                    tool_call_id=tool_call.id,
                    display_content=read_output,
                    result_content=read_output,
                ),
                event_content=read_output,
                is_error=False,
                result_output=read_output,
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages
    assert any(
        "Confirmed progress: `Examine the existing Fortran guide structure to understand the format and cadence`"
        in message
        for message in persistent_messages
    )
    assert any("Resume by creating `index.html` now." in message for message in persistent_messages)
    assert not any(
        "Continue with the next pending item: `Create each chapter file with appropriate content`"
        in message
        for message in persistent_messages
    )
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_duplicate_read_ignores_unplanned_expansion_after_plan_complete(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run for this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "Create 07-performance-tuning.html",
        "Verify all guide files are linked and complete",
        "Complete the requested work",
    ]

    tool_call = ToolCall(
        id="read-dup",
        name="read",
        arguments={"file_path": str(chapter_one)},
    )
    duplicate_message = (
        "[Skipped - duplicate action: Already read "
        f"{chapter_one} recently without any intervening changes; "
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert len(persistent_messages) == 1
    assert "Verify all guide files are linked and complete" in persistent_messages[0]
    assert "Create 07-performance-tuning.html" not in persistent_messages[0]
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_duplicate_read_after_plan_complete_pushes_verification_handoff(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run for this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.verification_commands = [f"ls -la {guide_root}"]
    dod.pending_items = [
        "Create 07-performance-tuning.html",
        "Complete the requested work",
    ]

    tool_call = ToolCall(
        id="read-dup",
        name="read",
        arguments={"file_path": str(chapter_one)},
    )
    duplicate_message = (
        "[Skipped - duplicate action: Already read "
        f"{chapter_one} recently without any intervening changes; "
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert len(persistent_messages) == 1
    assert "All explicitly planned artifacts already exist." in persistent_messages[0]
    assert (
        "Move to verification or final confirmation using the files already on disk."
        in persistent_messages[0]
    )
    assert "Create 07-performance-tuning.html" not in persistent_messages[0]
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_duplicate_read_after_plan_complete_ignores_stale_creation_todos(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run for this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run for this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.verification_commands = [f"ls -la {guide_root}"]
    dod.pending_items = [
        "Create 01-getting-started.html",
        "Creating 02-installation.html",
        "Complete the requested work",
    ]

    tool_call = ToolCall(
        id="read-dup-built-stale",
        name="read",
        arguments={"file_path": str(chapter_one)},
    )
    duplicate_message = (
        "[Skipped - duplicate action: Already read "
        f"{chapter_one} recently without any intervening changes; "
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert len(persistent_messages) == 1
    assert "All explicitly planned artifacts already exist." in persistent_messages[0]
    assert (
        "Move to verification or final confirmation using the files already on disk."
        in persistent_messages[0]
    )
    assert "Create 01-getting-started.html" not in persistent_messages[0]
    assert "Creating 02-installation.html" not in persistent_messages[0]
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_observation_handoff_pushes_mutation_step(
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

    reference = temp_dir / "fortran" / "chapters" / "01-introduction.html"
    reference.parent.mkdir(parents=True)
    reference.write_text("<h1>Introduction</h1>\n<p>Guide cadence.</p>\n")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Examine the existing Fortran guide structure to understand the cadence and format",
                "active_form": "Working on: Examine the existing Fortran guide structure to understand the cadence and format",
                "status": "pending",
            },
            {
                "content": "Create the nginx index.html file",
                "active_form": "Working on: Create the nginx index.html file",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="read-reference",
        name="read",
        arguments={"file_path": str(reference)},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="<h1>Introduction</h1>\n<p>Guide cadence.</p>\n",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert any(
        "Continue with the next pending item: `Create the nginx index.html file`"
        in message
        for message in persistent_messages
    )
    assert any(
        "stop gathering more reference material and perform the change now" in message
        for message in persistent_messages
    )
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_discovery_completion_handoff_stays_persistent(
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

    reference = temp_dir / "fortran" / "chapters" / "01-introduction.html"
    reference.parent.mkdir(parents=True)
    reference.write_text("<h1>Introduction</h1>\n<p>Guide cadence.</p>\n")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the existing fortran guide structure and content",
                "active_form": "Working on: First, examine the existing fortran guide structure and content",
                "status": "pending",
            },
            {
                "content": "Create the nginx directory structure",
                "active_form": "Working on: Create the nginx directory structure",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="read-reference",
        name="read",
        arguments={"file_path": str(reference)},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="<h1>Introduction</h1>\n<p>Guide cadence.</p>\n",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages
    assert any(
        "Continue with the next pending item: `Create the nginx directory structure`"
        in message
        for message in persistent_messages
    )
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_missing_artifact_nudge_names_next_file_after_setup_mkdir(
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

    nginx_root = temp_dir / "Loader" / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{nginx_root / 'index.html'}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create the nginx directory structure",
                "active_form": "Creating the nginx directory structure",
                "status": "pending",
            },
            {
                "content": "Develop the main index.html file with proper structure",
                "active_form": "Developing the main index.html file with proper structure",
                "status": "pending",
            },
        ],
    )

    tool_call = ToolCall(
        id="mkdir-nginx",
        name="bash",
        arguments={"command": f"mkdir -p {chapters}"},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages
    message = persistent_messages[-1]
    assert "Directory setup is complete." in message
    assert "Continue with the next pending item: `Develop the main index.html file with proper structure`." in message
    assert "Resume by creating `index.html` now." in message
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_first_chapter_handoff_becomes_ephemeral_after_first_file(
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

    nginx_root = temp_dir / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = nginx_root / "index.html"

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapters / '01-introduction.html'}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create the main index.html file with proper structure",
                "active_form": "Creating the main index.html file with proper structure",
                "status": "pending",
            },
            {
                "content": "Create each chapter file with appropriate content",
                "active_form": "Creating each chapter file with appropriate content",
                "status": "pending",
            },
        ],
    )

    tool_call = ToolCall(
        id="write-index",
        name="write",
        arguments={
            "file_path": str(index_path),
            "content": "<html></html>\n",
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=f"Successfully wrote 14 bytes to {index_path}",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages == []
    assert ephemeral_messages
    message = ephemeral_messages[-1]
    assert "Confirmed progress:" in message
    assert "Next step: create `01-introduction.html`." in message
    assert (
        f"Prefer one `write(file_path=..., content=...)` call for `{(chapters / '01-introduction.html').resolve(strict=False)}` now."
        in message
    )
    assert "Do not reread reference material or spend the next turn on bookkeeping." in message


@pytest.mark.asyncio
async def test_tool_batch_runner_redirects_post_write_self_audit_to_next_missing_artifact(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    nginx_root = temp_dir / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = nginx_root / "index.html"
    index_path.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/01-introduction.html">Chapter 1: Introduction to Nginx</a>',
                '<a href="chapters/02-installation.html">Chapter 2: Installation and Setup</a>',
                "</html>",
            ]
        )
        + "\n"
    )

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{nginx_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapters / '01-introduction.html'}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(index_path))
    dod.completed_items.append("Develop the main index.html file for the nginx guide")
    dod.pending_items.append("Create chapter files for the nginx guide")

    tool_call = ToolCall(
        id="read-index-self-audit",
        name="read",
        arguments={"file_path": str(index_path)},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="1\t<html>\n",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages
    message = persistent_messages[-1]
    assert "You already have the current contents of `index.html` from the successful write." in message
    assert "Resume by creating `01-introduction.html` now." in message
    assert "Do not spend another turn rereading the file you just wrote or on TodoWrite alone." in message
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_softens_first_file_handoff_after_recovery_prompt(
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

    nginx_root = temp_dir / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = nginx_root / "index.html"

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapters / '01-introduction.html'}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(
                role=Role.USER,
                content=(
                    "[EMPTY ASSISTANT RESPONSE]\n"
                    "Respond with that concrete mutation tool call now. Do not return an empty response."
                ),
            )
        ],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create the main index.html file with proper structure",
                "active_form": "Creating the main index.html file with proper structure",
                "status": "pending",
            },
            {
                "content": "Create each chapter file with appropriate content",
                "active_form": "Creating each chapter file with appropriate content",
                "status": "pending",
            },
        ],
    )

    tool_call = ToolCall(
        id="write-index-recovered",
        name="write",
        arguments={
            "file_path": str(index_path),
            "content": "<html></html>\n",
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=f"Successfully wrote 14 bytes to {index_path}",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages == []
    assert ephemeral_messages
    message = ephemeral_messages[-1]
    assert "Next step: create `01-introduction.html`." in message


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_uses_concrete_output_language_for_aggregate_chapter_step(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    index_path.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/01-introduction.html">Chapter 1: Introduction to Nginx</a>',
                '<a href="chapters/02-installation.html">Chapter 2: Installation and Setup</a>',
                "</html>",
            ]
        )
        + "\n"
    )

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    queued_messages: list[str] = []
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(index_path))
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Develop the main index.html file with proper structure",
                "active_form": "Developing the main index.html file with proper structure",
                "status": "completed",
            },
            {
                "content": "Create chapter files with content and structure",
                "active_form": "Creating chapter files with content and structure",
                "status": "pending",
            },
        ],
    )

    todos = [
        {
            "content": "Develop the main index.html file with proper structure",
            "active_form": "Developing the main index.html file with proper structure",
            "status": "completed",
        },
        {
            "content": "Create chapter files with content and structure",
            "active_form": "Creating chapter files with content and structure",
            "status": "pending",
        },
    ]
    tool_call = ToolCall(
        id="todo-aggregate",
        name="TodoWrite",
        arguments={"todos": todos},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={"new_todos": todos},
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Continue with the next concrete output: `01-introduction.html`." in message
    assert "Resume by creating `01-introduction.html` now." in message
    assert (
        "Continue with the next pending item: `Create chapter files with content and structure`."
        not in message
    )


@pytest.mark.asyncio
async def test_duplicate_observation_nudge_prioritizes_missing_artifact_over_review(
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

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_one.write_text("<h1>One</h1>\n")
    index_path.write_text("<a href=\"chapters/01-getting-started.html\">One</a>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapters / '06-ssl-configuration.html'}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Ensure all files are properly linked and formatted consistently",
                "active_form": "Working on: Ensure all files are properly linked and formatted consistently",
                "status": "pending",
            },
            {
                "content": "Create the final chapter (06-ssl-configuration.html)",
                "active_form": "Working on: Create the final chapter (06-ssl-configuration.html)",
                "status": "pending",
            },
        ],
    )
    assert tool_batches_should_prioritize_missing_artifact(
        dod=dod,
        next_pending=dod.pending_items[0],
        missing_artifact=(chapters / "06-ssl-configuration.html", False),
        project_root=temp_dir,
    )

    tool_call = ToolCall(
        id="dup-read",
        name="read",
        arguments={"file_path": str(index_path)},
    )
    runner._queue_duplicate_observation_nudge(tool_call, dod=dod)  # type: ignore[attr-defined]

    assert persistent_messages
    message = persistent_messages[-1]
    assert "06-ssl-configuration.html" in message
    assert "Do not switch into review or consistency-check mode" in message
    assert (
        "Continue with the next pending item: `Ensure all files are properly linked and formatted consistently`"
        not in message
    )


@pytest.mark.asyncio
async def test_tool_batch_runner_hands_off_to_verification_once_planned_artifacts_exist(
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

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<a href=\"chapters/01-getting-started.html\">One</a>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create the guide files",
                "active_form": "Working on: Create the guide files",
                "status": "completed",
            },
            {
                "content": "Ensure all files are properly linked and formatted consistently",
                "active_form": "Working on: Ensure all files are properly linked and formatted consistently",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="write-final",
        name="write",
        arguments={
            "file_path": str(chapter_two),
            "content": "<h1>Two</h1>\n",
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=f"Successfully wrote {chapter_two}",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert any(
        "All explicitly planned artifacts now exist." in message
        for message in persistent_messages
    )
    assert any(
        "Ensure all files are properly linked and formatted consistently" in message
        for message in persistent_messages
    )
    assert any(
        "Move to verification once no specific mismatch remains." in message
        for message in persistent_messages
    )


@pytest.mark.asyncio
async def test_tool_batch_runner_mutation_handoff_points_at_next_missing_artifact(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    index_path.write_text("<html></html>\n")
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create the main index.html file with proper structure",
                "active_form": "Working on: Create the main index.html file with proper structure",
                "status": "pending",
            },
            {
                "content": "Create each chapter file in sequence, following the established pattern",
                "active_form": "Working on: Create each chapter file in sequence, following the established pattern",
                "status": "pending",
            },
            {
                "content": "Ensure all files are properly linked and formatted consistently",
                "active_form": "Working on: Ensure all files are properly linked and formatted consistently",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="write-index",
        name="write",
        arguments={"file_path": str(index_path), "content": "<html></html>\n"},
    )
    executor = FakeExecutor(
        [tool_outcome(tool_call=tool_call, output=f"Successfully wrote {index_path}", is_error=False)]
    )

    summary = TurnSummary(final_response="")
    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=summary,
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages == []
    assert ephemeral_messages
    message = ephemeral_messages[-1]
    assert "Next step: create `01-getting-started.html`." in message
    assert "refresh `TodoWrite`" not in message
    assert "Do not reread reference material or spend the next turn on bookkeeping." in message


@pytest.mark.asyncio
async def test_tool_batch_runner_large_plan_does_not_claim_completion_early(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    index_path.write_text("<html></html>\n")

    chapter_paths = [
        chapters / "01-getting-started.html",
        chapters / "02-installation.html",
        chapters / "03-first-website.html",
        chapters / "04-configuration-basics.html",
        chapters / "05-advanced-configurations.html",
        chapters / "06-performance-tuning.html",
        chapters / "07-security-best-practices.html",
    ]
    for chapter in chapter_paths[:4]:
        chapter.write_text(f"<h1>{chapter.stem}</h1>\n")
    chapter_paths[4].write_text("<h1>Advanced configurations</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                *[f"- `{path}`" for path in chapter_paths],
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a thorough nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create the nginx guide artifacts",
                "active_form": "Creating nginx guide artifacts",
                "status": "pending",
            },
            {
                "content": "Verify all guide files are linked and complete",
                "active_form": "Verifying guide linkage and completeness",
                "status": "pending",
            },
        ],
    )
    tool_call = ToolCall(
        id="write-chapter-05",
        name="write",
        arguments={
            "file_path": str(chapter_paths[4]),
            "content": "<h1>Advanced configurations</h1>\n",
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=f"Successfully wrote {chapter_paths[4]}",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert any(
        "Next step: create `06-performance-tuning.html`." in message
        for message in ephemeral_messages
    )
    assert not any(
        "All explicitly planned artifacts now exist." in message
        for message in ephemeral_messages
    )


@pytest.mark.asyncio
async def test_tool_batch_runner_uses_compact_missing_artifact_nudge_after_substantial_progress(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_paths = [
        chapters / "01-introduction.html",
        chapters / "02-installation.html",
        chapters / "03-configuration.html",
        chapters / "04-basic-usage.html",
        chapters / "05-advanced-features.html",
    ]
    for path in (index_path, *chapter_paths[:4]):
        path.write_text("<html></html>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                *[f"- `{path}`" for path in chapter_paths],
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a thorough nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend(str(path) for path in (index_path, *chapter_paths[:4]))
    dod.completed_items.extend(
        [
            "Create the nginx directory structure",
            "Create the main index.html file with proper structure",
        ]
    )
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create each chapter file with appropriate content",
                "active_form": "Creating each chapter file with appropriate content",
                "status": "pending",
            }
        ],
    )
    tool_call = ToolCall(
        id="write-chapter-04",
        name="write",
        arguments={
            "file_path": str(chapter_paths[3]),
            "content": "<html>updated</html>\n",
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=f"Successfully wrote {chapter_paths[3]}",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert ephemeral_messages
    message = ephemeral_messages[-1]
    assert "Next step: create `05-advanced-features.html`." in message
    assert "Do not reread reference material or spend the next turn on bookkeeping." in message
    assert "refresh `TodoWrite`" not in message


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_with_missing_artifact_requeues_exact_resume_step(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    index_path.write_text("<html></html>\n")
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    chapter_one.write_text("<h1>One</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    persistent_messages: list[str] = []
    ephemeral_messages: list[str] = []
    context.queue_steering_message_callback = persistent_messages.append
    context.queue_ephemeral_steering_message_callback = ephemeral_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create 01-getting-started.html",
                "active_form": "Creating 01-getting-started.html",
                "status": "completed",
            },
            {
                "content": "Create 02-installation.html",
                "active_form": "Creating 02-installation.html",
                "status": "pending",
            },
        ],
    )
    dod.touched_files.extend([str(index_path), str(chapter_one)])

    tool_call = ToolCall(
        id="todo-only",
        name="TodoWrite",
        arguments={
            "todos": [
                {
                    "content": "Create 01-getting-started.html",
                    "active_form": "Creating 01-getting-started.html",
                    "status": "completed",
                },
                {
                    "content": "Create 02-installation.html",
                    "active_form": "Creating 02-installation.html",
                    "status": "pending",
                },
            ]
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={
                    "new_todos": [
                        {
                            "content": "Create 01-getting-started.html",
                            "active_form": "Creating 01-getting-started.html",
                            "status": "completed",
                        },
                        {
                            "content": "Create 02-installation.html",
                            "active_form": "Creating 02-installation.html",
                            "status": "pending",
                        },
                    ]
                },
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert persistent_messages
    message = persistent_messages[-1]
    assert "Todo tracking is updated. A declared output artifact is still missing." in message
    assert "Resume by creating `02-installation.html` now." in message
    assert "refresh `TodoWrite`" in message
    assert "Do not spend the next turn on TodoWrite alone" in message
    assert ephemeral_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_after_artifacts_exist_pushes_verification_handoff(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")
    chapter_two.write_text("<h1>Two</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

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
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.verification_commands = [f"ls -la {guide_root}"]
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "First, examine the existing Fortran guide structure to understand the format and content organization",
                "active_form": "Working on: First, examine the existing Fortran guide structure to understand the format and content organization",
                "status": "pending",
            },
            {
                "content": "Verify all guide files are linked and complete",
                "active_form": "Working on: Verify all guide files are linked and complete",
                "status": "pending",
            },
        ],
        project_root=temp_dir,
    )

    tool_call = ToolCall(
        id="todo-only",
        name="TodoWrite",
        arguments={
            "todos": [
                {
                    "content": "First, examine the existing Fortran guide structure to understand the format and content organization",
                    "active_form": "Working on: First, examine the existing Fortran guide structure to understand the format and content organization",
                    "status": "pending",
                },
                {
                    "content": "Verify all guide files are linked and complete",
                    "active_form": "Working on: Verify all guide files are linked and complete",
                    "status": "pending",
                },
            ]
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={
                    "new_todos": [
                        {
                            "content": "First, examine the existing Fortran guide structure to understand the format and content organization",
                            "active_form": "Working on: First, examine the existing Fortran guide structure to understand the format and content organization",
                            "status": "pending",
                        },
                        {
                            "content": "Verify all guide files are linked and complete",
                            "active_form": "Working on: Verify all guide files are linked and complete",
                            "status": "pending",
                        },
                    ]
                },
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Todo tracking is updated. All explicitly planned artifacts now exist." in message
    assert "Verify all guide files are linked and complete" in message
    assert "Move to verification once no specific mismatch remains." in message
    assert "reopen reference materials" in message
    assert "Fortran guide structure" not in message


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_with_existing_output_roots_requeues_next_mutation(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    index_path.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html>",
                "<body>",
                '<a href="chapters/01-introduction.html">Introduction</a>',
                "</body>",
                "</html>",
                "",
            ]
        )
    )

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                "",
            ]
        )
    )

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
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.append(str(index_path))
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Examine the existing Fortran guide structure",
                "active_form": "Examining the existing Fortran guide structure",
                "status": "completed",
            },
            {
                "content": "Create the nginx directory structure",
                "active_form": "Creating the nginx directory structure",
                "status": "completed",
            },
            {
                "content": "Write the introduction chapter",
                "active_form": "Writing the introduction chapter",
                "status": "pending",
            },
        ],
        project_root=temp_dir,
    )

    tool_call = ToolCall(
        id="todo-next-mutation",
        name="TodoWrite",
        arguments={
            "todos": [
                {
                    "content": "Examine the existing Fortran guide structure",
                    "active_form": "Examining the existing Fortran guide structure",
                    "status": "completed",
                },
                {
                    "content": "Create the nginx directory structure",
                    "active_form": "Creating the nginx directory structure",
                    "status": "completed",
                },
                {
                    "content": "Write the introduction chapter",
                    "active_form": "Writing the introduction chapter",
                    "status": "pending",
                },
            ]
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={
                    "new_todos": [
                        {
                            "content": "Examine the existing Fortran guide structure",
                            "active_form": "Examining the existing Fortran guide structure",
                            "status": "completed",
                        },
                        {
                            "content": "Create the nginx directory structure",
                            "active_form": "Creating the nginx directory structure",
                            "status": "completed",
                        },
                        {
                            "content": "Write the introduction chapter",
                            "active_form": "Writing the introduction chapter",
                            "status": "pending",
                        },
                    ]
                },
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Todo tracking is updated. A declared output artifact is still missing." in message
    assert "Continue with the next pending item: `Write the introduction chapter`." in message
    assert "Resume by creating `01-introduction.html` now." in message
    assert "Prefer one `write` call for `" in message
    assert "01-introduction.html` instead of more rereads." in message
    assert "Do not spend the next turn on TodoWrite alone" in message


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_prefers_pending_index_over_empty_output_directory(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "Loader" / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Examine the existing Fortran guide structure to understand the format and depth",
                "active_form": "Examining the existing Fortran guide structure",
                "status": "completed",
            },
            {
                "content": "Create the new nginx guide directory structure",
                "active_form": "Creating the new nginx guide directory structure",
                "status": "completed",
            },
            {
                "content": "Create a new index.html for the nginx guide",
                "active_form": "Creating a new index.html for the nginx guide",
                "status": "pending",
            },
            {
                "content": "Create the first chapter for the nginx guide",
                "active_form": "Creating the first chapter for the nginx guide",
                "status": "pending",
            },
        ],
        project_root=temp_dir,
    )

    queued_messages: list[str] = []
    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    todos = [
        {
            "content": "Examine the existing Fortran guide structure to understand the format and depth",
            "active_form": "Examining the existing Fortran guide structure",
            "status": "completed",
        },
        {
            "content": "Create the new nginx guide directory structure",
            "active_form": "Creating the new nginx guide directory structure",
            "status": "completed",
        },
        {
            "content": "Create a new index.html for the nginx guide",
            "active_form": "Creating a new index.html for the nginx guide",
            "status": "pending",
        },
        {
            "content": "Create the first chapter for the nginx guide",
            "active_form": "Creating the first chapter for the nginx guide",
            "status": "pending",
        },
    ]
    tool_call = ToolCall(
        id="todo-index-before-chapter",
        name="TodoWrite",
        arguments={"todos": todos},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={"new_todos": todos},
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Continue with the next pending item: `Create a new index.html for the nginx guide`." in message
    assert "Resume by creating `index.html` now." in message
    assert f"Prefer one `write` call for `{index_path.resolve(strict=False)}`" in message
    assert "01-introduction.html" not in message


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_with_declared_child_targets_names_next_missing_file(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    index_path.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/introduction.html">Introduction</a>',
                '<a href="chapters/installation.html">Installation</a>',
                "</html>",
            ]
        )
        + "\n"
    )

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "Write the introduction chapter",
        "Complete the requested work",
    ]
    dod.touched_files.append(str(index_path))

    queued_messages: list[str] = []
    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    tool_call = ToolCall(
        id="todo-1",
        name="TodoWrite",
        arguments={
            "todos": [
                {
                    "content": "Write the introduction chapter",
                    "activeForm": "Writing the introduction chapter",
                    "status": "pending",
                }
            ]
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={
                    "new_todos": [
                        {
                            "content": "Write the introduction chapter",
                            "active_form": "Writing the introduction chapter",
                            "status": "pending",
                        }
                    ]
                },
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Todo tracking is updated. A declared output artifact is still missing." in message
    assert "Continue with the next pending item: `Write the introduction chapter`." in message
    assert "Resume by creating `introduction.html` now." in message
    assert "Prefer one `write` call for `" in message
    assert "introduction.html` instead of more rereads." in message
    assert "Do not spend the next turn on TodoWrite alone" in message


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_names_concrete_pending_file_after_artifacts_exist(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-introduction.html"
    index_path.write_text(
        "\n".join(
            [
                "<html>",
                '<a href="chapters/01-introduction.html">Chapter 1: Introduction to NGINX Tool</a>',
                '<a href="chapters/02-installation.html">Chapter 2: Installation and Setup</a>',
                "</html>",
            ]
        )
        + "\n"
    )
    chapter_one.write_text("<html></html>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "Creating Chapter 2: Installation and Setup",
        "Complete the requested work",
    ]
    dod.touched_files.extend([str(index_path), str(chapter_one)])

    queued_messages: list[str] = []
    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    tool_call = ToolCall(
        id="todo-1",
        name="TodoWrite",
        arguments={
            "todos": [
                {
                    "content": "Creating Chapter 2: Installation and Setup",
                    "activeForm": "Creating Chapter 2: Installation and Setup",
                    "status": "pending",
                }
            ]
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={
                    "new_todos": [
                        {
                            "content": "Creating Chapter 2: Installation and Setup",
                            "active_form": "Creating Chapter 2: Installation and Setup",
                            "status": "pending",
                        }
                    ]
                },
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Todo tracking is updated. A declared output artifact is still missing." in message
    assert "Continue with the next pending item: `Creating Chapter 2: Installation and Setup`." in message
    assert "Resume by creating `02-installation.html` now." in message
    assert (
        f"Prefer one `write` call for `{(chapters / '02-installation.html').resolve(strict=False)}` "
        "instead of more rereads."
        in message
    )
    assert "Make your next response the concrete mutation tool call itself" in message


@pytest.mark.asyncio
async def test_tool_batch_runner_todowrite_uses_observed_sibling_pattern_for_next_file(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    reference_chapters = temp_dir / "fortran" / "chapters"
    reference_chapters.mkdir(parents=True)
    (reference_chapters / "01-introduction.html").write_text("<h1>Introduction</h1>\n")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    index_path.write_text("<html></html>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                "",
            ]
        )
    )

    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items = [
        "Write the introduction chapter",
        "Complete the requested work",
    ]
    dod.touched_files.append(str(index_path))

    queued_messages: list[str] = []
    context = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id="read-ref-1",
                        name="read",
                        arguments={"file_path": str(reference_chapters / "01-introduction.html")},
                    )
                ],
            )
        ],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    tool_call = ToolCall(
        id="todo-observed-1",
        name="TodoWrite",
        arguments={
            "todos": [
                {
                    "content": "Write the introduction chapter",
                    "activeForm": "Writing the introduction chapter",
                    "status": "pending",
                }
            ]
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Todos updated",
                is_error=False,
                metadata={
                    "new_todos": [
                        {
                            "content": "Write the introduction chapter",
                            "active_form": "Writing the introduction chapter",
                            "status": "pending",
                        }
                    ]
                },
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Todo tracking is updated. A declared output artifact is still missing." in message
    assert "Continue with the next pending item: `Write the introduction chapter`." in message
    assert "Resume by creating `01-introduction.html` now." in message
    assert (
        "It mirrors the observed filename pattern from another `chapters/` directory "
        "you already inspected."
        in message
    )
    assert "01-introduction.html` instead of more rereads." in message


@pytest.mark.asyncio
async def test_tool_batch_runner_bookkeeping_note_with_missing_artifact_requeues_resume_step(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>One</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}/`",
                f"- `{chapters}/`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

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
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Create 01-getting-started.html",
                "active_form": "Creating 01-getting-started.html",
                "status": "completed",
            },
            {
                "content": "Create 02-installation.html",
                "active_form": "Creating 02-installation.html",
                "status": "pending",
            },
        ],
        project_root=temp_dir,
    )
    dod.touched_files.extend([str(index_path), str(chapter_one)])

    tool_call = ToolCall(
        id="working-note",
        name="notepad_write_working",
        arguments={"content": "Creating the second chapter file: Installation"},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Working note recorded",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Bookkeeping note is recorded. A declared output artifact is still missing." in message
    assert "Resume by creating `02-installation.html` now." in message
    assert "Make your next response the concrete mutation tool call itself" in message
    assert "refresh `TodoWrite`" in message
    assert "Do not spend the next turn on additional notes, rediscovery, verification, or final confirmation" in message


@pytest.mark.asyncio
async def test_tool_batch_runner_working_note_respects_discovery_first_pending_step(
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
        raise AssertionError("Verification should not run in this scenario")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'guides' / 'nginx' / 'index.html'}`",
                f"- `{temp_dir / 'guides' / 'nginx' / 'chapters'}`",
                "",
            ]
        )
    )

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
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items.extend(
        [
            "First, examine the existing fortran guide structure and content to understand the format",
            "Create the nginx directory structure",
            "Develop the main index.html file for the nginx guide",
        ]
    )

    tool_call = ToolCall(
        id="working-note",
        name="notepad_write_working",
        arguments={"content": "Analyzing the fortran guide structure before creating nginx guide"},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Working note recorded",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert (
        "Continue with the next pending item: `First, examine the existing fortran guide structure and content to understand the format`."
        in message
    )
    assert "one concrete evidence-gathering tool call" in message
    assert "Resume by creating `index.html` now." not in message


@pytest.mark.asyncio
async def test_tool_batch_runner_working_note_prefers_declared_output_gap_over_stale_discovery(
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
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters_dir = guide_root / "chapters"
    chapters_dir.mkdir(parents=True)
    index_path = guide_root / "index.html"
    first_chapter = chapters_dir / "01-introduction.html"
    index_path.write_text(
        "\n".join(
            [
                '<a href="chapters/01-introduction.html">Introduction</a>',
                '<a href="chapters/02-installation.html">Installation</a>',
                '<a href="chapters/03-configuration.html">Configuration</a>',
            ]
        )
    )
    first_chapter.write_text("<h1>Introduction</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root / 'index.html'}`",
                f"- `{chapters_dir}/`",
                "",
            ]
        )
    )

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
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items.extend(
        [
            "First, examine the existing fortran guide structure and content to understand the format",
            "Create chapter files following the established pattern",
        ]
    )
    dod.touched_files.extend([str(index_path), str(first_chapter)])

    tool_call = ToolCall(
        id="working-note",
        name="notepad_write_working",
        arguments={"content": "Created index and first chapter; next is chapter 2"},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output="Working note recorded",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages
    message = queued_messages[-1]
    assert "Bookkeeping note is recorded. A declared output artifact is still missing." in message
    assert "Resume by creating `02-installation.html` now." in message
    assert "Continue with the next pending item: `First, examine the existing fortran guide structure" not in message


@pytest.mark.asyncio
async def test_tool_batch_runner_shallow_glob_does_not_handoff_before_content_read(
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
        raise AssertionError("Verification should not run in this scenario")

    fortran_root = temp_dir / "Loader" / "guides" / "fortran"
    chapters_dir = fortran_root / "chapters"
    chapters_dir.mkdir(parents=True)

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{temp_dir / 'Loader' / 'guides' / 'nginx' / 'index.html'}`",
                f"- `{temp_dir / 'Loader' / 'guides' / 'nginx' / 'chapters'}`",
                "",
            ]
        )
    )

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
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.pending_items.extend(
        [
            "First, examine the existing fortran guide structure and content",
            "Create the nginx directory structure",
            "Develop the main index.html file for nginx guide",
        ]
    )

    tool_call = ToolCall(
        id="glob-1",
        name="glob",
        arguments={"pattern": "**", "path": str(fortran_root)},
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=f"{fortran_root}\n{chapters_dir}",
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
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages == []


@pytest.mark.asyncio
async def test_tool_batch_runner_hands_off_noop_toc_edit_when_file_is_already_valid(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence scoring should not run in this scenario")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run in this scenario")

    prompt = (
        "Have a look at ~/Loader/guides/fortran/index.html, then "
        "~/Loader/guides/fortran/chapters. The table of contents links in "
        "index.html are inaccurate and the href’s are wrong. Let’s update the "
        "links and their link texts to be correct."
    )
    chapters = temp_dir / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    (chapters / "02-setup.html").write_text(
        "<h1>Chapter 2: Setting Up Your Environment</h1>\n"
    )
    current_block = (
        "<h2>Table of Contents</h2>\n"
        '        <ul class="chapter-list">\n'
        '            <li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>\n'
        '            <li><a href="chapters/02-setup.html">Chapter 2: Setting Up Your Environment</a></li>\n'
        "        </ul>\n"
    )
    index_path = temp_dir / "index.html"
    index_path.write_text(current_block)

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    context.session.current_task = prompt  # type: ignore[attr-defined]
    queued_messages: list[str] = []
    context.queue_steering_message_callback = queued_messages.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="edit-1",
        name="edit",
        arguments={
            "file_path": str(index_path),
            "old_string": current_block,
            "new_string": current_block,
        },
    )
    executor = FakeExecutor(
        [
            tool_outcome(
                tool_call=tool_call,
                output=(
                    "[Blocked - old_string and new_string are identical - no change "
                    "would occur] Suggestion: Provide different old and new strings"
                ),
                is_error=True,
                state=ToolExecutionState.BLOCKED,
            )
        ]
    )

    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=TurnSummary(final_response=""),
        dod=create_definition_of_done(prompt),
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued_messages == []


def test_tool_batch_runner_blocked_noop_edit_nudge_stays_on_active_repair_target(
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
        raise AssertionError("Verification should not run in this scenario")

    repair_target = temp_dir / "guide" / "chapters" / "04-basic-usage.html"
    context = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `05-advanced-topics.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{temp_dir / 'guide' / 'chapters' / '05-advanced-topics.html'}`; otherwise remove or replace `05-advanced-topics.html`.\n"
                ),
            )
        ],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    queued: list[str] = []
    context.queue_steering_message_callback = queued.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    runner._queue_blocked_html_edit_nudge(
        ToolCall(
            id="edit-1",
            name="edit",
            arguments={
                "file_path": str(repair_target),
                "old_string": "same",
                "new_string": "same",
            },
        ),
        "[Blocked - old_string and new_string are identical - no change would occur] Suggestion: Provide different old and new strings",
    )

    assert queued
    assert str(repair_target) in queued[0]
    assert "no on-disk change" in queued[0]
    assert "replace the surrounding block" in queued[0]
    assert "Do not reopen unrelated reference materials" in queued[0]


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
async def test_tool_batch_runner_does_not_mark_verification_planned_after_setup_only_mkdir(
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
        raise AssertionError("Verification should not run in this scenario")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    nginx_root = temp_dir / "Loader" / "guides" / "nginx"
    chapters = nginx_root / "chapters"
    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{chapters}/`",
                f"- `{nginx_root / 'index.html'}`",
                "",
            ]
        )
    )

    tool_call = ToolCall(
        id="mkdir-1",
        name="bash",
        arguments={"command": f"mkdir -p {chapters}"},
    )
    executor = FakeExecutor(
        [tool_outcome(tool_call=tool_call, output="", is_error=False)]
    )
    summary = TurnSummary(final_response="")
    dod = create_definition_of_done("Create an equally thorough nginx guide with chapters.")
    dod.implementation_plan = str(implementation_plan)
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

    assert dod.last_verification_result is None
    assert "Collect verification evidence" not in dod.pending_items
    assert not any(
        entry.reason_code == "verification_planned" for entry in summary.workflow_timeline
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


def test_tool_batch_runner_blocked_active_repair_nudge_uses_repair_scope(temp_dir: Path) -> None:
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
        raise AssertionError("Verification should not run in this scenario")

    repair_target = temp_dir / "guide" / "index.html"
    context = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `chapters/01-getting-started.html` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{temp_dir / 'guide' / 'chapters' / '01-getting-started.html'}`; otherwise remove or replace `chapters/01-getting-started.html`.\n"
                ),
            )
        ],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    queued: list[str] = []
    context.queue_steering_message_callback = queued.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    runner._queue_blocked_active_repair_nudge(
        "[Blocked - active repair scope: verification already identified the repair target.]"
    )

    assert queued
    assert str(repair_target) in queued[0]
    assert str(temp_dir / "guide" / "chapters" / "01-getting-started.html") in queued[0]
    assert "Do not reopen unrelated reference materials" in queued[0]


def test_tool_batch_runner_blocked_active_repair_mutation_nudge_uses_allowed_paths(
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
        raise AssertionError("Verification should not run in this scenario")

    repair_target = temp_dir / "guide" / "chapters" / "05-advanced-configurations.html"
    stylesheet = temp_dir / "guide" / "styles.css"
    context = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Repair focus:\n"
                    f"- Fix the broken local reference `../styles.css` in `{repair_target}`.\n"
                    f"- Immediate next step: edit `{repair_target}`.\n"
                    f"- If the broken reference should remain, create `{stylesheet}`; otherwise remove or replace `../styles.css`.\n"
                ),
            )
        ],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    queued: list[str] = []
    context.queue_steering_message_callback = queued.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    runner._queue_blocked_active_repair_mutation_nudge(
        "[Blocked - active repair mutation scope: verification already identified the repair target.]"
    )

    assert queued
    assert str(repair_target) in queued[0]
    assert str(stylesheet) in queued[0]
    assert "before widening the change set" in queued[0]


def test_tool_batch_runner_blocked_late_reference_drift_nudge_points_to_missing_artifact(
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
        raise AssertionError("Verification should not run in this scenario")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    queued: list[str] = []
    context.queue_steering_message_callback = queued.append
    store = DefinitionOfDoneStore(temp_dir)
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    plan_path = temp_dir / "implementation.md"
    plan_path.write_text(
        "# File Changes\n"
        "- `guide/index.html`\n"
        "- `guide/chapters/01-getting-started.html`\n"
        "- `guide/chapters/02-installation.html`\n"
        "- `guide/chapters/03-first-website.html`\n"
    )
    dod.implementation_plan = str(plan_path)
    (temp_dir / "guide" / "chapters").mkdir(parents=True, exist_ok=True)
    (temp_dir / "guide" / "index.html").write_text("index")
    (temp_dir / "guide" / "chapters" / "01-getting-started.html").write_text("one")
    (temp_dir / "guide" / "chapters" / "02-installation.html").write_text("two")
    runner = ToolBatchRunner(context, store)

    runner._queue_blocked_late_reference_drift_nudge(
        "[Blocked - late reference drift: several planned artifacts already exist.]",
        dod=dod,
    )

    assert queued
    assert "03-first-website.html" in queued[0]
    assert "older reference materials" in queued[0]


def test_tool_batch_runner_blocked_completed_artifact_scope_nudge_prefers_verification(
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
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guide"
    chapters = guide_root / "chapters"
    guide_root.mkdir(parents=True)
    chapters.mkdir()
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-getting-started.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("index")
    chapter_one.write_text("one")
    chapter_two.write_text("two")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{guide_root}`",
                f"- `{chapters}`",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    queued: list[str] = []
    context.queue_steering_message_callback = queued.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    dod = create_definition_of_done("Create a multi-file guide from a reference")
    dod.implementation_plan = str(implementation_plan)
    dod.verification_commands = [f"ls -la {guide_root}"]
    sync_todos_to_definition_of_done(
        dod,
        [
            {
                "content": "Verify all guide files are linked and complete",
                "active_form": "Working on: Verify all guide files are linked and complete",
                "status": "pending",
            }
        ],
        project_root=temp_dir,
    )

    runner._queue_blocked_completed_artifact_scope_nudge(
        "[Blocked - completed artifact set scope: all explicitly planned artifacts already exist.]",
        dod=dod,
    )

    assert queued
    assert "All explicitly planned artifacts already exist." in queued[0]
    assert "Verify all guide files are linked and complete" in queued[0]
    assert "Do not reopen earlier reference materials." in queued[0]


def test_tool_batch_runner_blocked_html_declared_target_nudge_uses_closest_declared_target(
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
        raise AssertionError("Verification should not run in this scenario")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    queued: list[str] = []
    context.queue_steering_message_callback = queued.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))

    runner._queue_blocked_html_declared_target_nudge(
        ToolCall(
            id="write-ch1",
            name="write",
            arguments={"file_path": str(temp_dir / "guide" / "chapters" / "01-introduction.html")},
        ),
        (
            "[Blocked - HTML page introduces new local targets outside the current declared artifact set] "
            "Suggestion: Keep non-root HTML pages within the root-declared local-link set and avoid "
            "introducing new sibling targets that the guide root does not declare, for example fix: 02-setup.html. "
            "Already-declared local targets include: chapters/01-introduction.html, chapters/02-installation.html, "
            "chapters/03-configuration.html. Closest declared local targets include: chapters/02-installation.html"
        ),
    )

    assert queued
    assert str(temp_dir / "guide" / "chapters" / "01-introduction.html") in queued[0]
    assert "`chapters/02-installation.html`" in queued[0]
    assert "same file now" in queued[0]


@pytest.mark.asyncio
async def test_tool_batch_runner_blocked_empty_file_path_nudges_concrete_next_artifact(
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
        raise AssertionError("Verification should not run in this scenario")

    guide_root = temp_dir / "guides" / "nginx"
    chapters = guide_root / "chapters"
    chapters.mkdir(parents=True)
    index_path = guide_root / "index.html"
    chapter_one = chapters / "01-introduction.html"
    chapter_two = chapters / "02-installation.html"
    index_path.write_text("<html></html>\n")
    chapter_one.write_text("<h1>Intro</h1>\n")

    implementation_plan = temp_dir / "implementation.md"
    implementation_plan.write_text(
        "\n".join(
            [
                "# Implementation Plan",
                "",
                "## File Changes",
                f"- `{index_path}`",
                f"- `{chapter_one}`",
                f"- `{chapter_two}`",
                "",
            ]
        )
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        safeguards=FakeSafeguards(),
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        auto_recover=False,
    )
    queued: list[str] = []
    context.queue_steering_message_callback = queued.append
    runner = ToolBatchRunner(context, DefinitionOfDoneStore(temp_dir))
    tool_call = ToolCall(
        id="write-2",
        name="write",
        arguments={"file_path": "", "content": "<html></html>\n"},
    )
    blocked_message = "[Blocked - Empty file path] Suggestion: Provide a valid file path"
    executor = FakeExecutor(
        [
            ToolExecutionOutcome(
                tool_call=tool_call,
                state=ToolExecutionState.BLOCKED,
                message=Message.tool_result_message(
                    tool_call_id=tool_call.id,
                    display_content=blocked_message,
                    result_content=blocked_message,
                    is_error=True,
                ),
                event_content=blocked_message,
                is_error=True,
                result_output=blocked_message,
            )
        ]
    )
    dod = create_definition_of_done("Create a multi-file nginx guide.")
    dod.implementation_plan = str(implementation_plan)
    dod.touched_files.extend([str(index_path), str(chapter_one)])
    dod.pending_items.append("Creating Chapter 2: Installation and Setup")

    await runner.execute_batch(
        tool_calls=[tool_call],
        tool_source="assistant",
        pending_tool_calls_seen=set(),
        emit=_noop_emit,
        summary=TurnSummary(final_response=""),
        dod=dod,
        executor=executor,  # type: ignore[arg-type]
        on_confirmation=None,
        on_user_question=None,
        emit_confirmation=None,
        consecutive_errors=0,
    )

    assert queued
    assert "did not provide a valid `file_path`" in queued[0]
    assert "Resume by creating `02-installation.html` now." in queued[0]
    assert (
        f"Prefer one `write` call for `{chapter_two}` instead of more rereads."
        in queued[0]
    )
    assert context.recovery_context is not None
    assert context.recovery_context.attempts[-1].error == blocked_message
