"""Tests for finalization helpers on RuntimeContext."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import Message, Role, ToolCall
from loader.runtime.completion_trace import CompletionTraceEntry
from loader.runtime.context import RuntimeContext
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.events import TurnSummary
from loader.runtime.executor import ToolExecutionOutcome, ToolExecutionState
from loader.runtime.finalization import TurnFinalizer
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.tracing import RuntimeTracer
from loader.runtime.verification_observations import VerificationObservationStatus
from loader.tools.base import ToolResult as RegistryToolResult
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeSession:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.session_id = "session-test-123"
        self.recorded_calls: list[dict[str, object]] = []
        self.last_completion_decision_code = "verification_passed"
        self.last_completion_decision_summary = (
            "accepted the response after verification evidence passed"
        )
        self.completion_trace = [
            CompletionTraceEntry(
                stage="definition_of_done",
                outcome="complete",
                decision_code="verification_passed",
                decision_summary="accepted the response after verification evidence passed",
            )
        ]
        self.last_turn_transition_summary = (
            "completion -> finalize [terminal] Finalizing completed turn"
        )
        self.workflow_timeline = []

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def append_workflow_timeline_entry(self, entry) -> None:
        self.workflow_timeline.append(entry)

    def record_turn_usage(
        self,
        usage: dict[str, int],
        *,
        tool_calls: int,
        iterations: int,
    ) -> dict[str, int]:
        payload = {
            "usage": dict(usage),
            "tool_calls": tool_calls,
            "iterations": iterations,
        }
        self.recorded_calls.append(payload)
        return {"turns": 1, "tool_calls": tool_calls, "iterations": iterations}


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

    def detect_text_loop(self, content: str) -> tuple[bool, str]:
        return False, ""

    def detect_loop(self) -> tuple[bool, str]:
        return False, ""


class FakeExecutor:
    def __init__(self, outcomes: list[ToolExecutionOutcome]) -> None:
        self._outcomes = list(outcomes)

    async def execute_tool_call(self, tool_call: ToolCall, **_: object) -> ToolExecutionOutcome:
        if not self._outcomes:
            raise AssertionError("No fake verification outcome queued")
        return self._outcomes.pop(0)


class RecordingExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def execute_tool_call(self, tool_call: ToolCall, **_: object) -> ToolExecutionOutcome:
        command = str(tool_call.arguments.get("command", ""))
        self.commands.append(command)
        return tool_outcome(
            tool_call=tool_call,
            output="ok",
            is_error=False,
            exit_code=0,
            stdout="ok",
        )


def build_context(temp_dir: Path, session: FakeSession) -> RuntimeContext:
    registry = create_default_registry(temp_dir)
    registry.configure_workspace_root(temp_dir)
    rule_status = load_permission_rules(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
        rules=rule_status.rules,
    )
    return RuntimeContext(
        project_root=temp_dir,
        backend=ScriptedBackend(),
        registry=registry,
        session=session,  # type: ignore[arg-type]
        config=SimpleNamespace(
            force_react=False,
            verification_retry_budget=3,
            reasoning=SimpleNamespace(
                rollback=False,
                show_rollback_plan=False,
                completion_check=True,
                use_quick_completion=True,
                max_continuation_prompts=5,
                self_critique=False,
                confidence_scoring=False,
                min_confidence_for_action=3,
                verification=False,
            ),
        ),
        capability_profile=SimpleNamespace(supports_native_tools=True),  # type: ignore[arg-type]
        project_context=None,
        permission_policy=policy,
        permission_config_status=rule_status,
        workflow_mode="execute",
        safeguards=FakeSafeguards(),
    )


def tool_outcome(
    *,
    tool_call: ToolCall,
    output: str,
    is_error: bool,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
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
            metadata={
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
            },
        ),
    )


async def _noop_set_workflow_mode(mode, dod, emit, summary) -> None:
    return None


def test_turn_finalizer_finalize_summary_uses_runtime_context(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    context = build_context(temp_dir, session)
    tracer = RuntimeTracer()
    tracer.record("turn.completed", reason="done")
    finalizer = TurnFinalizer(
        context,
        tracer,
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    dod = create_definition_of_done("Finish the task")
    dod.status = "done"
    summary = TurnSummary(
        final_response="All set.",
        definition_of_done=dod,
        iterations=2,
        usage={"prompt_tokens": 10},
        tool_result_messages=[Message(role=Role.TOOL, content="tool output")],
    )
    captured: dict[str, str] = {}

    def capture_definition_of_done(self, summary_text: str) -> None:
        captured["summary"] = summary_text

    monkeypatch.setattr(
        "loader.runtime.finalization.MemoryStore.capture_definition_of_done",
        capture_definition_of_done,
    )

    final_summary = finalizer.finalize_summary(summary)

    assert final_summary.session_id == "session-test-123"
    assert final_summary.cumulative_usage == {"turns": 1, "tool_calls": 1, "iterations": 2}
    assert session.recorded_calls == [
        {
            "usage": {"prompt_tokens": 10, "tool_calls": 1, "iterations": 2},
            "tool_calls": 1,
            "iterations": 2,
        }
    ]
    assert "summary" in captured
    assert final_summary.trace
    assert final_summary.completion_decision_code == "verification_passed"
    assert final_summary.completion_decision_summary == (
        "accepted the response after verification evidence passed"
    )
    assert [entry.decision_code for entry in final_summary.completion_trace] == [
        "verification_passed"
    ]


@pytest.mark.asyncio
async def test_turn_finalizer_records_skipped_verification_observation(
    temp_dir: Path,
) -> None:
    session = FakeSession()
    context = build_context(temp_dir, session)
    finalizer = TurnFinalizer(
        context,
        RuntimeTracer(),
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    dod = create_definition_of_done("Explain Loader's clarify loop.")
    summary = TurnSummary(final_response="")
    events = []

    async def capture(event) -> None:
        events.append(event)

    result = await finalizer.run_definition_of_done_gate(
        dod=dod,
        candidate_response="Loader uses a bounded clarify loop before execution.",
        emit=capture,
        summary=summary,
        executor=FakeExecutor([]),  # type: ignore[arg-type]
    )

    assert result.should_continue is False
    assert result.reason_code == "non_mutating_response_accepted"
    assert [item.status for item in result.verification_observations] == [
        VerificationObservationStatus.SKIPPED.value
    ]
    assert [item.summary for item in result.verification_observations] == [
        "verification was skipped because no mutating work required checks"
    ]
    assert summary.verification_status == "skipped"
    assert session.workflow_timeline[-1].kind == "verify_skip"
    assert [item.status for item in session.workflow_timeline[-1].verification_observations] == [
        VerificationObservationStatus.SKIPPED.value
    ]
    assert any(event.type == "dod_status" and event.dod_status == "done" for event in events)


@pytest.mark.asyncio
async def test_turn_finalizer_records_passed_verification_observation(
    temp_dir: Path,
) -> None:
    session = FakeSession()
    context = build_context(temp_dir, session)
    finalizer = TurnFinalizer(
        context,
        RuntimeTracer(),
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    dod = create_definition_of_done("Update the runtime tests.")
    dod.mutating_actions.append("write")
    dod.verification_commands = ["uv run pytest -q"]
    summary = TurnSummary(final_response="")
    tool_call = ToolCall(
        id="verify-1-1",
        name="bash",
        arguments={"command": "uv run pytest -q", "cwd": str(temp_dir)},
    )

    async def capture(event) -> None:
        return None

    result = await finalizer.run_definition_of_done_gate(
        dod=dod,
        candidate_response="Updated the runtime tests.",
        emit=capture,
        summary=summary,
        executor=FakeExecutor(
            [
                tool_outcome(
                    tool_call=tool_call,
                    output="219 passed",
                    is_error=False,
                    exit_code=0,
                    stdout="219 passed",
                )
            ]
        ),  # type: ignore[arg-type]
    )

    assert result.should_continue is False
    assert result.reason_code == "verification_passed"
    assert [item.status for item in result.verification_observations] == [
        VerificationObservationStatus.PASSED.value
    ]
    assert result.verification_observations[0].attempt_id == "verification-attempt-1"
    assert result.verification_observations[0].attempt_number == 1
    assert result.verification_observations[0].command == "uv run pytest -q"
    assert result.verification_observations[0].detail == "219 passed"
    assert summary.verification_status == "passed"
    assert [entry.reason_code for entry in session.workflow_timeline[-2:]] == [
        "verification_pending",
        "verification_command_passed",
    ]
    assert [item.status for item in session.workflow_timeline[-2].verification_observations] == [
        VerificationObservationStatus.PENDING.value
    ]
    assert (
        session.workflow_timeline[-2].verification_observations[0].attempt_id
        == "verification-attempt-1"
    )
    assert session.workflow_timeline[-2].verification_observations[0].command == (
        "uv run pytest -q"
    )
    assert session.workflow_timeline[-1].kind == "verify_observation"
    assert session.workflow_timeline[-1].reason_code == "verification_command_passed"
    assert [item.status for item in session.workflow_timeline[-1].verification_observations] == [
        VerificationObservationStatus.PASSED.value
    ]


@pytest.mark.asyncio
async def test_turn_finalizer_appends_runtime_semantic_verifier_to_planned_commands(
    temp_dir: Path,
) -> None:
    chapters = temp_dir / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    index = temp_dir / "index.html"
    index.write_text(
        "\n".join(
            [
                '<ul class="chapter-list">',
                '  <li><a href="chapters/01-introduction.html">Chapter 1: Introduction to Fortran</a></li>',
                "</ul>",
            ]
        )
    )

    session = FakeSession()
    context = build_context(temp_dir, session)
    finalizer = TurnFinalizer(
        context,
        RuntimeTracer(),
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    dod = create_definition_of_done(
        "Update index.html so the table of contents links and chapter titles are correct."
    )
    dod.mutating_actions.append("edit")
    dod.touched_files.append(str(index))
    dod.verification_commands = ['grep -n "href=" index.html']
    summary = TurnSummary(final_response="")
    executor = RecordingExecutor()

    async def capture(event) -> None:
        return None

    result = await finalizer.run_definition_of_done_gate(
        dod=dod,
        candidate_response="Updated the index.html links.",
        emit=capture,
        summary=summary,
        executor=executor,  # type: ignore[arg-type]
    )

    assert result.should_continue is False
    assert any(command == 'grep -n "href=" index.html' for command in executor.commands)
    assert any(command.startswith("python3 - <<'PY'") for command in executor.commands)
    assert (
        session.workflow_timeline[-1].verification_observations[0].attempt_id
        == "verification-attempt-1"
    )


@pytest.mark.asyncio
async def test_turn_finalizer_does_not_append_repo_defaults_to_external_verification_plan(
    temp_dir: Path,
) -> None:
    (temp_dir / "pyproject.toml").write_text("[project]\nname='loader'\n")
    (temp_dir / "package.json").write_text("{}\n")
    external_root = temp_dir.parent / "external-nginx-guide"
    external_root.mkdir(exist_ok=True)
    external_index = external_root / "index.html"
    external_index.write_text("<html></html>\n")

    session = FakeSession()
    context = build_context(temp_dir, session)
    finalizer = TurnFinalizer(
        context,
        RuntimeTracer(),
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    dod = create_definition_of_done("Create an external nginx guide.")
    dod.mutating_actions.append("write")
    dod.touched_files.append(str(external_index))
    dod.verification_commands = [
        f"ls -la {external_root}",
        f"grep -n \"html\" {external_index}",
    ]
    summary = TurnSummary(final_response="")
    executor = RecordingExecutor()

    async def capture(event) -> None:
        return None

    result = await finalizer.run_definition_of_done_gate(
        dod=dod,
        candidate_response="Created the external nginx guide.",
        emit=capture,
        summary=summary,
        executor=executor,  # type: ignore[arg-type]
    )

    assert result.should_continue is False
    assert executor.commands == [
        f"ls -la {external_root}",
        f'grep -n "html" {external_index}',
    ]


@pytest.mark.asyncio
async def test_turn_finalizer_records_missing_verification_observation(
    temp_dir: Path,
) -> None:
    session = FakeSession()
    context = build_context(temp_dir, session)
    finalizer = TurnFinalizer(
        context,
        RuntimeTracer(),
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    dod = create_definition_of_done("Edit the loader bootstrap.")
    dod.mutating_actions.append("edit")
    summary = TurnSummary(final_response="")

    async def capture(event) -> None:
        return None

    result = await finalizer.run_definition_of_done_gate(
        dod=dod,
        candidate_response="Updated the bootstrap code.",
        emit=capture,
        summary=summary,
        executor=FakeExecutor([]),  # type: ignore[arg-type]
    )

    assert result.should_continue is True
    assert result.reason_code == "verification_failed_reentry"
    assert [item.status for item in result.verification_observations] == [
        VerificationObservationStatus.MISSING.value
    ]
    assert result.verification_observations[0].attempt_id == "verification-attempt-1"
    assert result.verification_observations[0].attempt_number == 1
    assert [item.summary for item in result.verification_observations] == [
        "verification commands were still missing at execution time"
    ]
    assert summary.verification_status == "failed"
    assert session.workflow_timeline[-1].kind == "verify_observation"
    assert session.workflow_timeline[-1].reason_code == "verification_commands_missing"
    assert [item.status for item in session.workflow_timeline[-1].verification_observations] == [
        VerificationObservationStatus.MISSING.value
    ]
    assert (
        session.workflow_timeline[-1].verification_observations[0].attempt_id
        == "verification-attempt-1"
    )
    assert session.messages[-1].role == Role.USER
    assert session.messages[-1].content.startswith("[DEFINITION OF DONE CHECK FAILED]")


@pytest.mark.asyncio
async def test_turn_finalizer_does_not_reverify_without_new_changes(
    temp_dir: Path,
) -> None:
    session = FakeSession()
    context = build_context(temp_dir, session)
    finalizer = TurnFinalizer(
        context,
        RuntimeTracer(),
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    index = temp_dir / "index.html"
    index.write_text("<ul></ul>\n")
    dod = create_definition_of_done("Fix the chapter list in index.html.")
    dod.mutating_actions.append("edit")
    dod.touched_files.append(str(index))
    dod.line_changes = 12
    dod.last_verification_result = "failed"
    dod.last_verification_signature = (
        f"lines={dod.line_changes};touched={index};actions=1;commands="
    )
    dod.evidence = []
    summary = TurnSummary(final_response="")
    executor = RecordingExecutor()

    async def capture(event) -> None:
        return None

    result = await finalizer.run_definition_of_done_gate(
        dod=dod,
        candidate_response="I checked the file again.",
        emit=capture,
        summary=summary,
        executor=executor,  # type: ignore[arg-type]
    )

    assert result.should_continue is True
    assert result.reason_code == "verification_failed_no_new_changes"
    assert executor.commands == []
    assert summary.verification_status == "failed"
    assert session.messages[-1].content.startswith("[DEFINITION OF DONE CHECK STILL FAILING]")


@pytest.mark.asyncio
async def test_turn_finalizer_accepts_missing_optional_html5validator_when_semantic_check_passes(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = FakeSession()
    context = build_context(temp_dir, session)
    finalizer = TurnFinalizer(
        context,
        RuntimeTracer(),
        DefinitionOfDoneStore(temp_dir),
        set_workflow_mode=_noop_set_workflow_mode,
    )
    dod = create_definition_of_done(
        "Update index.html so the table of contents links and chapter titles are correct."
    )
    dod.mutating_actions.append("edit")
    dod.touched_files.append(str(temp_dir / "index.html"))
    dod.verification_commands = [
        "python3 - <<'PY'\nprint('semantic ok')\nPY",
        "html5validator --root /tmp/fortran-qwen-recovery-check/",
    ]
    summary = TurnSummary(final_response="")
    semantic_call = ToolCall(
        id="verify-1-1",
        name="bash",
        arguments={"command": dod.verification_commands[0], "cwd": str(temp_dir)},
    )
    html5validator_call = ToolCall(
        id="verify-1-2",
        name="bash",
        arguments={"command": dod.verification_commands[1], "cwd": str(temp_dir)},
    )

    async def capture(event) -> None:
        return None

    monkeypatch.setattr(
        "loader.runtime.finalization.derive_verification_commands",
        lambda *args, **kwargs: [],
    )

    result = await finalizer.run_definition_of_done_gate(
        dod=dod,
        candidate_response="Updated the chapter links and titles.",
        emit=capture,
        summary=summary,
        executor=FakeExecutor(
            [
                tool_outcome(
                    tool_call=semantic_call,
                    output="semantic ok",
                    is_error=False,
                    exit_code=0,
                    stdout="semantic ok",
                ),
                tool_outcome(
                    tool_call=html5validator_call,
                    output="/bin/sh: html5validator: command not found",
                    is_error=True,
                    exit_code=127,
                    stderr="/bin/sh: html5validator: command not found",
                ),
            ]
        ),  # type: ignore[arg-type]
    )

    assert result.should_continue is False
    assert result.reason_code == "verification_passed"
    assert summary.verification_status == "passed"
    assert dod.status == "done"
    assert dod.last_verification_result == "passed"
    assert [item.passed for item in dod.evidence] == [True, False]
    assert [item.skipped for item in dod.evidence] == [False, True]
    assert "SKIP" in result.final_response
    assert "html5validator" in result.final_response
    assert session.workflow_timeline[-2].reason_code == "verification_command_passed"
    assert session.workflow_timeline[-1].reason_code == "verification_command_skipped"
    assert [item.status for item in session.workflow_timeline[-1].verification_observations] == [
        VerificationObservationStatus.SKIPPED.value
    ]
