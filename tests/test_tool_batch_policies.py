"""Direct tests for tool-batch confidence, verification, and recovery helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import Message, Role, ToolCall
from loader.runtime.context import RuntimeContext
from loader.runtime.events import AgentEvent
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
from loader.runtime.tool_batch_checks import (
    ToolBatchConfidenceGate,
    ToolBatchVerificationGate,
)
from loader.runtime.tool_batch_recovery import ToolBatchRecoveryController
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


def build_context(
    *,
    temp_dir: Path,
    messages: list[Message],
    assess_confidence,
    verify_action,
    recovery_context: RecoveryContext | None = None,
    confidence_scoring: bool = False,
    verification: bool = False,
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
    return RuntimeContext(
        project_root=temp_dir,
        backend=ScriptedBackend(),
        registry=registry,
        session=FakeSession(messages),  # type: ignore[arg-type]
        config=SimpleNamespace(
            force_react=False,
            max_recovery_attempts=2,
            auto_recover=True,
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
        safeguards=FakeSafeguards(),
        reasoning=SimpleNamespace(
            assess_confidence=assess_confidence,
            verify_action=verify_action,
        ),
        recovery_context=recovery_context,
    )


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
async def test_tool_batch_confidence_gate_skips_low_confidence_actions(
    temp_dir: Path,
) -> None:
    captured: dict[str, str] = {}

    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        captured["context"] = context
        return ConfidenceAssessment(
            action=f"{tool_name} with {tool_args}",
            tool_name=tool_name,
            tool_args=tool_args,
            level=ConfidenceLevel.LOW,
            reasoning="Need more context first.",
            risks=["Unknown file contents"],
        )

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run here")

    context = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(role=Role.USER, content="Inspect the README."),
            Message(role=Role.ASSISTANT, content="I'll read it next."),
        ],
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        confidence_scoring=True,
    )
    gate = ToolBatchConfidenceGate(context)
    tool_call = ToolCall(id="read-1", name="read", arguments={"file_path": "README.md"})
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    should_skip = await gate.should_skip(tool_call=tool_call, emit=emit)

    assert should_skip is True
    assert "Inspect the README." in captured["context"]
    assert context.session.messages[-1].role == Role.USER
    assert "[LOW CONFIDENCE WARNING]" in context.session.messages[-1].content
    assert [event.type for event in events] == ["confidence"]


@pytest.mark.asyncio
async def test_tool_batch_verification_gate_requests_correction(
    temp_dir: Path,
) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        return ActionVerification(
            tool_name=tool_name,
            tool_args=tool_args,
            expected_outcome="Success",
            actual_result=result,
            verified=True,
            discrepancies=["Output did not match the requested content"],
            needs_correction=True,
            correction_suggestion="Read the file before editing again.",
        )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        verification=True,
    )
    gate = ToolBatchVerificationGate(context)
    tool_call = ToolCall(id="read-1", name="read", arguments={"file_path": "README.md"})
    outcome = tool_outcome(tool_call=tool_call, output="unexpected contents", is_error=False)
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    should_continue = await gate.should_continue(
        tool_call=tool_call,
        outcome=outcome,
        emit=emit,
    )

    assert should_continue is True
    assert context.session.messages[-1].role == Role.USER
    assert "[VERIFICATION FAILED]" in context.session.messages[-1].content
    assert [event.type for event in events] == ["verification"]


@pytest.mark.asyncio
async def test_tool_batch_recovery_controller_returns_follow_up(
    temp_dir: Path,
) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run here")

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    context.session.current_task = (
        "Update index.html so every chapter link and title matches the real HTML files in chapters/."
    )
    controller = ToolBatchRecoveryController(context)
    tool_call = ToolCall(id="bash-1", name="bash", arguments={"command": "pytest"})
    outcome = tool_outcome(tool_call=tool_call, output="command failed", is_error=True)
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    follow_up = await controller.build_follow_up(
        tool_call=tool_call,
        outcome=outcome,
        emit=emit,
    )

    assert follow_up is not None
    assert context.recovery_context is not None
    assert "Previous attempts:" in follow_up.content
    assert any(event.type == "recovery" for event in events)


@pytest.mark.asyncio
async def test_tool_batch_recovery_controller_includes_known_state_for_missing_file(
    temp_dir: Path,
) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run here")

    messages = [
        Message(
            role=Role.TOOL,
            content=(
                "Observation [glob]: Result: "
                "/Users/mfwolffe/Loader/guides/fortran/chapters/01-introduction.html\n"
                "/Users/mfwolffe/Loader/guides/fortran/chapters/02-setup.html\n"
                "/Users/mfwolffe/Loader/guides/fortran/chapters/03-basics.html\n"
                "/Users/mfwolffe/Loader/guides/fortran/chapters/04-variables.html"
            ),
            tool_results=[],
        ),
        Message(
            role=Role.ASSISTANT,
            content="I already inspected the setup chapter.",
            tool_calls=[
                ToolCall(
                    id="read-setup",
                    name="read",
                    arguments={"file_path": "~/Loader/guides/fortran/chapters/02-setup.html"},
                )
            ],
        ),
        Message.tool_result_message(
            tool_call_id="read-setup",
            display_content="<h1>Chapter 2: Setting Up Fortran</h1>\n",
            result_content="<h1>Chapter 2: Setting Up Fortran</h1>\n",
        ),
        Message(
            role=Role.TOOL,
            content=(
                "Observation [notepad_write_working]: Result: "
                "- 02-basic-syntax.html -> 02-setup.html\n"
                "- 03-variables-data-types.html -> 03-basics.html\n"
                "- 04-operators-expressions.html -> 04-variables.html"
            ),
            tool_results=[],
        ),
        Message(
            role=Role.ASSISTANT,
            content="I should update the index now.",
            tool_calls=[
                ToolCall(
                    id="read-index",
                    name="read",
                    arguments={"file_path": "~/Loader/guides/fortran/index.html"},
                )
            ],
        ),
    ]
    context = build_context(
        temp_dir=temp_dir,
        messages=messages,
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    context.session.current_task = (
        "Update ~/Loader/guides/fortran/index.html with the right chapter links."
    )
    controller = ToolBatchRecoveryController(context)
    tool_call = ToolCall(
        id="read-missing",
        name="read",
        arguments={"file_path": "~/Loader/guides/fortran/chapters/04-data-types.html"},
    )
    outcome = tool_outcome(
        tool_call=tool_call,
        output="File not found: ~/Loader/guides/fortran/chapters/04-data-types.html",
        is_error=True,
    )

    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    follow_up = await controller.build_follow_up(
        tool_call=tool_call,
        outcome=outcome,
        emit=emit,
    )

    assert follow_up is not None
    assert "## CONTINUE FROM KNOWN STATE" in follow_up.content
    assert "apply the fix using confirmed findings" in follow_up.content
    assert "## ACTION BIAS FOR THIS RECOVERY" in follow_up.content
    assert "Prefer edit/write/patch on the target file" in follow_up.content
    assert "04-variables.html" in follow_up.content
    assert "02-basic-syntax.html -> 02-setup.html" in follow_up.content
    assert "02-setup.html = Chapter 2: Setting Up Fortran" in follow_up.content
    assert "/Users/mfwolffe/Loader/guides/fortran/index.html" in follow_up.content
    assert any(event.type == "recovery" for event in events)


@pytest.mark.asyncio
async def test_tool_batch_recovery_controller_suggests_known_sibling_files(
    temp_dir: Path,
) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run here")

    chapters = temp_dir / "chapters"
    chapters.mkdir()
    (chapters / "04-variables.html").write_text(
        "<h1>Chapter 4: Variables and Data Types</h1>\n"
    )
    (chapters / "05-input-output.html").write_text(
        "<h1>Chapter 5: Input and Output</h1>\n"
    )

    messages: list[Message] = []
    context = build_context(
        temp_dir=temp_dir,
        messages=messages,
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    controller = ToolBatchRecoveryController(context)
    tool_call = ToolCall(
        id="read-missing",
        name="read",
        arguments={"file_path": str(chapters / "04-data-types.html")},
    )
    outcome = tool_outcome(
        tool_call=tool_call,
        output=f"File not found: {chapters / '04-data-types.html'}",
        is_error=True,
    )

    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    follow_up = await controller.build_follow_up(
        tool_call=tool_call,
        outcome=outcome,
        emit=emit,
    )

    assert follow_up is not None
    assert "## LIKELY FILE CANDIDATES" in follow_up.content
    assert "`04-variables.html`" in follow_up.content
    assert "Chapter 4: Variables and Data Types" in follow_up.content
    assert "instead of retrying the missing path" in follow_up.content


@pytest.mark.asyncio
async def test_tool_batch_recovery_controller_includes_current_html_target_excerpt(
    temp_dir: Path,
) -> None:
    async def assess_confidence(tool_name: str, tool_args: dict, context: str) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(tool_name: str, tool_args: dict, result: str, expected: str = "") -> ActionVerification:
        raise AssertionError("Verification should not run here")

    chapters = temp_dir / "chapters"
    chapters.mkdir()
    (chapters / "01-introduction.html").write_text(
        "<h1>Chapter 1: Introduction to Fortran</h1>\n"
    )
    (chapters / "02-setup.html").write_text(
        "<h1>Chapter 2: Setting Up Your Environment</h1>\n"
    )
    index = temp_dir / "index.html"
    index.write_text(
        "<h2>Table of Contents</h2>\n"
        "<ul class=\"chapter-list\">\n"
        "    <li><a href=\"chapters/01-introduction.html\">Chapter 1: Introduction to Fortran</a></li>\n"
        "    <li><a href=\"chapters/02-basic-syntax.html\">Chapter 2: Basic Syntax</a></li>\n"
        "</ul>\n"
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    context.session.current_task = (
        "Update index.html so every chapter link and title matches the real HTML files in chapters/."
    )
    controller = ToolBatchRecoveryController(context)
    tool_call = ToolCall(
        id="patch-index",
        name="patch",
        arguments={
            "file_path": str(index),
            "hunks": [
                {
                    "old_start": 1,
                    "old_lines": 1,
                    "new_start": 1,
                    "new_lines": 1,
                    "lines": ["-bad", "+good"],
                }
            ],
        },
    )
    outcome = tool_outcome(
        tool_call=tool_call,
        output="Patch failed: hunk did not apply cleanly",
        is_error=True,
    )

    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    follow_up = await controller.build_follow_up(
        tool_call=tool_call,
        outcome=outcome,
        emit=emit,
    )

    assert follow_up is not None
    assert "## CURRENT TARGET EXCERPT" in follow_up.content
    assert "Verified chapter inventory:" in follow_up.content
    assert "<ul class=\"chapter-list\">" in follow_up.content
    assert "chapters/02-setup.html = Chapter 2: Setting Up Your Environment" in follow_up.content
    assert "Suggested replacement block:" in follow_up.content
    assert '<li><a href="chapters/02-setup.html">Chapter 2: Setting Up Your Environment</a></li>' in follow_up.content
    assert "Exact edit guidance:" in follow_up.content
    assert "old_string: use the Current TOC block above exactly" in follow_up.content
    assert "new_string: use the Suggested replacement block above exactly" in follow_up.content
    assert "Do not rewrite the whole file." in follow_up.content
    assert "Suggested edit call:" in follow_up.content
    assert 'old_string="""' in follow_up.content


@pytest.mark.asyncio
async def test_tool_batch_recovery_controller_scopes_known_state_to_active_target(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run here")

    nginx_chapters = temp_dir / "nginx" / "chapters"
    nginx_chapters.mkdir(parents=True)
    nginx_index = temp_dir / "nginx" / "index.html"
    nginx_index.write_text(
        "<h2>Table of Contents</h2>\n"
        "<ul>\n"
        '    <li><a href="chapters/01_getting_started.html">Getting Started with NGINX</a></li>\n'
        '    <li><a href="chapters/02_installation.html">Installation</a></li>\n'
        "</ul>\n"
    )
    (nginx_chapters / "01_getting_started.html").write_text(
        "<h1>Getting Started with NGINX</h1>\n"
    )

    context = build_context(
        temp_dir=temp_dir,
        messages=[
            Message(
                role=Role.TOOL,
                content=(
                    "Observation [read]: Result: "
                    f"{temp_dir / 'fortran' / 'index.html'}\n"
                    "Semantic verification preview: validated 12 toc links in index.html"
                ),
            ),
        ],
        assess_confidence=assess_confidence,
        verify_action=verify_action,
    )
    context.session.current_task = (  # type: ignore[attr-defined]
        "Have a look at ~/Loader/guides/fortran and chapters/ within. Get a feel "
        "for the structure and cadence of the guide. We are going to make an all "
        "new equally thorough guide on how to use the nginx tool. It will live in "
        "~/Loader/guides/nginx/index.html and ~/Loader/guides/nginx/chapters/."
    )
    controller = ToolBatchRecoveryController(context)
    tool_call = ToolCall(
        id="edit-nginx",
        name="edit",
        arguments={
            "file_path": str(nginx_index),
            "old_string": "<ul>\n</ul>",
            "new_string": "<ul class=\"chapter-list\">\n</ul>",
        },
    )
    outcome = tool_outcome(
        tool_call=tool_call,
        output=(
            "Tool execution error: EditTool.execute() missing 1 required positional "
            "argument: 'new_string'"
        ),
        is_error=True,
    )

    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    follow_up = await controller.build_follow_up(
        tool_call=tool_call,
        outcome=outcome,
        emit=emit,
    )

    assert follow_up is not None
    assert (
        "Preferred next step: Update "
        f"`{temp_dir / 'fortran' / 'index.html'}`"
    ) not in follow_up.content


@pytest.mark.asyncio
async def test_tool_batch_recovery_controller_reuses_context_for_related_missing_files(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run here")

    existing = RecoveryContext(
        original_tool="read",
        original_args={"file_path": "~/Loader/guides/fortran/chapters/04-data-types.html"},
        max_retries=3,
    )
    existing.add_attempt(
        "read",
        {"file_path": "~/Loader/guides/fortran/chapters/04-data-types.html"},
        "File not found: ~/Loader/guides/fortran/chapters/04-data-types.html",
    )
    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        recovery_context=existing,
    )
    controller = ToolBatchRecoveryController(context)
    tool_call = ToolCall(
        id="read-missing-2",
        name="read",
        arguments={"file_path": "~/Loader/guides/fortran/chapters/02-basic-syntax.html"},
    )
    outcome = tool_outcome(
        tool_call=tool_call,
        output="File not found: ~/Loader/guides/fortran/chapters/02-basic-syntax.html",
        is_error=True,
    )

    follow_up = await controller.build_follow_up(
        tool_call=tool_call,
        outcome=outcome,
        emit=lambda event: _noop_emit(event),
    )

    assert follow_up is not None
    assert context.recovery_context is existing
    assert len(existing.attempts) == 2
    assert "## Current attempt: 2/3" in follow_up.content
    assert "02-basic-syntax.html" in follow_up.content


@pytest.mark.asyncio
async def test_tool_batch_recovery_controller_resets_context_for_unrelated_failures(
    temp_dir: Path,
) -> None:
    async def assess_confidence(
        tool_name: str,
        tool_args: dict,
        context: str,
    ) -> ConfidenceAssessment:
        raise AssertionError("Confidence should not run here")

    async def verify_action(
        tool_name: str,
        tool_args: dict,
        result: str,
        expected: str = "",
    ) -> ActionVerification:
        raise AssertionError("Verification should not run here")

    existing = RecoveryContext(
        original_tool="read",
        original_args={"file_path": "~/Loader/guides/fortran/chapters/04-data-types.html"},
        max_retries=3,
    )
    existing.add_attempt(
        "read",
        {"file_path": "~/Loader/guides/fortran/chapters/04-data-types.html"},
        "File not found: ~/Loader/guides/fortran/chapters/04-data-types.html",
    )
    context = build_context(
        temp_dir=temp_dir,
        messages=[],
        assess_confidence=assess_confidence,
        verify_action=verify_action,
        recovery_context=existing,
    )
    controller = ToolBatchRecoveryController(context)
    tool_call = ToolCall(
        id="bash-timeout",
        name="bash",
        arguments={"command": "pytest"},
    )
    outcome = tool_outcome(
        tool_call=tool_call,
        output="command failed",
        is_error=True,
    )

    follow_up = await controller.build_follow_up(
        tool_call=tool_call,
        outcome=outcome,
        emit=lambda event: _noop_emit(event),
    )

    assert follow_up is not None
    assert context.recovery_context is not None
    assert context.recovery_context is not existing
    assert len(context.recovery_context.attempts) == 1
    assert "## Current attempt: 1/2" in follow_up.content


async def _noop_emit(event: AgentEvent) -> None:
    return None
