"""Tests for completion-policy helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import Message, Role
from loader.runtime.completion_policy import CompletionPolicy
from loader.runtime.context import RuntimeContext
from loader.runtime.events import TurnSummary
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.task_completion import (
    assess_completion_follow_through,
    detect_premature_completion,
    get_continuation_prompt,
)
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeCodeFilter:
    def reset(self) -> None:
        return None


class FakeSafeguards:
    def __init__(self, *, text_loop: tuple[bool, str] = (False, "")) -> None:
        self.action_tracker = object()
        self.validator = object()
        self.code_filter = FakeCodeFilter()
        self._text_loop = text_loop
        self.recorded: list[str] = []

    def filter_stream_chunk(self, content: str) -> str:
        return content

    def filter_complete_content(self, content: str) -> str:
        return content

    def should_steer(self) -> bool:
        return False

    def get_steering_message(self) -> str | None:
        return None

    def record_response(self, content: str) -> None:
        self.recorded.append(content)

    def detect_text_loop(self, content: str) -> tuple[bool, str]:
        return self._text_loop

    def detect_loop(self) -> tuple[bool, str]:
        return False, ""


class FakeSession:
    def __init__(self) -> None:
        self.messages: list[Message] = []

    def append(self, message: Message) -> None:
        self.messages.append(message)


def build_context(
    temp_dir: Path,
    *,
    safeguards: FakeSafeguards,
    max_continuation_prompts: int = 5,
    use_quick_completion: bool = True,
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
        session=FakeSession(),  # type: ignore[arg-type]
        config=SimpleNamespace(
            force_react=False,
            reasoning=SimpleNamespace(
                max_continuation_prompts=max_continuation_prompts,
                use_quick_completion=use_quick_completion,
            ),
        ),
        capability_profile=SimpleNamespace(supports_native_tools=True),  # type: ignore[arg-type]
        project_context=None,
        permission_policy=policy,
        permission_config_status=rule_status,
        workflow_mode="execute",
        safeguards=safeguards,
    )


def test_completion_policy_finalize_response_text_keeps_original_response() -> None:
    response = CompletionPolicy.finalize_response_text(
        content="Inspected the file successfully.",
        actions_taken=["read: README.md"],
    )

    assert response == "Inspected the file successfully."


def test_detect_premature_completion_respects_explicit_done_without_actions() -> None:
    assert detect_premature_completion(
        "Explain how Loader works.",
        "Done.",
        [],
    ) is False


def test_get_continuation_prompt_surfaces_missing_verification_steps() -> None:
    prompt = get_continuation_prompt(
        "Create the script and test that it works.",
        ["write: script.py"],
        "The script has been created.",
    )

    assert "Continue with" in prompt
    assert "run the relevant tests" in prompt.lower() or "verify" in prompt.lower()


def test_assess_completion_follow_through_tracks_missing_evidence() -> None:
    check = assess_completion_follow_through(
        task="Create the script and test that it works.",
        response="The script has been created.",
        actions_taken=["write: script.py"],
    )

    assert check.is_complete is False
    assert "showing the requested work was actually carried out" in check.required_evidence
    assert "showing the result was run or verified" in check.required_evidence
    assert check.missing_evidence == ["showing the result was run or verified"]
    assert check.suggested_next_steps == [
        "Execute what you created or run the relevant tests now"
    ]


def test_assess_completion_follow_through_accepts_informational_tasks() -> None:
    check = assess_completion_follow_through(
        task="Explain how Loader's workflow timeline works.",
        response="Loader records workflow decisions and policy events in a timeline.",
        actions_taken=[],
    )

    assert check.is_complete is True
    assert check.required_evidence == []
    assert check.missing_evidence == []


@pytest.mark.asyncio
async def test_completion_policy_stops_for_text_loop_using_runtime_context(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir,
        safeguards=FakeSafeguards(text_loop=(True, "assistant repeated the same summary")),
    )
    policy = CompletionPolicy(context)
    summary = TurnSummary(final_response="")
    events = []

    async def emit(event) -> None:
        events.append(event)

    decision = await policy.maybe_stop_for_text_loop(
        content="Same summary again.",
        emit=emit,
        summary=summary,
    )

    assert decision.should_stop is True
    assert decision.decision_code == "text_loop_bailout"
    assert decision.decision_summary == (
        "stopped after detecting a repeated text loop"
    )
    assert summary.final_response == (
        "I stopped because I was repeating myself and couldn't make further progress."
    )
    assert summary.assistant_messages[-1].role == Role.ASSISTANT
    assert context.session.messages[-1].content == summary.final_response
    assert events[0].type == "error"
    assert events[1].type == "response"


@pytest.mark.asyncio
async def test_completion_policy_requests_continuation_using_runtime_context(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir,
        safeguards=FakeSafeguards(),
    )
    policy = CompletionPolicy(context)
    events = []

    async def emit(event) -> None:
        events.append(event)

    decision = await policy.maybe_continue_for_completion(
        content="I can handle that.",
        response_content="I can handle that.",
        task="Create the file and verify it works.",
        actions_taken=[],
        continuation_count=0,
        emit=emit,
    )

    assert decision.should_continue is True
    assert decision.decision_code == "premature_completion_nudge"
    assert decision.decision_summary == (
        "requested one continuation because the non-mutating response looked incomplete"
    )
    assert decision.completion_check is not None
    assert decision.completion_check.missing_evidence == [
        "showing the requested work was actually carried out",
        "showing the result was run or verified",
    ]
    assert context.session.messages[-2] == Message(
        role=Role.ASSISTANT,
        content="I can handle that.",
    )
    assert context.session.messages[-1].role == Role.USER
    assert "verify it works" in context.session.messages[-1].content.lower()
    assert events[0].type == "completion_check"
    assert events[0].completion_check is not None
    assert events[0].completion_check.missing_evidence == [
        "showing the requested work was actually carried out",
        "showing the result was run or verified",
    ]


@pytest.mark.asyncio
async def test_completion_policy_finalizes_when_budget_is_exhausted(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir,
        safeguards=FakeSafeguards(),
        max_continuation_prompts=1,
    )
    policy = CompletionPolicy(context)
    events = []

    async def emit(event) -> None:
        events.append(event)

    decision = await policy.maybe_continue_for_completion(
        content="I looked into it.",
        response_content="I looked into it.",
        task="Fix the README heading.",
        actions_taken=[],
        continuation_count=1,
        emit=emit,
    )

    assert decision.should_continue is False
    assert decision.should_finalize is True
    assert decision.decision_code == "continuation_budget_exhausted"
    assert decision.completion_check is not None
    assert decision.completion_check.missing_evidence == [
        "showing the requested work was actually carried out"
    ]
    assert "Missing evidence" in decision.final_response
    assert events[0].type == "completion_check"
