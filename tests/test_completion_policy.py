"""Tests for completion-policy helpers on RuntimeContext."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import Message, Role
from loader.runtime.completion_policy import CompletionPolicy
from loader.runtime.context import RuntimeContext, RuntimeLegacyServices
from loader.runtime.events import AgentEvent, TurnSummary
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeSession:
    def __init__(self, messages: list[Message] | None = None) -> None:
        self.messages = list(messages or [])

    def append(self, message: Message) -> None:
        self.messages.append(message)


class FakeCodeFilter:
    def reset(self) -> None:
        return None


class FakeSafeguards:
    def __init__(self, *, text_loop_result: tuple[bool, str] = (False, "")) -> None:
        self.action_tracker = object()
        self.validator = object()
        self.code_filter = FakeCodeFilter()
        self._text_loop_result = text_loop_result

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
        return self._text_loop_result

    def detect_loop(self) -> tuple[bool, str]:
        return False, ""


def build_context(
    *,
    temp_dir: Path,
    safeguards: FakeSafeguards,
    use_quick_completion: bool = True,
    max_continuation_prompts: int = 5,
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
    session = FakeSession()
    return RuntimeContext(
        project_root=temp_dir,
        backend=ScriptedBackend(),
        registry=registry,
        session=session,  # type: ignore[arg-type]
        config=SimpleNamespace(
            force_react=False,
            reasoning=SimpleNamespace(
                rollback=False,
                show_rollback_plan=False,
                completion_check=True,
                use_quick_completion=use_quick_completion,
                max_continuation_prompts=max_continuation_prompts,
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
        safeguards=safeguards,
        legacy=RuntimeLegacyServices(
            message_history=lambda: session.messages,
            drain_steering_queue=lambda: [],
            queue_steering_message=lambda message: None,
            set_workflow_mode=lambda mode: None,
            refresh_capability_profile=lambda: None,
            self_critique=lambda response, task: None,  # type: ignore[arg-type]
            assess_confidence=lambda tool_name, tool_args, context: None,  # type: ignore[arg-type]
            verify_action=lambda tool_name, tool_args, result, expected: None,  # type: ignore[arg-type]
            contains_unexecuted_code=lambda content: False,
            get_recovery_context=lambda: None,
            set_recovery_context=lambda value: None,
        ),
    )


@pytest.mark.asyncio
async def test_completion_policy_stops_on_text_loop_using_context_safeguards(
    temp_dir: Path,
) -> None:
    context = build_context(
        temp_dir=temp_dir,
        safeguards=FakeSafeguards(text_loop_result=(True, "repeated continuation text")),
    )
    policy = CompletionPolicy(context)
    summary = TurnSummary(final_response="")
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    decision = await policy.maybe_stop_for_text_loop(
        content="I'll keep going. I'll keep going. I'll keep going.",
        emit=emit,
        summary=summary,
    )

    assert decision.should_stop is True
    assert summary.final_response == (
        "I seem to be repeating myself. "
        "Let me know if you'd like me to try a different approach."
    )
    assert context.session.messages[-1].role == Role.ASSISTANT
    assert any(event.type == "error" for event in events)
    assert any(event.type == "response" for event in events)


def test_completion_policy_finalize_response_text_keeps_original_response() -> None:
    response = CompletionPolicy.finalize_response_text(
        content="Inspected the file successfully.",
        actions_taken=["read: README.md"],
    )

    assert response == "Inspected the file successfully."
