"""Tests for finalization helpers on RuntimeContext."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import Message, Role
from loader.runtime.context import RuntimeContext, RuntimeLegacyServices
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.events import TurnSummary
from loader.runtime.finalization import TurnFinalizer
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.tracing import RuntimeTracer
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeSession:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.session_id = "session-test-123"
        self.recorded_calls: list[dict[str, object]] = []

    def append(self, message: Message) -> None:
        self.messages.append(message)

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
        legacy=RuntimeLegacyServices(
            drain_steering_queue=lambda: [],
            queue_steering_message=lambda message: None,
            set_workflow_mode=lambda mode: None,
            refresh_capability_profile=lambda: None,
            self_critique=lambda response, task: None,  # type: ignore[arg-type]
            assess_confidence=lambda tool_name, tool_args, context: None,  # type: ignore[arg-type]
            verify_action=lambda tool_name, tool_args, result, expected: None,  # type: ignore[arg-type]
            contains_unexecuted_code=lambda content: False,
            extract_raw_json_tool_calls=lambda content: [],
            get_recovery_context=lambda: None,
            set_recovery_context=lambda value: None,
        ),
    )


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
        set_workflow_mode=lambda mode, dod, emit, summary, reason: None,  # type: ignore[arg-type]
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
