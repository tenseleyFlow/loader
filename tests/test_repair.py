"""Tests for response-repair helpers on RuntimeContext."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from loader.llm.base import ToolCall
from loader.runtime.context import RuntimeContext, RuntimeLegacyServices
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.repair import ResponseRepairer
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


class FakeSession:
    def __init__(self) -> None:
        self.messages = []

    def append(self, message) -> None:
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

    def detect_text_loop(self, content: str) -> tuple[bool, str]:
        return False, ""

    def detect_loop(self) -> tuple[bool, str]:
        return False, ""


def build_context(
    *,
    temp_dir: Path,
    use_react: bool,
    contains_unexecuted_code,
    extract_raw_json_tool_calls,
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
        config=SimpleNamespace(force_react=use_react),
        capability_profile=SimpleNamespace(supports_native_tools=not use_react),  # type: ignore[arg-type]
        project_context=None,
        permission_policy=policy,
        permission_config_status=rule_status,
        workflow_mode="execute",
        safeguards=FakeSafeguards(),
        legacy=RuntimeLegacyServices(
            message_history=lambda: session.messages,
            drain_steering_queue=lambda: [],
            queue_steering_message=lambda message: None,
            set_workflow_mode=lambda mode: None,
            refresh_capability_profile=lambda: None,
            self_critique=lambda response, task: None,  # type: ignore[arg-type]
            assess_confidence=lambda tool_name, tool_args, context: None,  # type: ignore[arg-type]
            verify_action=lambda tool_name, tool_args, result, expected: None,  # type: ignore[arg-type]
            contains_unexecuted_code=contains_unexecuted_code,
            extract_raw_json_tool_calls=extract_raw_json_tool_calls,
            get_recovery_context=lambda: None,
            set_recovery_context=lambda value: None,
        ),
    )


def test_response_repairer_uses_context_legacy_raw_fallback(temp_dir: Path) -> None:
    tool_call = ToolCall(id="raw_ask_0", name="AskUserQuestion", arguments={"question": "Which path?"})
    context = build_context(
        temp_dir=temp_dir,
        use_react=False,
        contains_unexecuted_code=lambda content: False,
        extract_raw_json_tool_calls=lambda content: [tool_call] if "AskUserQuestion" in content else [],
    )
    repairer = ResponseRepairer(context)

    analysis = repairer.analyze_response(
        content="I need clarification.",
        response_content="I should call AskUserQuestion tool now.",
        tool_calls=[],
        extracted_iterations=0,
        max_extracted_iterations=3,
    )

    assert analysis.tool_calls == [tool_call]
    assert analysis.tool_source == "raw_text"
    assert analysis.clear_stream is True
