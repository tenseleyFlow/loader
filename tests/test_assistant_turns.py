"""Tests for assistant-turn helpers running on RuntimeContext."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.llm.base import CompletionResponse, Message, Role, StreamChunk, ToolCall
from loader.runtime.assistant_turns import AssistantTurnRequester
from loader.runtime.capabilities import resolve_backend_capability_profile
from loader.runtime.context import RuntimeContext, RuntimeLegacyServices
from loader.runtime.events import AgentEvent
from loader.runtime.permissions import (
    PermissionMode,
    build_permission_policy,
    load_permission_rules,
)
from loader.runtime.tracing import RuntimeTracer
from loader.tools.base import create_default_registry
from tests.helpers.runtime_harness import ScriptedBackend


@dataclass
class FakeCompaction:
    removed_message_count: int
    original_input_tokens: int
    compressed_input_tokens: int


class FakeSession:
    def __init__(
        self,
        *,
        project_root: Path,
        request_messages: list[Message],
        compaction: FakeCompaction | None = None,
    ) -> None:
        self.messages = list(request_messages)
        self.storage_path = project_root / ".loader" / "sessions" / "fake.json"
        self._request_messages = list(request_messages)
        self._compaction = compaction

    def maybe_compact(self) -> FakeCompaction | None:
        result = self._compaction
        self._compaction = None
        return result

    def build_request_messages(self) -> list[Message]:
        return list(self._request_messages)


class FakeCodeFilter:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class FakeSafeguards:
    def __init__(
        self,
        *,
        stream_transform=str,
        complete_transform=str,
        steering_message: str | None = None,
    ) -> None:
        self.action_tracker = object()
        self.validator = object()
        self.code_filter = FakeCodeFilter()
        self._stream_transform = stream_transform
        self._complete_transform = complete_transform
        self._steering_message = steering_message

    def filter_stream_chunk(self, content: str) -> str:
        return self._stream_transform(content)

    def filter_complete_content(self, content: str) -> str:
        return self._complete_transform(content)

    def should_steer(self) -> bool:
        return self._steering_message is not None

    def get_steering_message(self) -> str | None:
        return self._steering_message

    def record_response(self, content: str) -> None:
        return None

    def detect_text_loop(self, content: str) -> tuple[bool, str]:
        return False, ""

    def detect_loop(self) -> tuple[bool, str]:
        return False, ""


def build_runtime_context(
    *,
    temp_dir: Path,
    backend: ScriptedBackend,
    session: FakeSession,
    safeguards: FakeSafeguards,
    supports_native_tools: bool = True,
) -> tuple[RuntimeContext, list[str]]:
    registry = create_default_registry(temp_dir)
    registry.configure_workspace_root(temp_dir)
    rule_status = load_permission_rules(temp_dir)
    policy = build_permission_policy(
        active_mode=PermissionMode.WORKSPACE_WRITE,
        workspace_root=temp_dir,
        tool_requirements=registry.get_tool_requirements(),
        rules=rule_status.rules,
    )
    queued_messages: list[str] = []
    context = RuntimeContext(
        project_root=temp_dir,
        backend=backend,
        registry=registry,
        session=session,  # type: ignore[arg-type]
        config=SimpleNamespace(
            force_react=not supports_native_tools,
            stream=bool(backend._streams),
            temperature=0.2,
        ),
        capability_profile=resolve_backend_capability_profile(backend),
        project_context=None,
        permission_policy=policy,
        permission_config_status=rule_status,
        workflow_mode="execute",
        safeguards=safeguards,
        legacy=RuntimeLegacyServices(
            message_history=lambda: session.messages,  # type: ignore[attr-defined]
            drain_steering_queue=lambda: [],
            queue_steering_message=queued_messages.append,
            set_workflow_mode=lambda mode: None,
            refresh_capability_profile=lambda: None,
            assess_confidence=lambda tool_name, tool_args, context: None,  # type: ignore[arg-type]
            verify_action=lambda tool_name, tool_args, result, expected: None,  # type: ignore[arg-type]
            get_recovery_context=lambda: None,
            set_recovery_context=lambda value: None,
        ),
    )
    return context, queued_messages


@pytest.mark.asyncio
async def test_assistant_turn_requester_streams_with_fake_runtime_context(
    temp_dir: Path,
) -> None:
    pending_tool = ToolCall(id="call_read", name="read", arguments={"file_path": "README.md"})
    backend = ScriptedBackend(
        streams=[[
            StreamChunk(content="hello", pending_tool_call=pending_tool),
            StreamChunk(
                content="",
                full_content="hello",
                tool_calls=[pending_tool],
                is_done=True,
                usage={"completion_tokens": 1},
            ),
        ]],
        supports_native_tools=True,
    )
    session = FakeSession(
        project_root=temp_dir,
        request_messages=[Message(role=Role.USER, content="Read the file")],
        compaction=FakeCompaction(
            removed_message_count=2,
            original_input_tokens=120,
            compressed_input_tokens=50,
        ),
    )
    context, queued_messages = build_runtime_context(
        temp_dir=temp_dir,
        backend=backend,
        session=session,
        safeguards=FakeSafeguards(stream_transform=str.upper),
        supports_native_tools=True,
    )
    requester = AssistantTurnRequester(context, RuntimeTracer())
    events: list[AgentEvent] = []

    async def emit(event: AgentEvent) -> None:
        events.append(event)

    turn = await requester.request_turn(emit=emit, max_tokens=256)

    assert turn.content == "hello"
    assert turn.response_content == "hello"
    assert turn.tool_calls == [pending_tool]
    assert turn.pending_tool_calls_seen == {"call_read"}
    assert turn.usage == {"completion_tokens": 1}
    assert queued_messages == []
    assert context.safeguards.code_filter.reset_calls == 1
    assert [event.type for event in events] == ["artifact", "stream", "tool_call", "stream"]
    assert events[1].content == "HELLO"
    assert events[2].tool_name == "read"


@pytest.mark.asyncio
async def test_assistant_turn_requester_completes_with_fake_runtime_context(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(
        completions=[CompletionResponse(content="final answer", usage={"completion_tokens": 2})],
        supports_native_tools=False,
    )
    session = FakeSession(
        project_root=temp_dir,
        request_messages=[Message(role=Role.USER, content="Explain the issue")],
    )
    context, queued_messages = build_runtime_context(
        temp_dir=temp_dir,
        backend=backend,
        session=session,
        safeguards=FakeSafeguards(
            complete_transform=lambda content: f"FILTERED: {content}",
            steering_message="Stay on the task at hand.",
        ),
        supports_native_tools=False,
    )
    requester = AssistantTurnRequester(context, RuntimeTracer())

    async def emit(_: AgentEvent) -> None:
        return None

    turn = await requester.request_turn(emit=emit, max_tokens=256)

    assert turn.content == "FILTERED: final answer"
    assert turn.response_content == "final answer"
    assert turn.tool_calls == []
    assert turn.usage == {"completion_tokens": 2}
    assert queued_messages == ["Stay on the task at hand."]
    assert context.safeguards.code_filter.reset_calls == 1
