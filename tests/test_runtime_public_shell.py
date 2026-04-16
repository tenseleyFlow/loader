"""Direct tests for runtime-owned public shell helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, Message, Role, StreamChunk
from loader.runtime.completion_trace import CompletionTraceEntry
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.public_shell import (
    SteeringMailbox,
    apply_runtime_session_install,
    build_event_emitter,
    build_fresh_runtime_session_install,
    build_runtime_few_shot_examples,
    build_runtime_system_message,
    clear_runtime_shell_history,
    create_runtime_session,
    create_runtime_session_install,
    get_runtime_shell_few_shot_examples,
    get_runtime_shell_system_message,
    load_runtime_session_install,
    refresh_runtime_capability_state,
    refresh_runtime_shell_capability_profile,
    resolve_runtime_shell_use_react,
    restore_runtime_session_state,
    resume_runtime_shell_session,
    run_runtime_shell,
    run_runtime_shell_explore,
    set_runtime_shell_workflow_mode,
    stream_runtime_shell,
)
from loader.runtime.runtime_handle import RuntimeHandle
from loader.runtime.session import ConversationSession
from tests.helpers.runtime_harness import ScriptedBackend


def _dummy_system() -> Message:
    return Message(role=Role.SYSTEM, content="system")


def _dummy_few_shots() -> list[Message]:
    return []


def _runtime_handle(
    temp_dir: Path,
    *,
    backend: ScriptedBackend | None = None,
    config: AgentConfig | None = None,
) -> RuntimeHandle:
    return RuntimeHandle(
        backend=backend or ScriptedBackend(),
        config=config or AgentConfig(auto_context=False),
        project_root=temp_dir,
    )


def test_create_runtime_session_copies_public_shell_state(temp_dir: Path) -> None:
    handle = _runtime_handle(temp_dir)

    session = create_runtime_session(
        project_root=handle.project_root,
        messages=handle.messages,
        permission_policy=handle.permission_policy,
        permission_config_status=handle.permission_config_status,
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context"],
        workflow_mode="execute",
        runtime_owner_type="RuntimeHandle",
        runtime_owner_path="runtime-handle",
        rotate_after_bytes=handle.config.session_rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            handle.config.session_auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=handle.config.session_compaction_keep_last_messages,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
    )

    assert session.permission_mode == handle.active_permission_mode
    assert session.permission_prompting_enabled is handle.permission_policy.prompting_enabled
    assert session.permission_rule_counts == handle.permission_policy.rule_counts()
    assert session.runtime_owner_type == "RuntimeHandle"
    assert session.runtime_owner_path == "runtime-handle"
    assert session.prompt_format == "native"
    assert session.prompt_sections == ["Runtime Config", "Workflow Context"]


def test_build_runtime_system_message_updates_session_metadata(temp_dir: Path) -> None:
    handle = _runtime_handle(temp_dir)
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )

    prompt_state = build_runtime_system_message(
        registry=handle.registry,
        use_react=False,
        project_context=None,
        workflow_mode="plan",
        permission_mode="workspace-write",
        cwd=temp_dir,
        current_task="Plan the next runtime refactor.",
        session=session,
    )

    assert prompt_state.prompt_format == "native"
    assert prompt_state.system_message.role == Role.SYSTEM
    assert "Plan Mode" in prompt_state.system_message.content
    assert session.prompt_format == "native"
    assert session.prompt_sections == prompt_state.prompt_sections
    assert len(session.prompt_history) == 1
    assert session.prompt_history[0].workflow_mode == "plan"
    assert session.prompt_history[0].current_task == "Plan the next runtime refactor."


def test_build_runtime_few_shot_examples_switches_tool_format() -> None:
    react_examples = build_runtime_few_shot_examples(use_react=True)
    native_examples = build_runtime_few_shot_examples(use_react=False)

    assert "<tool_call>" in react_examples[1].content
    assert '"background": true' in react_examples[5].content
    assert native_examples == []


def test_resolve_runtime_shell_use_react_respects_force_react_and_capabilities(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(
        temp_dir,
        backend=ScriptedBackend(supports_native_tools=True),
    )

    assert resolve_runtime_shell_use_react(handle) is False
    assert handle._use_react is False

    forced = _runtime_handle(
        temp_dir,
        backend=ScriptedBackend(supports_native_tools=True),
        config=AgentConfig(auto_context=False, force_react=True),
    )

    assert resolve_runtime_shell_use_react(forced) is True
    assert forced._use_react is True


def test_get_runtime_shell_system_message_caches_prompt_state_on_owner(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(temp_dir)

    first = get_runtime_shell_system_message(handle)
    second = get_runtime_shell_system_message(handle)

    assert first is second
    assert handle.prompt_format in {"native", "react"}
    assert handle.prompt_sections
    assert len(handle.session.prompt_history) == 1


def test_set_runtime_shell_workflow_mode_invalidates_prompt_cache(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(temp_dir)
    original = get_runtime_shell_system_message(handle)

    set_runtime_shell_workflow_mode(handle, "plan")

    assert handle.workflow_mode == "plan"
    assert handle._system_message is None

    updated = get_runtime_shell_system_message(handle)

    assert updated is not original
    assert handle.session.prompt_history[-1].workflow_mode == "plan"


def test_get_runtime_shell_few_shot_examples_uses_owner_prompt_mode(
    temp_dir: Path,
) -> None:
    native_handle = _runtime_handle(
        temp_dir,
        backend=ScriptedBackend(supports_native_tools=True),
    )
    react_handle = _runtime_handle(
        temp_dir,
        backend=ScriptedBackend(supports_native_tools=True),
        config=AgentConfig(auto_context=False, force_react=True),
    )

    native_examples = get_runtime_shell_few_shot_examples(native_handle)
    react_examples = get_runtime_shell_few_shot_examples(react_handle)

    assert native_examples == []
    assert "<tool_call>" in react_examples[1].content


@pytest.mark.asyncio
async def test_build_event_emitter_supports_sync_and_async_callbacks() -> None:
    seen: list[tuple[str, str]] = []

    def on_event_sync(event) -> None:
        seen.append(("sync", event.type))

    async def on_event_async(event) -> None:
        seen.append(("async", event.type))

    sync_emit = build_event_emitter(on_event_sync)
    async_emit = build_event_emitter(on_event_async)

    await sync_emit(SimpleNamespace(type="response"))
    await async_emit(SimpleNamespace(type="stream"))
    await build_event_emitter(None)(SimpleNamespace(type="ignored"))

    assert seen == [("sync", "response"), ("async", "stream")]


@pytest.mark.asyncio
async def test_run_runtime_shell_uses_runtime_launcher_entrypoint(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(
        temp_dir,
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Runtime shell reply.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
    )
    handle.config.reasoning.completion_check = False
    events = []

    async def capture(event) -> None:
        events.append(event)

    response = await run_runtime_shell(
        handle,
        "Summarize the runtime shell state.",
        on_event=capture,
        use_plan=False,
    )

    assert response == "Runtime shell reply."
    assert handle.last_turn_summary is not None
    assert handle.last_turn_summary.final_response == "Runtime shell reply."
    assert handle.steering.is_running is False
    assert any(event.type == "response" for event in events)


@pytest.mark.asyncio
async def test_stream_runtime_shell_yields_streamed_events(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(
        temp_dir,
        backend=ScriptedBackend(
            streams=[
                [
                    StreamChunk(content="Quick ", is_done=False),
                    StreamChunk(
                        content="reply.",
                        full_content="Quick reply.",
                        is_done=True,
                    ),
                ]
            ]
        ),
        config=AgentConfig(auto_context=False),
    )

    events = [event async for event in stream_runtime_shell(handle, "thanks")]

    assert any(event.type == "response" and event.content == "Quick reply." for event in events)
    assert handle.steering.is_running is False


@pytest.mark.asyncio
async def test_run_runtime_shell_explore_updates_last_turn_summary(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(
        temp_dir,
        backend=ScriptedBackend(
            completions=[CompletionResponse(content="Quick repo summary.")]
        ),
        config=AgentConfig(auto_context=False, stream=False),
    )
    events = []

    async def capture(event) -> None:
        events.append(event)

    response = await run_runtime_shell_explore(
        handle,
        "Give me a quick repo summary.",
        on_event=capture,
    )

    assert response == "Quick repo summary."
    assert handle.last_turn_summary is not None
    assert handle.last_turn_summary.workflow_mode == "explore"
    assert any(event.type == "response" for event in events)


def test_steering_mailbox_tracks_running_state_and_fifo_messages() -> None:
    mailbox = SteeringMailbox()

    assert mailbox.is_running is False
    assert mailbox.steer("stay in runtime") is False
    assert mailbox.drain() == []

    mailbox.mark_running()

    assert mailbox.steer("stay in runtime") is True

    mailbox.queue("double-check the current task")

    assert mailbox.drain() == [
        "stay in runtime",
        "double-check the current task",
    ]

    mailbox.mark_idle()
    assert mailbox.is_running is False

    mailbox.mark_running()
    mailbox.queue("stale message")
    mailbox.clear()
    assert mailbox.is_running is False
    assert mailbox.drain() == []


def test_refresh_runtime_capability_state_reports_prompt_reset_requirement() -> None:
    backend = ScriptedBackend(supports_native_tools=False)
    current_profile = SimpleNamespace(supports_native_tools=True)

    refresh = refresh_runtime_capability_state(
        backend=backend,
        current_profile=current_profile,  # type: ignore[arg-type]
    )

    assert refresh.capability_profile.supports_native_tools is False
    assert refresh.prompt_reset_required is True


def test_refresh_runtime_shell_capability_profile_updates_owner_cache_state(
    temp_dir: Path,
) -> None:
    backend = ScriptedBackend(supports_native_tools=True)
    handle = _runtime_handle(temp_dir, backend=backend)
    handle._system_message = Message(role=Role.SYSTEM, content="cached")
    handle._use_react = True
    backend._supports_native_tools = False  # type: ignore[attr-defined]

    refresh = refresh_runtime_shell_capability_profile(handle)

    assert refresh.capability_profile.supports_native_tools is False
    assert refresh.prompt_reset_required is True
    assert handle.capability_profile.supports_native_tools is False
    assert handle._system_message is None
    assert handle._use_react is None


def test_create_runtime_session_install_builds_restored_shell_state(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(temp_dir)

    install = create_runtime_session_install(
        project_root=handle.project_root,
        messages=handle.messages,
        permission_policy=handle.permission_policy,
        permission_config_status=handle.permission_config_status,
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context"],
        workflow_mode="execute",
        runtime_owner_type="RuntimeHandle",
        runtime_owner_path="runtime-handle",
        rotate_after_bytes=handle.config.session_rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            handle.config.session_auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=handle.config.session_compaction_keep_last_messages,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
    )

    assert install.session.permission_mode == handle.active_permission_mode
    assert install.restored.workflow_mode == "execute"
    assert install.restored.prompt_format == "native"
    assert install.restored.prompt_sections == ["Runtime Config", "Workflow Context"]
    assert install.restored.last_turn_summary is None


def test_apply_runtime_session_install_updates_owner_shell_state(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(temp_dir)
    install = create_runtime_session_install(
        project_root=handle.project_root,
        messages=[Message(role=Role.USER, content="Resume the runtime session.")],
        permission_policy=handle.permission_policy,
        permission_config_status=handle.permission_config_status,
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context"],
        workflow_mode="plan",
        runtime_owner_type="Agent",
        runtime_owner_path="public-agent",
        rotate_after_bytes=handle.config.session_rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            handle.config.session_auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=handle.config.session_compaction_keep_last_messages,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
    )
    install.session.permission_mode = "prompt"
    install.restored.current_task = "Resume the runtime session."
    install.restored.permission_mode = "prompt"

    apply_runtime_session_install(handle, install)

    assert handle.session is install.session
    assert handle.messages[-1].content == "Resume the runtime session."
    assert handle.current_task == "Resume the runtime session."
    assert handle.workflow_mode == "plan"
    assert handle.active_permission_mode == "prompt"
    assert handle.prompt_format == "native"
    assert handle.prompt_sections == ["Runtime Config", "Workflow Context"]
    assert handle.session.runtime_owner_type == "RuntimeHandle"
    assert handle.session.runtime_owner_path == "runtime-handle"


def test_build_fresh_runtime_session_install_uses_current_owner_shell_state(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(temp_dir)
    handle.current_task = "Keep the runtime shell tidy."
    handle.prompt_format = "native"
    handle.prompt_sections = ["Runtime Config", "Workflow Context"]
    handle.set_workflow_mode("clarify")

    install = build_fresh_runtime_session_install(
        handle,
        messages=[Message(role=Role.USER, content="Fresh runtime task.")],
    )

    assert install.session.prompt_format == "native"
    assert install.session.prompt_sections == ["Runtime Config", "Workflow Context"]
    assert install.restored.workflow_mode == "clarify"
    assert install.restored.messages[-1].content == "Fresh runtime task."


def test_restore_runtime_session_state_recovers_last_turn_summary(
    temp_dir: Path,
) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )
    dod = create_definition_of_done("Ship the runtime shell cleanup.")
    dod_path = DefinitionOfDoneStore(temp_dir).save(dod)
    session.active_dod_path = str(dod_path)
    session.current_task = "Ship the runtime shell cleanup."
    session.workflow_mode = "verify"
    session.workflow_reason_code = "verification_needed"
    session.workflow_reason_summary = "pending verification evidence remains"
    session.workflow_decision_kind = "handoff"
    session.last_completion_decision_code = "verification_passed"
    session.last_completion_decision_summary = (
        "accepted the response after verification evidence passed"
    )
    session.append_completion_trace_entry(
        CompletionTraceEntry(
            stage="definition_of_done",
            outcome="complete",
            decision_code="verification_passed",
            decision_summary="accepted the response after verification evidence passed",
        )
    )
    session.usage_totals = {"input_tokens": 10, "output_tokens": 4}

    restored = restore_runtime_session_state(
        project_root=temp_dir,
        session=session,
    )

    assert restored.current_task == "Ship the runtime shell cleanup."
    assert restored.workflow_mode == "verify"
    assert restored.last_turn_summary is not None
    assert (
        restored.last_turn_summary.definition_of_done.task_statement
        == "Ship the runtime shell cleanup."
    )
    assert restored.last_turn_summary.workflow_reason_code == "verification_needed"
    assert restored.last_completion_decision_code == "verification_passed"
    assert restored.last_completion_decision_summary == (
        "accepted the response after verification evidence passed"
    )
    assert restored.last_turn_summary.completion_decision_code == "verification_passed"
    assert restored.last_turn_summary.completion_trace[0].decision_code == (
        "verification_passed"
    )


def test_load_runtime_session_install_reconstructs_saved_shell_state(
    temp_dir: Path,
) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )
    session.current_task = "Resume the saved runtime session."
    session.workflow_mode = "execute"
    session.permission_mode = "prompt"
    session.prompt_format = "native"
    session.prompt_sections = ["Runtime Config", "Workflow Context"]
    session.append(Message(role=Role.USER, content="Resume the saved runtime session."))
    session.persist()

    install = load_runtime_session_install(
        project_root=temp_dir,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        session_id=session.session_id,
        rotate_after_bytes=256 * 1024,
        auto_compaction_input_tokens_threshold=100_000,
        compaction_keep_last_messages=4,
    )

    assert install is not None
    assert install.session.session_id == session.session_id
    assert install.restored.current_task == "Resume the saved runtime session."
    assert install.restored.permission_mode == "prompt"
    assert install.restored.messages[-1].content == "Resume the saved runtime session."


def test_resume_runtime_shell_session_restores_saved_owner_state(
    temp_dir: Path,
) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )
    session.current_task = "Resume the saved runtime session."
    session.workflow_mode = "plan"
    session.permission_mode = "prompt"
    session.prompt_format = "native"
    session.prompt_sections = ["Runtime Config", "Workflow Context"]
    session.append(Message(role=Role.USER, content="Resume the saved runtime session."))
    session.persist()

    handle = _runtime_handle(temp_dir)

    assert resume_runtime_shell_session(handle, session_id=session.session_id) is True
    assert handle.session.session_id == session.session_id
    assert handle.current_task == "Resume the saved runtime session."
    assert handle.workflow_mode == "plan"
    assert handle.active_permission_mode == "prompt"
    assert handle.prompt_format == "native"


def test_clear_runtime_shell_history_resets_owner_shell_state(
    temp_dir: Path,
) -> None:
    handle = _runtime_handle(temp_dir)
    original_session_id = handle.session.session_id
    handle.current_task = "Keep runtime state tidy."
    handle.prompt_format = "native"
    handle.prompt_sections = ["Runtime Config", "Workflow Context"]
    handle.set_workflow_mode("clarify")
    handle.queue_steering_message("Stay in runtime.")

    clear_runtime_shell_history(handle)

    assert handle.session.session_id != original_session_id
    assert handle.current_task is None
    assert handle.workflow_mode == "execute"
    assert handle.prompt_format is None
    assert handle.prompt_sections == []
    assert handle.messages == []
    assert handle.last_turn_summary is None
    assert handle.drain_steering_messages() == []
