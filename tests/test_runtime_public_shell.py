"""Direct tests for runtime-owned public shell helpers."""

from __future__ import annotations

from pathlib import Path

from loader.agent.loop import Agent, AgentConfig
from loader.llm.base import Message, Role
from loader.runtime.dod import DefinitionOfDoneStore, create_definition_of_done
from loader.runtime.public_shell import (
    build_runtime_few_shot_examples,
    build_runtime_system_message,
    create_runtime_session,
    restore_runtime_session_state,
)
from loader.runtime.session import ConversationSession
from tests.helpers.runtime_harness import ScriptedBackend


def _dummy_system() -> Message:
    return Message(role=Role.SYSTEM, content="system")


def _dummy_few_shots() -> list[Message]:
    return []


def test_create_runtime_session_copies_public_shell_state(temp_dir: Path) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )

    session = create_runtime_session(
        project_root=agent.project_root,
        messages=agent.messages,
        permission_policy=agent.permission_policy,
        permission_config_status=agent.permission_config_status,
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context"],
        workflow_mode="execute",
        rotate_after_bytes=agent.config.session_rotate_after_bytes,
        auto_compaction_input_tokens_threshold=(
            agent.config.session_auto_compaction_input_tokens_threshold
        ),
        compaction_keep_last_messages=agent.config.session_compaction_keep_last_messages,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
    )

    assert session.permission_mode == agent.active_permission_mode
    assert session.permission_prompting_enabled is agent.permission_policy.prompting_enabled
    assert session.permission_rule_counts == agent.permission_policy.rule_counts()
    assert session.prompt_format == "native"
    assert session.prompt_sections == ["Runtime Config", "Workflow Context"]


def test_build_runtime_system_message_updates_session_metadata(temp_dir: Path) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False),
        project_root=temp_dir,
    )
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )

    prompt_state = build_runtime_system_message(
        registry=agent.registry,
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
    assert native_examples[1].content.startswith("[write:")


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
