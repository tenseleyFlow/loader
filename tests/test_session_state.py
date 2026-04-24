"""Tests for persisted session state and resume support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader.agent.loop import Agent, AgentConfig, ReasoningConfig
from loader.llm.base import CompletionResponse, Message, Role, ToolCall
from loader.runtime.completion_trace import CompletionTraceEntry
from loader.runtime.evidence_provenance import EvidenceProvenance
from loader.runtime.prompt_history import PromptSnapshot
from loader.runtime.runtime_handle import RuntimeHandle
from loader.runtime.session import ConversationSession
from loader.runtime.workflow_ledger import WorkflowLedger, WorkflowLedgerItem
from loader.runtime.workflow_policy import WorkflowTimelineEntry
from tests.helpers.runtime_harness import ScriptedBackend


def _dummy_system() -> Message:
    return Message(role=Role.SYSTEM, content="system")


def _dummy_few_shots() -> list[Message]:
    return []


@pytest.mark.asyncio
async def test_session_persists_and_resumes_across_agent_restart(temp_dir: Path) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="I'll create the file.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(temp_dir / "hello.txt"),
                            "content": "hello\n",
                        },
                    )
                ],
                usage={"prompt_tokens": 12, "completion_tokens": 5},
            ),
            CompletionResponse(
                content="The file is written.",
                usage={"prompt_tokens": 10, "completion_tokens": 4},
            ),
        ]
    )
    config = AgentConfig(auto_context=False, stream=False)
    first_agent = Agent(backend=backend, config=config, project_root=temp_dir)

    response = await first_agent.run("Create hello.txt in the workspace root.")

    assert response.startswith("The file is written.")
    session_id = first_agent.session.session_id
    assert first_agent.session.storage_path.exists()

    resumed_agent = Agent(
        backend=ScriptedBackend(completions=[]),
        config=config,
        project_root=temp_dir,
    )

    assert resumed_agent.resume_session(session_id) is True
    assert resumed_agent.session.session_id == session_id
    assert resumed_agent._current_task == "Create hello.txt in the workspace root."
    assert resumed_agent.active_permission_mode == "workspace-write"
    assert resumed_agent.workflow_mode == first_agent.workflow_mode
    assert resumed_agent.last_turn_summary is not None
    assert resumed_agent.last_turn_summary.definition_of_done is not None
    assert resumed_agent.last_turn_summary.definition_of_done.task_statement == (
        "Create hello.txt in the workspace root."
    )
    assert any(
        message.role == Role.USER
        and message.content == "Create hello.txt in the workspace root."
        for message in resumed_agent.messages
    )


def test_agent_clear_history_rebuilds_a_fresh_runtime_session(temp_dir: Path) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    original_session_id = agent.session.session_id
    agent.current_task = "Keep runtime state tidy."
    agent.prompt_format = "native"
    agent.prompt_sections = ["Runtime Config", "Workflow Context"]
    agent.set_workflow_mode("clarify")
    agent.queue_steering_message("Stay in runtime.")

    agent.clear_history()

    assert agent.session.session_id != original_session_id
    assert agent.current_task is None
    assert agent.workflow_mode == "execute"
    assert agent.prompt_format is None
    assert agent.prompt_sections == []
    assert agent.messages == []
    assert agent.last_turn_summary is None
    assert agent.drain_steering_messages() == []


def test_session_rotation_kicks_in_at_size_cap(temp_dir: Path) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
        rotate_after_bytes=250,
    )

    for index in range(6):
        session.append(
            Message(
                role=Role.USER,
                content=f"Message {index}: " + ("x" * 120),
            )
        )

    assert session.storage_path.exists()
    assert session.storage_path.with_suffix(".1.json").exists()


def test_session_compaction_persists_summary_and_recent_messages(temp_dir: Path) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
        messages=[
            Message(role=Role.USER, content="Kick off runtime audit"),
            Message(role=Role.ASSISTANT, content="Initial findings"),
            Message(role=Role.USER, content="Focus on sessions"),
            Message(role=Role.ASSISTANT, content="Compaction design drafted"),
            Message(role=Role.USER, content="Preserve the latest four messages"),
            Message(role=Role.ASSISTANT, content="Ready to compact"),
        ],
        auto_compaction_input_tokens_threshold=1,
        compaction_keep_last_messages=4,
    )

    result = session.maybe_compact()

    assert result is not None
    assert session.compaction is not None
    assert session.storage_path.exists()
    assert session.messages[0].content.startswith("[COMPACTED CONTEXT]")
    assert [message.content for message in session.messages[-4:]] == [
        "Focus on sessions",
        "Compaction design drafted",
        "Preserve the latest four messages",
        "Ready to compact",
    ]


def test_build_request_messages_omits_large_mutation_tool_calls_from_history(
    temp_dir: Path,
) -> None:
    large_html = "<html>" + ("x" * 400) + "</html>"
    old_block = "old\n" * 120
    new_block = "new\n" * 120
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
        messages=[
            Message(role=Role.USER, content="Create the guide."),
            Message(
                role=Role.ASSISTANT,
                content="I'll write the first files now.",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="write",
                        arguments={
                            "file_path": str(temp_dir / "guides" / "nginx" / "index.html"),
                            "content": large_html,
                        },
                    ),
                    ToolCall(
                        id="edit-1",
                        name="edit",
                        arguments={
                            "file_path": str(temp_dir / "README.md"),
                            "old_string": old_block,
                            "new_string": new_block,
                        },
                    ),
                ],
            ),
        ],
    )

    request_messages = session.build_request_messages()

    assert request_messages[2].tool_calls == []
    assert request_messages[2].content == "I'll write the first files now."
    assert session.messages[1].tool_calls[0].arguments["content"] == large_html
    assert session.messages[1].tool_calls[1].arguments["old_string"] == old_block
    assert session.messages[1].tool_calls[1].arguments["new_string"] == new_block


def test_session_persists_permission_policy_metadata(temp_dir: Path) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
        permission_mode="prompt",
        permission_prompting_enabled=True,
        permission_rule_counts={"allow": 1, "deny": 2, "ask": 3},
        permission_rules_source=str(temp_dir / ".loader" / "permission-rules.json"),
        prompt_format="react",
        prompt_sections=["Runtime Config", "Workflow Context"],
    )

    session.update_runtime_state(
        current_task="Inspect permission history",
        runtime_owner_type="RuntimeHandle",
        permission_mode="allow",
        permission_prompting_enabled=True,
        permission_rule_counts={"allow": 2, "deny": 1, "ask": 4},
        permission_rules_source=str(temp_dir / ".loader" / "permission-rules.json"),
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context", "Project Context"],
        workflow_reason_code="task_is_complex",
        workflow_reason_summary="task looks complex enough to benefit from a persisted plan",
        workflow_decision_kind="initial_route",
        workflow_ambiguity_score=0.2,
        workflow_complexity_score=0.6,
        workflow_scheduled_next_mode="execute",
        last_completion_decision_code="verification_failed_reentry",
        last_completion_decision_summary=(
            "continued after verification failed and the runtime re-entered execute mode"
        ),
        last_turn_transition_summary="completion -> finalize [terminal] Finalizing completed turn",
        last_turn_transition_kind="terminal",
        last_turn_transition_reason_code="turn_complete",
    )
    session.append_workflow_timeline_entry(
        WorkflowTimelineEntry(
            timestamp="2026-04-07T12:00:00Z",
            kind="route",
            mode="plan",
            reason_code="task_is_complex",
            summary="plan: workflow pressure favors a persisted plan before execution",
            decision_kind="initial_route",
            route_score=0.72,
            runner_up_mode="clarify",
            runner_up_score=0.61,
            scheduled_next_mode="execute",
            unresolved_questions=["Scope is still broad."],
            prompt_format="native",
            prompt_sections=["Runtime Config", "Workflow Context", "Project Context"],
        )
    )
    session.append_completion_trace_entry(
        CompletionTraceEntry(
            stage="definition_of_done",
            outcome="continue",
            decision_code="verification_failed_reentry",
            decision_summary=(
                "continued after verification failed and the runtime "
                "re-entered execute mode"
            ),
            evidence_summary=["verification contradiction: pytest still failed"],
        )
    )

    reloaded = ConversationSession.load(
        project_root=temp_dir,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        session_id=session.session_id,
    )

    assert reloaded is not None
    assert reloaded.permission_mode == "allow"
    assert reloaded.permission_prompting_enabled is True
    assert reloaded.permission_rule_counts == {"allow": 2, "deny": 1, "ask": 4}
    assert reloaded.permission_rules_source == str(
        temp_dir / ".loader" / "permission-rules.json"
    )
    assert reloaded.runtime_owner_type == "RuntimeHandle"
    assert reloaded.runtime_owner_path == "runtime-handle"
    assert reloaded.prompt_format == "native"
    assert reloaded.prompt_sections == [
        "Runtime Config",
        "Workflow Context",
        "Project Context",
    ]
    assert reloaded.workflow_reason_code == "task_is_complex"
    assert reloaded.workflow_reason_summary == (
        "task looks complex enough to benefit from a persisted plan"
    )
    assert reloaded.workflow_decision_kind == "initial_route"
    assert reloaded.workflow_ambiguity_score == pytest.approx(0.2)
    assert reloaded.workflow_complexity_score == pytest.approx(0.6)
    assert reloaded.workflow_scheduled_next_mode == "execute"
    assert reloaded.last_completion_decision_code == "verification_failed_reentry"
    assert reloaded.last_completion_decision_summary == (
        "continued after verification failed and the runtime re-entered execute mode"
    )
    assert [entry.decision_code for entry in reloaded.completion_trace] == [
        "verification_failed_reentry"
    ]
    assert reloaded.completion_trace[0].evidence_summary == [
        "verification contradiction: pytest still failed"
    ]
    assert reloaded.last_turn_transition_summary == (
        "completion -> finalize [terminal] Finalizing completed turn"
    )
    assert reloaded.last_turn_transition_kind == "terminal"
    assert reloaded.last_turn_transition_reason_code == "turn_complete"
    assert len(reloaded.workflow_timeline) == 1
    assert reloaded.workflow_timeline[0].mode == "plan"
    assert reloaded.workflow_timeline[0].route_score == pytest.approx(0.72)
    assert reloaded.workflow_timeline[0].unresolved_questions == [
        "Scope is still broad."
    ]


def test_resume_session_updates_runtime_owner_metadata(temp_dir: Path) -> None:
    agent = Agent(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )
    agent.session.persist()
    session_id = agent.session.session_id

    handle = RuntimeHandle(
        backend=ScriptedBackend(),
        config=AgentConfig(auto_context=False, stream=False),
        project_root=temp_dir,
    )

    assert handle.resume_session(session_id) is True

    reloaded = ConversationSession.load(
        project_root=temp_dir,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        session_id=session_id,
    )

    assert reloaded is not None
    assert reloaded.runtime_owner_type == "RuntimeHandle"
    assert reloaded.runtime_owner_path == "runtime-handle"


def test_session_prefers_canonical_workflow_timeline_for_completion_trace(
    temp_dir: Path,
) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )

    session.update_runtime_state(
        current_task="Explain why the turn stopped",
        last_completion_decision_code="continuation_budget_exhausted",
        last_completion_decision_summary=(
            "stopped because the continuation budget was exhausted while "
            "follow-through evidence was still missing"
        ),
    )
    session.append_completion_trace_entry(
        CompletionTraceEntry(
            stage="definition_of_done",
            outcome="complete",
            decision_code="stale_completion_trace",
            decision_summary="this legacy trace entry should be ignored",
        )
    )
    session.append_workflow_timeline_entry(
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:00:00Z",
            kind="completion_check",
            mode="execute",
            reason_code="premature_completion_nudge",
            summary=(
                "completion: requested one continuation because the non-mutating "
                "response looked incomplete"
            ),
            decision_kind="forced",
            policy_stage="continuation_check",
            policy_outcome="continue",
            evidence_summary=["showing the requested work was actually carried out"],
        )
    )
    session.append_workflow_timeline_entry(
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:01:00Z",
            kind="completion_finalize",
            mode="execute",
            reason_code="continuation_budget_exhausted",
            summary=(
                "completion: stopped because the continuation budget was exhausted "
                "while follow-through evidence was still missing"
            ),
            decision_kind="forced",
            policy_stage="continuation_check",
            policy_outcome="finalize",
            evidence_summary=["showing the requested work was actually carried out"],
        )
    )

    persisted = json.loads(session.storage_path.read_text())
    assert "completion_trace" not in persisted

    reloaded = ConversationSession.load(
        project_root=temp_dir,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        session_id=session.session_id,
    )

    assert reloaded is not None
    assert [entry.decision_code for entry in reloaded.completion_trace] == [
        "premature_completion_nudge",
        "continuation_budget_exhausted",
    ]
    assert reloaded.completion_trace[-1].stage == "continuation_check"
    assert reloaded.completion_trace[-1].outcome == "finalize"
    assert reloaded.completion_trace[-1].evidence_summary == [
        "showing the requested work was actually carried out"
    ]


def test_session_projects_live_completion_trace_from_workflow_timeline(
    temp_dir: Path,
) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )

    session.append_workflow_timeline_entry(
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:00:00Z",
            kind="completion_check",
            mode="execute",
            reason_code="completion_response_accepted",
            summary="completion: accepted the response because follow-through evidence was present",
            decision_kind="forced",
            policy_stage="continuation_check",
            policy_outcome="accept",
        )
    )
    session.append_workflow_timeline_entry(
        WorkflowTimelineEntry(
            timestamp="2026-04-09T12:01:00Z",
            kind="completion_finalize",
            mode="execute",
            reason_code="continuation_budget_exhausted",
            summary="completion: stopped because verification evidence was still missing",
            decision_kind="forced",
            policy_stage="continuation_check",
            policy_outcome="finalize",
            evidence_summary=["a passing verification result from `pytest -q`"],
            evidence_provenance=[
                EvidenceProvenance(
                    category="verification",
                    source="dod.verification_commands",
                    summary="verification evidence was still missing for `pytest -q`",
                    status="missing",
                    subject="pytest -q",
                )
            ],
        )
    )
    session.update_runtime_state(
        last_completion_decision_code="continuation_budget_exhausted",
        last_completion_decision_summary=(
            "stopped because verification evidence was still missing"
        ),
    )

    assert [entry.decision_code for entry in session.completion_trace] == [
        "completion_response_accepted",
        "continuation_budget_exhausted",
    ]
    assert session.completion_trace[-1].stage == "continuation_check"
    assert session.completion_trace[-1].outcome == "finalize"
    assert session.completion_trace[-1].evidence_summary == [
        "a passing verification result from `pytest -q`"
    ]
    assert [item.summary for item in session.completion_trace[-1].evidence_provenance] == [
        "verification evidence was still missing for `pytest -q`"
    ]


def test_session_persists_workflow_ledger_state(temp_dir: Path) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )

    session.update_workflow_ledger(
        WorkflowLedger(
            assumptions=[
                WorkflowLedgerItem(
                    text="notes.txt stays out of scope unless clarified otherwise.",
                    status="contradicted",
                    introduced_phase="clarify",
                    updated_phase="recovery",
                    evidence=["Clarify scope assumed `notes.txt` stayed out of scope."],
                )
            ],
            acceptance_anchors=[
                WorkflowLedgerItem(
                    text="notes.txt exists in the workspace root.",
                    status="changed",
                    introduced_phase="clarify",
                    updated_phase="recovery",
                )
            ],
            decision_boundaries=[
                WorkflowLedgerItem(
                    text="Escalate before broad UX changes.",
                    status="tracked",
                    introduced_phase="clarify",
                )
            ],
        )
    )

    reloaded = ConversationSession.load(
        project_root=temp_dir,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        session_id=session.session_id,
    )

    assert reloaded is not None
    assert reloaded.workflow_ledger.assumptions[0].status == "contradicted"
    assert reloaded.workflow_ledger.assumptions[0].updated_phase == "recovery"
    assert reloaded.workflow_ledger.acceptance_anchors[0].status == "changed"
    assert reloaded.workflow_ledger.decision_boundaries[0].text == (
        "Escalate before broad UX changes."
    )


def test_session_persists_prompt_history_state(temp_dir: Path) -> None:
    session = ConversationSession(
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        project_root=temp_dir,
    )

    session.append_prompt_snapshot(
        PromptSnapshot(
            timestamp="2026-04-07T14:00:00Z",
            workflow_mode="plan",
            permission_mode="prompt",
            current_task="Tighten Loader workflow behavior",
            prompt_format="native",
            prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            content="# Introduction\nplan around planned.txt\n",
        )
    )
    session.append_prompt_snapshot(
        PromptSnapshot(
            timestamp="2026-04-07T14:02:00Z",
            workflow_mode="execute",
            permission_mode="prompt",
            current_task="Tighten Loader workflow behavior",
            prompt_format="native",
            prompt_sections=[
                "Runtime Config",
                "Workflow Context",
                "Mode Guidance",
                "Project Context",
            ],
            content="# Introduction\nexecute around notes.txt\n# Project Context\npython\n",
        )
    )

    reloaded = ConversationSession.load(
        project_root=temp_dir,
        system_message_factory=_dummy_system,
        few_shot_factory=_dummy_few_shots,
        session_id=session.session_id,
    )

    assert reloaded is not None
    assert len(reloaded.prompt_history) == 2
    assert reloaded.prompt_history[0].workflow_mode == "plan"
    assert reloaded.prompt_history[-1].workflow_mode == "execute"
    assert "notes.txt" in reloaded.prompt_history[-1].content


@pytest.mark.asyncio
async def test_turn_summary_usage_rolls_up_into_session_totals(temp_dir: Path) -> None:
    backend = ScriptedBackend(
        completions=[
            CompletionResponse(
                content="Here's the answer.",
                usage={"prompt_tokens": 9, "completion_tokens": 3},
            )
        ]
    )
    agent = Agent(
        backend=backend,
        config=AgentConfig(
            auto_context=False,
            stream=False,
            reasoning=ReasoningConfig(completion_check=False),
        ),
        project_root=temp_dir,
    )

    await agent.run("Write a short release-note style summary of what Loader does well.")

    assert agent.last_turn_summary is not None
    assert agent.last_turn_summary.usage["input_tokens"] == 9
    assert agent.last_turn_summary.usage["output_tokens"] == 3
    assert agent.last_turn_summary.cumulative_usage["input_tokens"] == 9
    assert agent.last_turn_summary.cumulative_usage["output_tokens"] == 3
    assert agent.last_turn_summary.cumulative_usage["turns"] == 1
