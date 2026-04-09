"""Tests for doctor, status, and session inspection surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

import loader.cli.main as cli_main_module
from loader.llm.base import Message, Role
from loader.runtime.completion_trace import CompletionTraceEntry
from loader.runtime.dod import (
    DefinitionOfDoneStore,
    VerificationEvidence,
    create_definition_of_done,
)
from loader.runtime.evidence_provenance import EvidenceProvenance
from loader.runtime.explore_state import ExploreSnapshot, ExploreStateStore
from loader.runtime.inspection import (
    CheckStatus,
    collect_doctor_report,
    collect_permission_snapshot,
    collect_prompt_diff,
    collect_prompt_preview,
    collect_status_snapshot,
    collect_workflow_artifact_diffs,
    collect_workflow_timeline,
    dry_run_permission_check,
    list_session_summaries,
    load_session_detail,
)
from loader.runtime.prompt_history import PromptSnapshot
from loader.runtime.session import SessionSnapshot, SessionStore
from loader.runtime.verification_observations import VerificationObservation
from loader.runtime.workflow_ledger import WorkflowLedger, WorkflowLedgerItem
from loader.runtime.workflow_policy import WorkflowTimelineEntry


class FakeOllamaBackend:
    """Small async backend stub for doctor tests."""

    def __init__(
        self,
        *,
        model: str,
        health: bool,
        models: list[dict[str, object]],
        model_details: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self._health = health
        self._models = models
        self._model_details = model_details

    async def list_models(self) -> list[dict[str, object]]:
        return list(self._models)

    async def health_check(self) -> bool:
        return self._health

    async def describe_model(self) -> dict[str, object] | None:
        return self._model_details

    async def close(self) -> None:
        return None


def _write_python_workspace(temp_dir: Path) -> None:
    (temp_dir / "pyproject.toml").write_text(
        "\n".join(
            [
                "[build-system]",
                'requires = ["hatchling"]',
                'build-backend = "hatchling.build"',
                "",
                "[tool.pytest.ini_options]",
                'testpaths = ["tests"]',
                "",
            ]
        )
        + "\n"
    )
    (temp_dir / "src").mkdir()
    (temp_dir / "tests").mkdir()


def _ensure_loader_dirs(temp_dir: Path) -> None:
    loader_root = temp_dir / ".loader"
    for name in ("sessions", "state", "dod", "briefs", "plans"):
        (loader_root / name).mkdir(parents=True, exist_ok=True)
    (loader_root / "project-memory.json").write_text("{}\n")


def _persist_session_with_dod(temp_dir: Path) -> tuple[str, str]:
    dod = create_definition_of_done("Fix the failing tests")
    dod.status = "fixing"
    dod.pending_items = ["Re-run pytest"]
    dod.completed_items = ["Patch the broken parser"]
    dod.last_verification_result = "failed"
    dod.evidence = [
        VerificationEvidence(
            command="pytest -q",
            passed=False,
            stderr="1 failed",
            kind="test",
        )
    ]
    dod_path = DefinitionOfDoneStore(temp_dir).save(dod)
    workflow_timeline = [
        WorkflowTimelineEntry(
            timestamp="2026-04-06T12:04:00Z",
            kind="handoff",
            mode="verify",
            reason_code="execute_completed",
            summary="verify: execution completed; verifying the parser fix",
            decision_kind="handoff",
            prompt_format="native",
            prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            artifact_paths=[str(temp_dir / ".loader" / "plans" / "fix-tests.md")],
        ),
        WorkflowTimelineEntry(
            timestamp="2026-04-06T12:05:00Z",
            kind="reentry",
            mode="execute",
            reason_code="verification_failed_reentry",
            summary="execute: verification failed; returning to execute for fixes",
            decision_kind="reentry",
            scheduled_next_mode="verify",
            runner_up_mode="verify",
            runner_up_score=0.52,
            verification_observations=[
                VerificationObservation(
                    status="failed",
                    summary="verification failed for `pytest -q`",
                    command="pytest -q",
                    kind="test",
                    detail="1 failed",
                )
            ],
            prompt_format="native",
            prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            artifact_paths=[str(temp_dir / ".loader" / "plans" / "fix-tests.md")],
        ),
    ]

    snapshot = SessionSnapshot(
        session_id="20260406T120000Z-abcdef01",
        created_at="2026-04-06T12:00:00Z",
        updated_at="2026-04-06T12:05:00Z",
        messages=[
            Message(role=Role.USER, content="Fix the failing tests"),
            Message(role=Role.ASSISTANT, content="I updated the parser."),
        ],
        usage={"turns": 1, "tool_calls": 2},
        active_dod_path=str(dod_path),
        current_task="Fix the failing tests",
        runtime_owner_type="RuntimeHandle",
        runtime_owner_path="runtime-handle",
        workflow_mode="execute",
        permission_mode="prompt",
        permission_prompting_enabled=True,
        permission_rule_counts={"allow": 1, "deny": 2, "ask": 1},
        permission_rules_source=str(temp_dir / ".loader" / "permission-rules.json"),
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
        prompt_history=[
            PromptSnapshot(
                timestamp="2026-04-06T12:04:00Z",
                workflow_mode="verify",
                permission_mode="prompt",
                current_task="Fix the failing tests",
                prompt_format="native",
                prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
                content="# Introduction\nverify parser fix\n",
            ),
            PromptSnapshot(
                timestamp="2026-04-06T12:05:00Z",
                workflow_mode="execute",
                permission_mode="prompt",
                current_task="Fix the failing tests",
                prompt_format="native",
                prompt_sections=[
                    "Runtime Config",
                    "Workflow Context",
                    "Mode Guidance",
                    "Project Context",
                ],
                content="# Introduction\nexecute parser fix\n# Project Context\npython\n",
            ),
        ],
        workflow_reason_code="verification_failed_reentry",
        workflow_reason_summary="verification failed; returning to execute for fixes",
        workflow_decision_kind="reentry",
        workflow_ambiguity_score=0.1,
        workflow_complexity_score=0.7,
        workflow_scheduled_next_mode="verify",
        active_turn_phase="completion",
        last_completion_decision_code="verification_failed_reentry",
        last_completion_decision_summary=(
            "continued after verification failed and the runtime re-entered execute mode"
        ),
        completion_trace=[
            CompletionTraceEntry(
                stage="continuation_check",
                outcome="accept",
                decision_code="completion_response_accepted",
                decision_summary="accepted the response because completion heuristics found no missing follow-through",
            ),
            CompletionTraceEntry(
                stage="definition_of_done",
                outcome="continue",
                decision_code="verification_failed_reentry",
                decision_summary="continued after verification failed and the runtime re-entered execute mode",
            ),
        ],
        last_turn_transition_summary="completion -> finalize [terminal] Finalizing completed turn",
        last_turn_transition_kind="terminal",
        last_turn_transition_reason_code="turn_complete",
        workflow_timeline=workflow_timeline,
    )
    SessionStore(temp_dir).save(snapshot)
    return snapshot.session_id, str(dod_path)


def _persist_explore_snapshot(temp_dir: Path) -> None:
    ExploreStateStore(temp_dir).save(
        ExploreSnapshot(
            turn_count=2,
            model_name="llama3.1:8b",
            messages=[
                Message(role=Role.USER, content="Where should I start?"),
                Message(role=Role.ASSISTANT, content="Start with README.md."),
                Message(role=Role.USER, content="What file did you mention?"),
                Message(role=Role.ASSISTANT, content="I mentioned README.md."),
            ],
            last_history_mode="continue",
            last_query="What file did you mention?",
            last_response="I mentioned README.md.",
        )
    )


def _persist_session_with_rich_workflow(temp_dir: Path) -> str:
    slug = "tighten-loader-workflow-behavior"
    brief_old = temp_dir / ".loader" / "briefs" / f"20260406T150000Z-{slug}.md"
    brief_new = temp_dir / ".loader" / "briefs" / f"20260406T150200Z-{slug}.md"
    brief_old.write_text(
        "# Task Brief\n\n## Likely Touchpoints\n- planned.txt\n\n## Acceptance Criteria\n- planned.txt exists.\n"
    )
    brief_new.write_text(
        "# Task Brief\n\n## Likely Touchpoints\n- notes.txt\n\n## Acceptance Criteria\n- notes.txt exists.\n"
    )
    plan_old_root = temp_dir / ".loader" / "plans" / f"20260406T150100Z-{slug}"
    plan_new_root = temp_dir / ".loader" / "plans" / f"20260406T150300Z-{slug}"
    plan_old_root.mkdir(parents=True, exist_ok=True)
    plan_new_root.mkdir(parents=True, exist_ok=True)
    (plan_old_root / "implementation.md").write_text(
        "# Implementation Plan\n\n## File Changes\n- Create planned.txt.\n"
    )
    (plan_old_root / "verification.md").write_text(
        "# Verification Plan\n\n## Acceptance Criteria\n- planned.txt exists.\n"
    )
    (plan_new_root / "implementation.md").write_text(
        "# Implementation Plan\n\n## File Changes\n- Keep notes.txt as the runtime artifact.\n"
    )
    (plan_new_root / "verification.md").write_text(
        "# Verification Plan\n\n## Acceptance Criteria\n- notes.txt exists.\n"
    )

    dod = create_definition_of_done("Tighten Loader workflow behavior")
    dod.status = "fixing"
    dod.clarify_brief = str(brief_new)
    dod.implementation_plan = str(plan_new_root / "implementation.md")
    dod.verification_plan = str(plan_new_root / "verification.md")
    dod.acceptance_criteria = ["notes.txt exists in the workspace root."]
    dod_path = DefinitionOfDoneStore(temp_dir).save(dod)

    snapshot = SessionSnapshot(
        session_id="20260406T150000Z-feedface",
        created_at="2026-04-06T15:00:00Z",
        updated_at="2026-04-06T15:04:00Z",
        messages=[
            Message(role=Role.USER, content="Tighten Loader workflow behavior"),
            Message(role=Role.ASSISTANT, content="I refreshed the workflow contract."),
        ],
        active_dod_path=str(dod_path),
        current_task="Tighten Loader workflow behavior",
        runtime_owner_type="RuntimeHandle",
        runtime_owner_path="runtime-handle",
        workflow_mode="execute",
        permission_mode="prompt",
        permission_prompting_enabled=True,
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
        prompt_history=[
            PromptSnapshot(
                timestamp="2026-04-06T15:02:00Z",
                workflow_mode="plan",
                permission_mode="prompt",
                current_task="Tighten Loader workflow behavior",
                prompt_format="native",
                prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
                content="# Introduction\nplan around planned.txt\n",
            ),
            PromptSnapshot(
                timestamp="2026-04-06T15:04:00Z",
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
            ),
        ],
        workflow_reason_code="full_replan_completed",
        workflow_reason_summary="clarify and plan artifacts refreshed; returning to execute",
        workflow_decision_kind="handoff",
        workflow_timeline=[
            WorkflowTimelineEntry(
                timestamp="2026-04-06T15:01:00Z",
                kind="clarify_continue",
                mode="clarify",
                reason_code="clarify_pressure_pass_required",
                summary="clarify: Loader still needs a tradeoff pass around non-goals",
                decision_kind="forced",
                unresolved_questions=["Concrete files or subsystems are still not pinned down."],
                signal_summary=["ambiguity=0.82", "open_questions=1"],
                clarify_stage="readiness",
                clarify_pressure_kind="tradeoff",
                pressure_pass_complete=False,
                missing_readiness_gates=["non_goals", "decision_boundaries"],
            ),
            WorkflowTimelineEntry(
                timestamp="2026-04-06T15:02:00Z",
                kind="reentry",
                mode="plan",
                reason_code="full_replan_required",
                summary="plan: clarify and plan artifacts drifted; rebuilding the plan",
                decision_kind="reentry",
                scheduled_next_mode="execute",
                unresolved_questions=["Touched files outside the current plan: notes.txt"],
                evidence_summary=[
                    "confirmed touchpoint: `notes.txt` was already touched during execution.",
                    (
                        "verification contradiction: Failed verification exposed "
                        "missing brief coverage for `notes.txt exists`."
                    ),
                ],
                signal_summary=["recent_reentry=1", "stale_plan=true"],
                artifact_paths=[
                    str(brief_new),
                    str(plan_new_root / "implementation.md"),
                    str(plan_new_root / "verification.md"),
                ],
            ),
            WorkflowTimelineEntry(
                timestamp="2026-04-06T15:03:00Z",
                kind="verify_skip",
                mode="verify",
                reason_code="verify_skip_no_commands",
                summary="verify: no verification commands were available for this turn",
                decision_kind="forced",
                signal_summary=["verify_pressure=low"],
            ),
        ],
        workflow_ledger=WorkflowLedger(
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
                    evidence=[
                        (
                            "Failed verification exposed missing brief coverage for "
                            "`notes.txt exists`."
                        )
                    ],
                )
            ],
            decision_boundaries=[
                WorkflowLedgerItem(
                    text="Escalate before broad UX changes.",
                    status="reopened",
                    introduced_phase="clarify",
                    updated_phase="recovery",
                    evidence=["The active task framing outgrew the persisted clarify brief."],
                )
            ],
        ),
    )
    SessionStore(temp_dir).save(snapshot)
    return snapshot.session_id


def _persist_session_with_policy_accountability(temp_dir: Path) -> str:
    snapshot = SessionSnapshot(
        session_id="20260406T160000Z-abcd1234",
        created_at="2026-04-06T16:00:00Z",
        updated_at="2026-04-06T16:03:00Z",
        messages=[
            Message(role=Role.USER, content="Explain Loader policy accountability"),
            Message(role=Role.ASSISTANT, content="The runtime tracked repair and completion decisions."),
        ],
        current_task="Explain Loader policy accountability",
        runtime_owner_type="RuntimeHandle",
        runtime_owner_path="runtime-handle",
        workflow_mode="execute",
        permission_mode="workspace-write",
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
        workflow_timeline=[
            WorkflowTimelineEntry(
                timestamp="2026-04-06T16:01:00Z",
                kind="repair_retry",
                mode="execute",
                reason_code="raw_text_tool_recovered",
                summary="repair: recovered raw-text tool calls into executable tool invocations",
                decision_kind="forced",
                policy_stage="raw_text_tool_fallback",
                policy_outcome="retry",
                prompt_format="native",
                prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            ),
            WorkflowTimelineEntry(
                timestamp="2026-04-06T16:02:00Z",
                kind="completion_check",
                mode="execute",
                reason_code="completion_response_accepted",
                summary="completion: accepted the response because completion heuristics found no missing follow-through",
                decision_kind="forced",
                policy_stage="continuation_check",
                policy_outcome="accept",
                prompt_format="native",
                prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            ),
            WorkflowTimelineEntry(
                timestamp="2026-04-06T16:03:00Z",
                kind="completion_continue",
                mode="execute",
                reason_code="verification_failed_reentry",
                summary="completion: continued after verification failed and the runtime re-entered execute mode",
                decision_kind="forced",
                policy_stage="definition_of_done",
                policy_outcome="continue",
                evidence_provenance=[
                    EvidenceProvenance(
                        category="verification",
                        source="dod.evidence",
                        summary="verification failed for `pytest -q`",
                        status="contradicts",
                        subject="pytest -q",
                    )
                ],
                verification_observations=[
                    VerificationObservation(
                        status="failed",
                        summary="verification failed for `pytest -q`",
                        command="pytest -q",
                        kind="test",
                        detail="1 failed",
                    )
                ],
                prompt_format="native",
                prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            ),
        ],
    )
    SessionStore(temp_dir).save(snapshot)
    return snapshot.session_id


def _persist_session_with_pending_verification(temp_dir: Path) -> str:
    snapshot = SessionSnapshot(
        session_id="20260406T160500Z-pending123",
        created_at="2026-04-06T16:05:00Z",
        updated_at="2026-04-06T16:05:30Z",
        messages=[
            Message(role=Role.USER, content="Verify the runtime changes"),
            Message(role=Role.ASSISTANT, content="Entering verification."),
        ],
        current_task="Verify the runtime changes",
        runtime_owner_type="RuntimeHandle",
        runtime_owner_path="runtime-handle",
        workflow_mode="verify",
        permission_mode="workspace-write",
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
        workflow_timeline=[
            WorkflowTimelineEntry(
                timestamp="2026-04-06T16:05:30Z",
                kind="verify_observation",
                mode="verify",
                reason_code="verification_pending",
                summary="verify: verification is pending for the active command set",
                decision_kind="forced",
                policy_stage="verification",
                policy_outcome="pending",
                verification_observations=[
                    VerificationObservation(
                        status="pending",
                        summary="verification pending for `uv run pytest -q`",
                        command="uv run pytest -q",
                        kind="test",
                    )
                ],
                prompt_format="native",
                prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            )
        ],
    )
    SessionStore(temp_dir).save(snapshot)
    return snapshot.session_id


def _persist_session_with_stale_verification(temp_dir: Path) -> str:
    snapshot = SessionSnapshot(
        session_id="20260406T160700Z-stale1234",
        created_at="2026-04-06T16:07:00Z",
        updated_at="2026-04-06T16:07:30Z",
        messages=[
            Message(role=Role.USER, content="Keep working on the runtime"),
            Message(role=Role.ASSISTANT, content="Fresh verification is required again."),
        ],
        current_task="Keep working on the runtime",
        runtime_owner_type="RuntimeHandle",
        runtime_owner_path="runtime-handle",
        workflow_mode="execute",
        permission_mode="workspace-write",
        prompt_format="native",
        prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
        workflow_timeline=[
            WorkflowTimelineEntry(
                timestamp="2026-04-06T16:07:30Z",
                kind="verify_observation",
                mode="execute",
                reason_code="verification_stale",
                summary="verify: previous verification became stale after new mutating work",
                decision_kind="forced",
                policy_stage="verification",
                policy_outcome="stale",
                verification_observations=[
                    VerificationObservation(
                        status="stale",
                        summary=(
                            "verification became stale for `uv run pytest -q` "
                            "after new mutating work"
                        ),
                        command="uv run pytest -q",
                        kind="runtime",
                        detail="write changed src/loader/runtime/finalization.py",
                    )
                ],
                prompt_format="native",
                prompt_sections=["Runtime Config", "Workflow Context", "Mode Guidance"],
            )
        ],
    )
    SessionStore(temp_dir).save(snapshot)
    return snapshot.session_id


@pytest.mark.asyncio
async def test_collect_doctor_report_passes_for_healthy_workspace(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)

    report = await collect_doctor_report(
        temp_dir,
        model="qwen2.5-coder:14b",
        backend_factory=lambda model: FakeOllamaBackend(
            model=model,
            health=True,
            models=[{"name": "qwen2.5-coder:14b"}],
            model_details={"details": {"family": "qwen2.5"}},
        ),
    )

    assert report.overall_status == CheckStatus.PASS
    assert {check.name for check in report.checks} == {
        "backend",
        "capabilities",
        "workspace",
        "write_access",
        "commands",
        "state",
        "permissions",
    }
    backend_check = next(check for check in report.checks if check.name == "backend")
    state_check = next(check for check in report.checks if check.name == "state")

    assert backend_check.status == CheckStatus.PASS
    assert state_check.status == CheckStatus.PASS


@pytest.mark.asyncio
async def test_collect_doctor_report_surfaces_backend_and_state_failures(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    (temp_dir / ".loader").mkdir()
    (temp_dir / ".loader" / "project-memory.json").write_text("{broken json")

    report = await collect_doctor_report(
        temp_dir,
        model="missing-model:latest",
        backend_factory=lambda model: FakeOllamaBackend(
            model=model,
            health=False,
            models=[{"name": "llama3.1:8b"}],
            model_details=None,
        ),
    )

    backend_check = next(check for check in report.checks if check.name == "backend")
    state_check = next(check for check in report.checks if check.name == "state")

    assert report.overall_status == CheckStatus.FAIL
    assert backend_check.status == CheckStatus.FAIL
    assert "not pulled" in backend_check.message
    assert state_check.status == CheckStatus.FAIL
    assert "corrupted" in state_check.message


@pytest.mark.asyncio
async def test_collect_doctor_report_fails_closed_on_invalid_permission_rules(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    (temp_dir / ".loader" / "permission-rules.json").write_text('{"allow": "nope"}\n')

    report = await collect_doctor_report(
        temp_dir,
        model="qwen2.5-coder:14b",
        permission_mode="prompt",
        backend_factory=lambda model: FakeOllamaBackend(
            model=model,
            health=True,
            models=[{"name": "qwen2.5-coder:14b"}],
        ),
    )

    permission_check = next(check for check in report.checks if check.name == "permissions")
    assert report.overall_status == CheckStatus.FAIL
    assert permission_check.status == CheckStatus.FAIL
    assert report.permission_rules_valid is False
    assert "invalid" in permission_check.message.lower()


def test_status_and_session_surfaces_reflect_persisted_state(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id, dod_path = _persist_session_with_dod(temp_dir)
    _persist_explore_snapshot(temp_dir)

    snapshot = collect_status_snapshot(
        temp_dir,
        model="llama3.1:8b",
    )
    sessions = list_session_summaries(temp_dir)
    detail = load_session_detail(session_id, project_root=temp_dir)

    assert snapshot.active_session_id == session_id
    assert snapshot.dod_status == "fixing"
    assert snapshot.dod_pending_items_count == 1
    assert snapshot.last_verification_result == "failed"
    assert snapshot.active_dod_path == dod_path
    assert snapshot.permission_mode == "prompt"
    assert snapshot.runtime_owner_type == "RuntimeHandle"
    assert snapshot.runtime_owner_path == "runtime-handle"
    assert snapshot.permission_rule_counts == {"allow": 1, "deny": 2, "ask": 1}
    assert snapshot.permission_prompting_enabled is True
    assert snapshot.permission_rules_valid is True
    assert snapshot.permission_rules_source == str(
        temp_dir / ".loader" / "permission-rules.json"
    )
    assert snapshot.prompt_format == "native"
    assert snapshot.prompt_sections == [
        "Runtime Config",
        "Workflow Context",
        "Mode Guidance",
    ]
    assert snapshot.workflow_reason_code == "verification_failed_reentry"
    assert snapshot.workflow_reason_summary == (
        "verification failed; returning to execute for fixes"
    )
    assert snapshot.workflow_decision_kind == "reentry"
    assert snapshot.workflow_scheduled_next_mode == "verify"
    assert snapshot.active_turn_phase == "completion"
    assert snapshot.completion_decision_code == "verification_failed_reentry"
    assert snapshot.completion_decision_summary == (
        "continued after verification failed and the runtime re-entered execute mode"
    )
    assert snapshot.last_turn_transition_summary == (
        "completion -> finalize [terminal] Finalizing completed turn"
    )
    assert snapshot.explore_turn_count == 2
    assert snapshot.explore_message_count == 4
    assert snapshot.explore_history_mode == "continue"
    assert snapshot.explore_last_query == "What file did you mention?"
    assert snapshot.explore_last_response == "I mentioned README.md."
    assert snapshot.explore_updated_at is not None
    assert [item.status for item in snapshot.recent_verification] == ["failed"]
    assert [item.command for item in snapshot.recent_verification] == ["pytest -q"]
    assert [item.detail for item in snapshot.recent_verification] == ["1 failed"]

    assert len(sessions) == 1
    assert sessions[0].session_id == session_id
    assert sessions[0].is_current is True
    assert sessions[0].runtime_owner_type == "RuntimeHandle"
    assert sessions[0].runtime_owner_path == "runtime-handle"
    assert sessions[0].dod_status == "fixing"
    assert sessions[0].permission_prompting_enabled is True
    assert sessions[0].permission_rule_counts == {"allow": 1, "deny": 2, "ask": 1}
    assert sessions[0].permission_rules_source == str(
        temp_dir / ".loader" / "permission-rules.json"
    )
    assert sessions[0].prompt_format == "native"
    assert sessions[0].workflow_reason_code == "verification_failed_reentry"
    assert sessions[0].workflow_reason_summary == (
        "verification failed; returning to execute for fixes"
    )
    assert sessions[0].workflow_decision_kind == "reentry"
    assert sessions[0].completion_decision_code == "verification_failed_reentry"
    assert sessions[0].completion_decision_summary == (
        "continued after verification failed and the runtime re-entered execute mode"
    )
    assert sessions[0].last_turn_transition_summary == (
        "completion -> finalize [terminal] Finalizing completed turn"
    )

    assert detail.snapshot.session_id == session_id
    assert detail.is_current is True
    assert detail.snapshot.runtime_owner_type == "RuntimeHandle"
    assert detail.snapshot.runtime_owner_path == "runtime-handle"
    assert detail.definition_of_done is not None
    assert detail.definition_of_done.status == "fixing"
    assert detail.snapshot.permission_rules_source == str(
        temp_dir / ".loader" / "permission-rules.json"
    )
    assert detail.snapshot.workflow_reason_code == "verification_failed_reentry"
    assert detail.snapshot.last_completion_decision_code == (
        "verification_failed_reentry"
    )
    assert [entry.decision_code for entry in detail.snapshot.completion_trace] == [
        "completion_response_accepted",
        "verification_failed_reentry",
    ]
    assert [item.status for item in detail.recent_verification] == ["failed"]
    assert [item.command for item in detail.recent_verification] == ["pytest -q"]
    assert detail.snapshot.last_turn_transition_reason_code == "turn_complete"
    assert len(detail.snapshot.workflow_timeline) == 2
    assert detail.snapshot.workflow_timeline[-1].scheduled_next_mode == "verify"


def test_collect_workflow_timeline_reflects_persisted_history(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id, _ = _persist_session_with_dod(temp_dir)

    snapshot = collect_workflow_timeline(project_root=temp_dir)

    assert snapshot.session_id == session_id
    assert snapshot.is_current is True
    assert snapshot.runtime_owner_type == "RuntimeHandle"
    assert snapshot.runtime_owner_path == "runtime-handle"
    assert snapshot.workflow_mode == "execute"
    assert snapshot.current_task == "Fix the failing tests"
    assert snapshot.total_entries == 2
    assert [entry.kind for entry in snapshot.entries] == ["handoff", "reentry"]
    assert snapshot.entries[-1].reason_code == "verification_failed_reentry"


def test_collect_workflow_timeline_supports_filters_and_highlights(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_rich_workflow(temp_dir)

    snapshot = collect_workflow_timeline(
        project_root=temp_dir,
        mode="clarify",
        limit=1,
    )

    assert snapshot.session_id == session_id
    assert snapshot.total_entries == 3
    assert snapshot.selected_mode == "clarify"
    assert snapshot.selected_kind is None
    assert snapshot.entry_limit == 1
    assert len(snapshot.entries) == 1
    assert snapshot.entries[0].kind == "clarify_continue"
    assert snapshot.entries[0].clarify_stage == "readiness"
    assert snapshot.entries[0].clarify_pressure_kind == "tradeoff"
    assert snapshot.entries[0].missing_readiness_gates == [
        "non_goals",
        "decision_boundaries",
    ]
    assert any(item.startswith("Asked again:") for item in snapshot.highlights)
    assert snapshot.workflow_ledger.assumptions[0].status == "contradicted"
    assert any(
        item.startswith("Contradicted assumptions:")
        for item in snapshot.highlights
    )


def test_collect_workflow_timeline_highlights_policy_accountability(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_policy_accountability(temp_dir)

    snapshot = collect_workflow_timeline(project_root=temp_dir)

    assert snapshot.session_id == session_id
    assert [entry.kind for entry in snapshot.entries] == [
        "repair_retry",
        "completion_check",
        "completion_continue",
    ]
    assert any(item.startswith("Repair path:") for item in snapshot.highlights)
    assert any(item.startswith("Completion decision:") for item in snapshot.highlights)
    assert any(
        "policy-stage=definition_of_done" in item for item in snapshot.highlights
    )


def test_collect_status_snapshot_includes_latest_policy_summary(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_session_with_policy_accountability(temp_dir)

    snapshot = collect_status_snapshot(temp_dir)

    assert snapshot.latest_policy_summary is not None
    assert "verification_failed_reentry" in snapshot.latest_policy_summary
    assert "observed=verification failed for `pytest -q` [1 failed]" in (
        snapshot.latest_policy_summary
    )
    assert "policy-stage=definition_of_done" in snapshot.latest_policy_summary
    assert snapshot.latest_policy_blocking_evidence == [
        "verification failed for `pytest -q`"
    ]
    assert snapshot.latest_policy_observed_verification == [
        "verification failed for `pytest -q` [1 failed]"
    ]
    assert [item.status for item in snapshot.recent_verification] == ["failed"]
    assert [item.command for item in snapshot.recent_verification] == ["pytest -q"]
    assert [item.detail for item in snapshot.recent_verification] == ["1 failed"]


def test_collect_status_snapshot_surfaces_pending_verification(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_session_with_pending_verification(temp_dir)

    snapshot = collect_status_snapshot(temp_dir)

    assert snapshot.latest_policy_summary is not None
    assert "verification_pending" in snapshot.latest_policy_summary
    assert "policy-outcome=pending" in snapshot.latest_policy_summary
    assert snapshot.latest_policy_observed_verification == [
        "verification pending for `uv run pytest -q`"
    ]
    assert [item.status for item in snapshot.recent_verification] == ["pending"]
    assert [item.command for item in snapshot.recent_verification] == [
        "uv run pytest -q"
    ]


def test_collect_status_snapshot_surfaces_stale_verification(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_session_with_stale_verification(temp_dir)

    snapshot = collect_status_snapshot(temp_dir)

    assert snapshot.latest_policy_summary is not None
    assert "verification_stale" in snapshot.latest_policy_summary
    assert "policy-outcome=stale" in snapshot.latest_policy_summary
    assert snapshot.latest_policy_observed_verification == [
        "verification became stale for `uv run pytest -q` after new mutating work [write changed src/loader/runtime/finalization.py]"
    ]
    assert [item.status for item in snapshot.recent_verification] == ["stale"]
    assert [item.command for item in snapshot.recent_verification] == [
        "uv run pytest -q"
    ]
    assert [item.detail for item in snapshot.recent_verification] == [
        "write changed src/loader/runtime/finalization.py"
    ]


def test_collect_prompt_diff_uses_persisted_prompt_history(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id, _ = _persist_session_with_dod(temp_dir)

    diff = collect_prompt_diff(project_root=temp_dir)

    assert diff.session_id == session_id
    assert diff.previous is not None
    assert diff.current is not None
    assert diff.current.workflow_mode == "execute"
    assert diff.previous.workflow_mode == "verify"
    assert any("Workflow mode changed:" in item for item in diff.highlights)
    assert "---" in diff.unified_diff
    assert "execute parser fix" in diff.unified_diff


def test_collect_workflow_artifact_diffs_reads_versioned_artifacts(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_rich_workflow(temp_dir)

    snapshot = collect_workflow_artifact_diffs(project_root=temp_dir)

    assert snapshot.session_id == session_id
    assert len(snapshot.entries) == 3
    assert {entry.kind for entry in snapshot.entries} == {
        "clarify_brief",
        "implementation_plan",
        "verification_plan",
    }
    assert any("notes.txt" in entry.unified_diff for entry in snapshot.entries)
    assert snapshot.highlights


def test_status_and_session_commands_render_persisted_state(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id, _ = _persist_session_with_dod(temp_dir)
    _persist_explore_snapshot(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    status_result = runner.invoke(cli_main_module.status_cli, ["--model", "llama3.1:8b"])
    list_result = runner.invoke(cli_main_module.session_cli, ["list"])
    show_result = runner.invoke(cli_main_module.session_cli, ["show", session_id])
    workflow_result = runner.invoke(cli_main_module.workflow_cli, ["show"])

    assert status_result.exit_code == 0
    assert session_id in status_result.output
    assert "fixing" in status_result.output
    assert "Runtime Owner" in status_result.output
    assert "runtime-handle (RuntimeHandle)" in status_result.output
    assert "1 allow / 2 deny / 1 ask" in status_result.output
    assert "native" in status_result.output
    assert "Runtime Config, Workflow Context, Mode Guidance" in status_result.output
    assert "Rules Source" in status_result.output
    assert "verification failed; returning to execute for fixes" in status_result.output
    assert "Completion Decision" in status_result.output
    assert "continued after verification failed" in status_result.output
    assert "completion -> finalize" in status_result.output
    assert "Finalizing completed turn" in status_result.output
    assert "Explore Turns" in status_result.output
    assert "Explore History" in status_result.output
    assert "What file did you mention?" in status_result.output
    assert "pytest -q" in status_result.output
    assert "1 failed" in status_result.output

    assert list_result.exit_code == 0
    assert session_id in list_result.output
    assert "Runtime Owner" in list_result.output
    assert "runtime-handle (RuntimeHandle)" in list_result.output
    assert "1 allow / 2 deny / 1 ask" in list_result.output
    assert "prompting enabled" in list_result.output
    assert "native" in list_result.output
    assert "Rules Source" in list_result.output
    assert "verification failed; returning to execute for fixes" in list_result.output
    assert "Completion Decision" in list_result.output
    assert "completion -> finalize" in list_result.output

    assert show_result.exit_code == 0
    assert session_id in show_result.output
    assert "Runtime Owner" in show_result.output
    assert "runtime-handle (RuntimeHandle)" in show_result.output
    assert "Patch the broken parser" in show_result.output
    assert "1 allow / 2 deny / 1 ask" in show_result.output
    assert "enabled" in show_result.output
    assert "Runtime Config, Workflow Context, Mode Guidance" in show_result.output
    assert "Rules Source" in show_result.output
    assert "verification failed; returning to execute for fixes" in show_result.output
    assert "Completion Decision" in show_result.output
    assert "Completion Trace" in show_result.output
    assert "Recent Verification" in show_result.output
    assert "continuation_check" in show_result.output
    assert "completion -> finalize" in show_result.output
    assert "Finalizing completed turn" in show_result.output
    assert "Policy Timeline" not in show_result.output
    assert "Workflow Timeline" in show_result.output
    assert "handoff" in show_result.output
    assert "next=verify" in show_result.output
    assert "pytest -q" in show_result.output
    assert "1 failed" in show_result.output

    assert workflow_result.exit_code == 0
    assert "Loader Workflow" in workflow_result.output
    assert "Workflow Timeline" in workflow_result.output
    assert session_id in workflow_result.output
    assert "Runtime Owner" in workflow_result.output
    assert "runtime-handle (RuntimeHandle)" in workflow_result.output
    assert "handoff" in workflow_result.output
    assert "next=verify" in workflow_result.output


def test_workflow_command_renders_policy_accountability_context(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_policy_accountability(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(cli_main_module.workflow_cli, ["show"])

    assert result.exit_code == 0
    assert session_id in result.output
    assert "repair_retry" in result.output
    assert "Repair path:" in result.output
    assert "Completion decision:" in result.output
    assert "verification_failed_reentry" in result.output
    assert "Policy Evidence Needed" in result.output
    assert "verification failed for `pytest -q`" in result.output
    assert "Observed Verification" in result.output
    assert "verification failed for `pytest -q` [1 failed]" in result.output
    assert "policy-stage=raw_text_tool_fallback" in result.output
    assert "policy-outcome=continue" in result.output
    assert "provenance=contradicts:verification@dod.evidence" in result.output
    assert "observed=verification failed for `pytest -q` [1 failed]" in result.output

    policy_result = runner.invoke(cli_main_module.workflow_cli, ["show", "--policy"])

    assert policy_result.exit_code == 0
    assert "Loader Workflow" in policy_result.output
    assert "Policy Timeline" in policy_result.output
    assert "policy-only" in policy_result.output
    assert "repair_retry" in policy_result.output
    assert "verification_failed_reentry" in policy_result.output
    assert "handoff" not in policy_result.output


def test_workflow_command_renders_stale_verification_context(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_stale_verification(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(cli_main_module.workflow_cli, ["show"])

    assert result.exit_code == 0
    assert session_id in result.output
    assert "Verify stale:" in result.output
    assert "verification_stale" in result.output
    assert "policy-outcome=stale" in result.output
    assert "Observed Verification" in result.output
    assert "uv run pytest -q" in result.output
    assert "new mutating work" in result.output


def test_collect_workflow_timeline_can_focus_on_policy_accountability(
    temp_dir: Path,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_policy_accountability(temp_dir)

    snapshot = collect_workflow_timeline(
        project_root=temp_dir,
        accountability_only=True,
    )

    assert snapshot.session_id == session_id
    assert snapshot.selected_accountability_only is True
    assert [entry.kind for entry in snapshot.entries] == [
        "repair_retry",
        "completion_check",
        "completion_continue",
    ]


def test_session_show_renders_policy_timeline_preview(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_policy_accountability(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    show_result = runner.invoke(cli_main_module.session_cli, ["show", session_id])

    assert show_result.exit_code == 0
    assert "Latest Policy" in show_result.output
    assert "verification_failed_reentry" in show_result.output
    assert "Policy Evidence Needed" in show_result.output
    assert "verification failed for `pytest -q`" in show_result.output
    assert "Observed Verification" in show_result.output
    assert "verification failed for `pytest -q` [1 failed]" in show_result.output
    assert "Policy Timeline" in show_result.output
    assert "repair_retry" in show_result.output
    assert "completion:" in show_result.output
    assert "provenance=contradicts:verification@dod.evidence" in show_result.output


def test_status_command_renders_latest_policy_summary(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_policy_accountability(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(cli_main_module.status_cli, [])

    assert result.exit_code == 0
    assert session_id in result.output
    assert "Latest Policy" in result.output
    assert "verification_failed_reentry" in result.output
    assert "Policy Evidence Needed" in result.output
    assert "verification failed for `pytest -q`" in result.output
    assert "Observed Verification" in result.output
    assert "verification failed for `pytest -q` [1 failed]" in result.output
    assert "Recent Verification" in result.output
    assert "policy-stage=definition_of_done" in result.output


def test_workflow_show_renders_workflow_ledger(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_session_with_rich_workflow(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(cli_main_module.workflow_cli, ["show"])

    assert result.exit_code == 0
    assert "Workflow Ledger" in result.output
    assert "Assumptions" in result.output
    assert "contradicted" in result.output
    assert "notes.txt stays out of scope" in result.output
    assert "Acceptance Anchors" in result.output
    assert "Decision Boundaries" in result.output


def test_workflow_show_command_supports_filters_and_highlights(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id = _persist_session_with_rich_workflow(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(
        cli_main_module.workflow_cli,
        ["show", "--kind", "reentry", "--limit", "1", session_id],
    )

    assert result.exit_code == 0
    assert "Loader Workflow" in result.output
    assert "1 shown / 3 total" in result.output
    assert "kind=reentry, limit=1" in result.output
    assert "Workflow Answers" in result.output
    assert "Recovered workflow:" in result.output
    assert "full_replan_required" in result.output
    assert "evidence=confirmed touchpoint:" in result.output

    clarify_result = runner.invoke(
        cli_main_module.workflow_cli,
        ["show", "--mode", "clarify", "--limit", "1", session_id],
    )

    assert clarify_result.exit_code == 0
    assert "stage=readiness" in clarify_result.output
    assert "pressure=tradeoff" in clarify_result.output
    assert "gates=non_goals,decision_boundaries" in clarify_result.output


def test_workflow_show_can_render_artifact_diffs(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_session_with_rich_workflow(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(
        cli_main_module.workflow_cli,
        ["show", "--diff", "--full-diff"],
    )

    assert result.exit_code == 0
    assert "Artifact Changes" in result.output
    assert "Artifact Diff Summary" in result.output
    assert "clarify_brief" in result.output
    assert "implementation_plan" in result.output
    assert "verification_plan" in result.output
    assert "notes.txt" in result.output


def test_collect_prompt_preview_uses_persisted_runtime_state(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    session_id, _ = _persist_session_with_dod(temp_dir)

    preview = collect_prompt_preview(
        temp_dir,
        model="qwen2.5-coder:14b",
    )

    assert preview.active_session_id == session_id
    assert preview.workflow_mode == "execute"
    assert preview.workflow_reason_code == "verification_failed_reentry"
    assert preview.workflow_decision_kind == "reentry"
    assert preview.permission_mode == "prompt"
    assert preview.prompt_format == (
        "native" if preview.capability_profile.supports_native_tools else "react"
    )
    assert preview.prompt_sections == [
        "Runtime Config",
        "Workflow Context",
        "Mode Guidance",
        "Project Context",
        "Project Tips",
    ]
    assert "## Execute Mode" in preview.content
    assert "Current task: Fix the failing tests" in preview.content


def test_prompt_show_command_renders_preview_without_model_call(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_session_with_dod(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)
    preview = collect_prompt_preview(
        temp_dir,
        model="qwen2.5-coder:14b",
        current_task="Preview the current Loader contract",
    )

    result = runner.invoke(
        cli_main_module.prompt_cli,
        ["show", "--model", "qwen2.5-coder:14b", "Preview the current Loader contract"],
    )

    assert result.exit_code == 0
    assert "Prompt Preview" in result.output
    assert "Prompt Body" in result.output
    assert "Preview the current Loader contract" in result.output
    assert preview.prompt_format in result.output
    assert "Workflow Context" in result.output
    assert "Execute Mode" in result.output


def test_prompt_diff_command_renders_persisted_prompt_changes(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_session_with_dod(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(cli_main_module.prompt_cli, ["diff", "--full"])

    assert result.exit_code == 0
    assert "Prompt Diff" in result.output
    assert "Prompt Changes" in result.output
    assert "Workflow mode changed:" in result.output
    assert "Prompt Unified Diff" in result.output
    assert "execute parser fix" in result.output


def test_permission_snapshot_and_dry_run_reflect_rules(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    (temp_dir / ".loader" / "permission-rules.json").write_text(
        "\n".join(
            [
                "{",
                '  "allow": [{"tool": "write", "contains": "safe change"}],',
                '  "deny": [{"tool": "write", "path_contains": "secrets"}],',
                '  "ask": [{"tool": "write", "path_contains": "README"}]',
                "}",
            ]
        )
        + "\n"
    )

    snapshot = collect_permission_snapshot(temp_dir, permission_mode="allow")
    check = dry_run_permission_check(
        "write",
        {
            "file_path": str(temp_dir / "README.md"),
            "content": "safe change\n",
        },
        project_root=temp_dir,
        permission_mode="allow",
    )

    assert snapshot.active_mode == "allow"
    assert snapshot.prompting_enabled is True
    assert snapshot.rules_valid is True
    assert snapshot.rule_counts == {"allow": 1, "deny": 1, "ask": 1}
    assert snapshot.normalized_rules["allow"][0].tool_name == "write"
    assert snapshot.normalized_rules["allow"][0].contains == "safe change"

    assert check.required_mode == "workspace-write"
    assert check.decision == "ask"
    assert check.matched_disposition == "ask"
    assert check.matched_rule == "tool=write, path_contains=README"
    assert "file_path=" in check.input_summary


def test_status_snapshot_reports_invalid_permission_rules(temp_dir: Path) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    (temp_dir / ".loader" / "permission-rules.json").write_text("{broken json")

    snapshot = collect_status_snapshot(temp_dir, permission_mode="prompt")

    assert snapshot.permission_rules_valid is False
    assert snapshot.permission_prompting_enabled is True
    assert snapshot.permission_rules_source.endswith(".loader/permission-rules.json")


def test_permissions_show_and_check_commands_render_policy(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    (temp_dir / ".loader" / "permission-rules.json").write_text(
        "\n".join(
            [
                "{",
                '  "allow": [{"tool": "write", "contains": "safe change"}],',
                '  "ask": [{"tool": "write", "path_contains": "README"}]',
                "}",
            ]
        )
        + "\n"
    )
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    show_result = runner.invoke(
        cli_main_module.permissions_cli,
        ["show", "--permission-mode", "allow"],
    )
    check_result = runner.invoke(
        cli_main_module.permissions_cli,
        [
            "check",
            "--permission-mode",
            "allow",
            "--args",
            '{"content":"safe change\\n"}',
            "write",
            "README.md",
        ],
    )

    assert show_result.exit_code == 0
    assert "Loader Permissions" in show_result.output
    assert "Permission Mode" in show_result.output
    assert "Rules Source" in show_result.output
    assert "safe change" in show_result.output
    assert "README" in show_result.output

    assert check_result.exit_code == 0
    assert "Permission Check" in check_result.output
    assert "workspace-write" in check_result.output
    assert "ask" in check_result.output
    assert "tool=write, path_contains=README" in check_result.output


def test_permissions_check_rejects_invalid_json_args(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(
        cli_main_module.permissions_cli,
        ["check", "bash", "--args", "{broken json", "ls"],
    )

    assert result.exit_code != 0
    assert "`--args` must be valid JSON" in result.output


def test_permissions_show_surfaces_invalid_rule_file(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    (temp_dir / ".loader" / "permission-rules.json").write_text("{broken json")
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    result = runner.invoke(cli_main_module.permissions_cli, ["show"])

    assert result.exit_code == 0
    assert "invalid" in result.output.lower()
    assert "Rule Error" in result.output
    assert "Rules Source" in result.output


def test_explore_command_can_show_and_reset_continuity(
    temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_workspace(temp_dir)
    _ensure_loader_dirs(temp_dir)
    _persist_explore_snapshot(temp_dir)
    runner = CliRunner()

    monkeypatch.chdir(temp_dir)

    status_result = runner.invoke(cli_main_module.explore_cli, ["--status"])

    assert status_result.exit_code == 0
    assert "Loader Explore State" in status_result.output
    assert "continue" in status_result.output
    assert "What file did you mention?" in status_result.output

    reset_result = runner.invoke(cli_main_module.explore_cli, ["--reset"])

    assert reset_result.exit_code == 0
    assert "Cleared persisted explore continuity." in reset_result.output
    assert ExploreStateStore(temp_dir).load() is None


def test_root_help_lists_special_commands() -> None:
    help_text = cli_main_module._loader_help_text()

    assert "loader doctor" in help_text
    assert "loader status" in help_text
    assert "loader explore <prompt>" in help_text
    assert "loader permissions show" in help_text
    assert "loader session resume <id>" in help_text


def test_main_dispatches_session_resume_to_primary_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_cli_main(*, args: list[str], prog_name: str) -> None:
        captured["args"] = args
        captured["prog_name"] = prog_name

    monkeypatch.setattr(cli_main_module.cli, "main", fake_cli_main)
    monkeypatch.setattr(sys, "argv", ["loader", "session", "resume", "abc123", "--no-tui"])

    cli_main_module.main()

    assert captured == {
        "args": ["--resume-target", "abc123", "--no-tui"],
        "prog_name": "loader",
    }
