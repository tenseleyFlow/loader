"""Definition-of-done gating and turn finalization for the runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..llm.base import Message, Role, ToolCall
from .context import RuntimeContext
from .dod import (
    DefinitionOfDone,
    DefinitionOfDoneStore,
    VerificationEvidence,
    build_verification_summary,
    collect_missing_declared_html_output_files,
    collect_planned_artifact_targets,
    derive_verification_commands,
    ensure_active_verification_attempt,
    planned_artifact_target_satisfied,
    synthesize_todo_items,
)
from .events import AgentEvent, TurnSummary
from .evidence_provenance import (
    EvidenceProvenance,
    EvidenceProvenanceStatus,
    summarize_evidence_provenance,
)
from .executor import ToolExecutor
from .logging import get_runtime_logger
from .memory import MemoryStore
from .policy_timeline import append_verification_timeline_entry
from .session import normalize_usage
from .tracing import RuntimeTracer
from .verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
)
from .workflow import (
    ModeDecision,
    WorkflowDecisionKind,
    WorkflowMode,
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
    effective_pending_todo_items,
    extract_verification_commands_from_markdown,
)

EventSink = Callable[[AgentEvent], Awaitable[None]]
WorkflowSetter = Callable[
    [ModeDecision, DefinitionOfDone, EventSink, TurnSummary],
    Awaitable[None],
]


@dataclass
class CompletionGateResult:
    """Outcome of the definition-of-done completion gate."""

    should_continue: bool
    reason_code: str
    reason_summary: str
    final_response: str
    evidence_provenance: list[EvidenceProvenance] = field(default_factory=list)
    verification_observations: list[VerificationObservation] = field(default_factory=list)


class TurnFinalizer:
    """Owns DoD verification, status emission, and turn finalization."""

    def __init__(
        self,
        context: RuntimeContext,
        tracer: RuntimeTracer,
        dod_store: DefinitionOfDoneStore,
        set_workflow_mode: WorkflowSetter,
    ) -> None:
        self.context = context
        self.tracer = tracer
        self.dod_store = dod_store
        self.set_workflow_mode = set_workflow_mode

    @property
    def _prompt_format(self) -> str | None:
        return self.context.prompt_format

    @property
    def _prompt_sections(self) -> list[str]:
        return list(self.context.prompt_sections)

    async def run_definition_of_done_gate(
        self,
        *,
        dod: DefinitionOfDone,
        candidate_response: str,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
    ) -> CompletionGateResult:
        """Gate completion on DoD state and verification evidence."""

        implementation_item = "Complete the requested work"
        verification_item = "Collect verification evidence"

        tracked_pending_items = [
            item
            for item in effective_pending_todo_items(
                dod,
                project_root=self.context.project_root,
            )
            if item not in {implementation_item, verification_item}
        ]
        missing_planned_artifacts = _missing_planned_artifact_labels(
            dod,
            project_root=self.context.project_root,
        )

        mutating_paths = [path for path in dod.touched_files if path]
        requires_verification = bool(mutating_paths or dod.mutating_actions)
        if (
            tracked_pending_items
            and not requires_verification
            and _response_declares_no_mutation_needed(candidate_response)
        ):
            tracked_pending_items = [
                item
                for item in tracked_pending_items
                if not _is_task_restatement_pending_item(item, dod.task_statement)
            ]
        rlog = get_runtime_logger()
        rlog.completion_check(
            "dod_gate",
            "requires_verification" if requires_verification else "no_verification",
            reason=f"files={mutating_paths[:3]}, actions={len(dod.mutating_actions)}"
            if requires_verification else None,
        )
        if missing_planned_artifacts:
            recovery_nudge = _build_missing_artifact_recovery_nudge(
                _first_missing_planned_artifact(
                    dod,
                    project_root=self.context.project_root,
                )
            )
            if recovery_nudge:
                self.context.queue_steering_message(recovery_nudge)
            missing_provenance = [
                EvidenceProvenance(
                    category="tracked_work",
                    source="dod.implementation_plan",
                    summary=f"planned artifact still missing: {label}",
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject=label,
                )
                for label in missing_planned_artifacts
            ]
            missing_text = "\n".join(
                f"- {label}" for label in missing_planned_artifacts[:8]
            )
            pending_text = ""
            if tracked_pending_items:
                pending_text = (
                    "\nRemaining tracked work:\n"
                    + "\n".join(f"- {item}" for item in tracked_pending_items[:6])
                )
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            self.context.session.append(
                Message(
                    role=Role.USER,
                    content=(
                        "[PLANNED ARTIFACTS STILL MISSING]\n"
                        "The explicit implementation plan is not complete yet. "
                        "Do not move to verification or final confirmation.\n\n"
                        "Missing planned artifacts:\n"
                        f"{missing_text}"
                        f"{pending_text}\n\n"
                        "Continue by creating or updating the missing planned artifacts."
                    ),
                )
            )
            return CompletionGateResult(
                should_continue=True,
                reason_code="planned_artifacts_missing_continue",
                reason_summary=(
                    "continued because explicitly planned artifacts were still missing "
                    "before verification"
                ),
                final_response="",
                evidence_provenance=missing_provenance,
            )
        if tracked_pending_items and not requires_verification:
            pending_provenance = [
                EvidenceProvenance(
                    category="tracked_work",
                    source="dod.pending_items",
                    summary=f"tracked work item still pending: {item}",
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject=item,
                )
                for item in tracked_pending_items
            ]
            pending_text = "\n".join(f"- {item}" for item in tracked_pending_items)
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            self.context.session.append(
                Message(
                    role=Role.USER,
                    content=(
                        "[PENDING WORK REMAINS]\n"
                        "The tracked work items are not complete yet:\n"
                        f"{pending_text}\n\n"
                        "Continue the task, and update TodoWrite as you make progress."
                    ),
                )
            )
            return CompletionGateResult(
                should_continue=True,
                reason_code="pending_items_continue",
                reason_summary="continued because tracked work items still remained incomplete",
                final_response="",
                evidence_provenance=pending_provenance,
            )

        if not requires_verification:
            if implementation_item in dod.pending_items:
                dod.pending_items.remove(implementation_item)
            if implementation_item not in dod.completed_items:
                dod.completed_items.append(implementation_item)
            skip_provenance = [
                EvidenceProvenance(
                    category="verification",
                    source="dod.mutating_actions",
                    summary="verification was skipped because no mutating work required checks",
                    status=EvidenceProvenanceStatus.CONTEXT.value,
                )
            ]
            skip_observations = [
                VerificationObservation(
                    status=VerificationObservationStatus.SKIPPED.value,
                    summary=(
                        "verification was skipped because no mutating work "
                        "required checks"
                    ),
                )
            ]
            dod.status = "done"
            dod.last_verification_result = "skipped"
            summary.verification_status = "skipped"
            summary.definition_of_done = dod
            self.context.session.append_workflow_timeline_entry(
                WorkflowTimelineEntry(
                    timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    kind=WorkflowTimelineEntryKind.VERIFY_SKIP.value,
                    mode=self.context.workflow_mode,
                    reason_code="verification_not_required",
                    summary="verification skipped because the turn made no mutating changes",
                    decision_kind=WorkflowDecisionKind.FORCED.value,
                    prompt_format=self._prompt_format,
                    prompt_sections=self._prompt_sections,
                    evidence_summary=summarize_evidence_provenance(skip_provenance),
                    evidence_provenance=skip_provenance,
                    verification_observations=skip_observations,
                )
            )
            summary.workflow_timeline = list(self.context.session.workflow_timeline)
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            await emit(AgentEvent(
                type="todo_update",
                todo_items=synthesize_todo_items(dod),
            ))
            return CompletionGateResult(
                should_continue=False,
                reason_code="non_mutating_response_accepted",
                reason_summary=(
                    "accepted the response because no mutating work required "
                    "verification"
                ),
                final_response=candidate_response,
                evidence_provenance=skip_provenance,
                verification_observations=skip_observations,
            )

        current_verification_signature = _verification_state_signature(dod)
        if (
            dod.last_verification_result == "failed"
            and dod.last_verification_signature
            and dod.last_verification_signature == current_verification_signature
        ):
            summary.verification_status = "failed"
            summary.definition_of_done = dod
            failed_provenance = _verification_result_provenance(dod, passed=False)
            if dod.retry_count >= dod.retry_budget:
                dod.status = "failed"
                dod.confidence = "low"
                self.dod_store.save(dod)
                await self.emit_dod_status(emit, dod)
                exhausted_response = (
                    "I couldn't verify that the task is complete within the retry budget.\n\n"
                    f"{build_verification_summary(dod.evidence)}"
                )
                return CompletionGateResult(
                    should_continue=False,
                    reason_code="verification_retry_budget_exhausted",
                    reason_summary="stopped after verification retry budget was exhausted",
                    final_response=exhausted_response,
                    evidence_provenance=failed_provenance,
                    verification_observations=_verification_result_observations(
                        dod,
                        passed=False,
                        attempt_id=dod.active_verification_attempt_id,
                        attempt_number=dod.active_verification_attempt_number,
                    ),
                )
            repair_prompt = (
                "[DEFINITION OF DONE CHECK STILL FAILING]\n"
                f"Task: {dod.task_statement}\n"
                "No new file changes were made since the last failed verification.\n\n"
                f"{build_verification_summary(dod.evidence)}\n\n"
                f"{_build_verification_repair_guidance(dod, project_root=self.context.project_root)}\n\n"
                "Apply a concrete edit or patch before trying to finish again."
            )
            recovery_nudge = _build_verification_failure_recovery_nudge(
                dod,
                project_root=self.context.project_root,
            )
            if recovery_nudge:
                self.context.queue_steering_message(recovery_nudge)
            self.context.session.append(Message(role=Role.USER, content=repair_prompt))
            return CompletionGateResult(
                should_continue=True,
                reason_code="verification_failed_no_new_changes",
                reason_summary=(
                    "continued because verification already failed and no new "
                    "mutating changes were made before trying to finish again"
                ),
                final_response="",
                evidence_provenance=failed_provenance,
                verification_observations=_verification_result_observations(
                    dod,
                    passed=False,
                    attempt_id=dod.active_verification_attempt_id,
                    attempt_number=dod.active_verification_attempt_number,
                ),
            )

        verify_item = "Collect verification evidence"
        if verify_item not in dod.pending_items and verify_item not in dod.completed_items:
            dod.pending_items.append(verify_item)

        if (
            not dod.verification_commands
            and dod.verification_plan
            and Path(dod.verification_plan).exists()
        ):
            dod.verification_commands = extract_verification_commands_from_markdown(
                Path(dod.verification_plan).read_text()
            )

        if (
            not dod.verification_commands
            and dod.implementation_plan
            and Path(dod.implementation_plan).exists()
        ):
            dod.verification_commands = extract_verification_commands_from_markdown(
                Path(dod.implementation_plan).read_text()
            )

        if not dod.verification_commands:
            dod.verification_commands = derive_verification_commands(
                dod,
                project_root=self.context.project_root,
                task_statement=dod.task_statement,
            )
        else:
            for command in derive_verification_commands(
                dod,
                project_root=self.context.project_root,
                task_statement=dod.task_statement,
                supplement_existing=True,
            ):
                if command not in dod.verification_commands:
                    dod.verification_commands.append(command)

        await self.set_workflow_mode(
            ModeDecision.transition(
                WorkflowMode.VERIFY,
                reason_code="definition_of_done_requires_verification",
                reason_summary="definition-of-done gate requires verification",
                decision_kind=WorkflowDecisionKind.HANDOFF,
            ),
            dod=dod,
            emit=emit,
            summary=summary,
        )
        if dod.verification_commands:
            attempt = ensure_active_verification_attempt(dod)
            dod.last_verification_result = VerificationObservationStatus.PENDING.value
            self.dod_store.save(dod)
            append_verification_timeline_entry(
                self.context,
                summary,
                reason_code="verification_pending",
                reason_summary=(
                    "verification is pending for the active command set"
                ),
                evidence_summary=[
                    f"verification command pending: {command}"
                    for command in dod.verification_commands[:2]
                ],
                evidence_provenance=_pending_verification_provenance(dod),
                verification_observations=_pending_verification_observations(
                    dod,
                    attempt_id=attempt.attempt_id,
                    attempt_number=attempt.attempt_number,
                ),
            )
        verification_passed = await self.verify_definition_of_done(
            dod=dod,
            emit=emit,
            summary=summary,
            executor=executor,
        )
        verification_observations = _verification_result_observations(
            dod,
            passed=verification_passed,
            attempt_id=dod.active_verification_attempt_id,
            attempt_number=dod.active_verification_attempt_number,
        )
        if verification_passed:
            passed_provenance = _verification_result_provenance(dod, passed=True)
            if verify_item in dod.pending_items:
                dod.pending_items.remove(verify_item)
            if verify_item not in dod.completed_items:
                dod.completed_items.append(verify_item)
            for pending in list(dod.pending_items):
                if pending not in dod.completed_items:
                    dod.completed_items.append(pending)
            dod.pending_items = []
            dod.status = "done"
            dod.last_verification_result = "passed"
            dod.confidence = "high"
            summary.verification_status = "passed"
            summary.definition_of_done = dod
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            # Auto-complete all todo items when DoD is done
            await emit(AgentEvent(
                type="todo_update",
                todo_items=synthesize_todo_items(dod),
            ))
            verified_response = candidate_response
            verification_summary = build_verification_summary(dod.evidence)
            if verification_summary not in verified_response:
                verified_response = f"{candidate_response.rstrip()}\n\n{verification_summary}"
            return CompletionGateResult(
                should_continue=False,
                reason_code="verification_passed",
                reason_summary="accepted the response after verification evidence passed",
                final_response=verified_response,
                evidence_provenance=passed_provenance,
                verification_observations=verification_observations,
            )

        dod.last_verification_result = "failed"
        summary.verification_status = "failed"
        summary.definition_of_done = dod
        failed_provenance = _verification_result_provenance(dod, passed=False)
        if dod.retry_count >= dod.retry_budget:
            dod.status = "failed"
            dod.confidence = "low"
            self.dod_store.save(dod)
            await self.emit_dod_status(emit, dod)
            failure_summary = build_verification_summary(dod.evidence)
            exhausted_response = (
                "I couldn't verify that the task is complete within the retry budget.\n\n"
                f"{failure_summary}"
            )
            return CompletionGateResult(
                should_continue=False,
                reason_code="verification_retry_budget_exhausted",
                reason_summary="stopped after verification retry budget was exhausted",
                final_response=exhausted_response,
                evidence_provenance=failed_provenance,
                verification_observations=verification_observations,
            )

        dod.retry_count += 1
        dod.status = "fixing"
        dod.confidence = "medium"
        self.dod_store.save(dod)
        await self.emit_dod_status(emit, dod)
        recovery_nudge = _build_verification_failure_recovery_nudge(
            dod,
            project_root=self.context.project_root,
        )
        if recovery_nudge:
            self.context.queue_steering_message(recovery_nudge)
        await self.set_workflow_mode(
            ModeDecision.transition(
                WorkflowMode.EXECUTE,
                reason_code="verification_failed_reentry",
                reason_summary="verification failed; returning to execute for fixes",
                decision_kind=WorkflowDecisionKind.REENTRY,
            ),
            dod=dod,
            emit=emit,
            summary=summary,
        )
        failure_prompt = (
            "[DEFINITION OF DONE CHECK FAILED]\n"
            f"Task: {dod.task_statement}\n"
            f"Attempt: {dod.retry_count}/{dod.retry_budget}\n"
            f"Pending items: {', '.join(dod.pending_items)}\n\n"
            f"{build_verification_summary(dod.evidence)}\n\n"
            f"{_build_verification_repair_guidance(dod, project_root=self.context.project_root)}\n\n"
            "Fix the failures above, then finish the task again."
        )
        self.context.session.append(Message(role=Role.USER, content=failure_prompt))
        return CompletionGateResult(
            should_continue=True,
            reason_code="verification_failed_reentry",
            reason_summary=(
                "continued after verification failed and the runtime re-entered "
                "execute mode"
            ),
            final_response="",
            evidence_provenance=failed_provenance,
            verification_observations=verification_observations,
        )

    async def verify_definition_of_done(
        self,
        *,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
    ) -> bool:
        """Collect verification evidence for one DoD."""

        dod.status = "verifying"
        dod.last_verification_signature = _verification_state_signature(dod)
        self.dod_store.save(dod)
        await self.emit_dod_status(emit, dod)
        attempt = ensure_active_verification_attempt(dod)

        if not dod.verification_commands:
            missing_provenance = _missing_verification_provenance()
            missing_observations = _missing_verification_observations(
                attempt_id=attempt.attempt_id,
                attempt_number=attempt.attempt_number,
            )
            append_verification_timeline_entry(
                self.context,
                summary,
                reason_code="verification_commands_missing",
                reason_summary="verification commands were still missing at execution time",
                evidence_provenance=missing_provenance,
                verification_observations=missing_observations,
            )
            summary.verification_status = "failed"
            return False

        dod.evidence = []
        all_passed = True
        for index, command in enumerate(dod.verification_commands, start=1):
            verification_call = ToolCall(
                id=f"verify-{summary.iterations}-{index}",
                name="bash",
                arguments={"command": command, "cwd": str(self.context.project_root)},
            )
            await emit(
                AgentEvent(
                    type="tool_call",
                    tool_name=verification_call.name,
                    tool_call_id=verification_call.id,
                    tool_args=verification_call.arguments,
                    phase="verification",
                )
            )
            outcome = await executor.execute_tool_call(
                verification_call,
                on_confirmation=None,
                emit_confirmation=None,
                source="verification",
                skip_duplicate_check=True,
                record_action=False,
                skip_confirmation=True,
            )
            await emit(
                AgentEvent(
                    type="tool_result",
                    content=outcome.event_content,
                    tool_name=verification_call.name,
                    tool_call_id=outcome.tool_call.id,
                    tool_metadata=(
                        outcome.registry_result.metadata
                        if outcome.registry_result is not None
                        else None
                    ),
                    is_error=outcome.is_error,
                    phase="verification",
                )
            )

            metadata = outcome.registry_result.metadata if outcome.registry_result else {}
            evidence = VerificationEvidence(
                command=command,
                passed=not outcome.is_error,
                exit_code=metadata.get("exit_code"),
                stdout=str(metadata.get("stdout", "")),
                stderr=str(metadata.get("stderr", "")),
                output=outcome.result_output,
                kind=_classify_verification_kind(command),
            )
            evidence = _maybe_mark_optional_verification_skip(evidence)
            dod.evidence.append(evidence)
            observation = _verification_observation_from_evidence(
                evidence,
                attempt_id=attempt.attempt_id,
                attempt_number=attempt.attempt_number,
            )
            provenance = _verification_provenance_from_evidence(evidence)
            append_verification_timeline_entry(
                self.context,
                summary,
                reason_code=_verification_timeline_reason_code(evidence),
                reason_summary=_verification_timeline_reason_summary(evidence),
                evidence_provenance=provenance,
                verification_observations=[observation],
            )
            all_passed = all_passed and (evidence.passed or evidence.skipped)
            summary.tool_result_messages.append(outcome.message)
            self.context.session.append(outcome.message)

        self.dod_store.save(dod)
        summary.verification_status = "passed" if all_passed else "failed"
        return all_passed

    def finalize_summary(self, summary: TurnSummary) -> TurnSummary:
        """Finalize usage, memory capture, and trace data for one turn."""

        summary.usage["tool_calls"] = len(summary.tool_result_messages)
        summary.usage["iterations"] = summary.iterations
        summary.cumulative_usage = self.context.session.record_turn_usage(
            summary.usage,
            tool_calls=len(summary.tool_result_messages),
            iterations=summary.iterations,
        )
        summary.session_id = self.context.session.session_id
        summary.completion_decision_code = getattr(
            self.context.session,
            "last_completion_decision_code",
            None,
        )
        summary.completion_decision_summary = getattr(
            self.context.session,
            "last_completion_decision_summary",
            None,
        )
        summary.completion_trace = list(
            getattr(self.context.session, "completion_trace", [])
        )
        summary.last_turn_transition_summary = (
            getattr(self.context.session, "last_turn_transition_summary", None)
        )
        summary.workflow_timeline = list(
            getattr(self.context.session, "workflow_timeline", [])
        )
        if summary.definition_of_done and summary.definition_of_done.status == "done":
            MemoryStore(self.context.project_root).capture_definition_of_done(
                build_verification_summary(summary.definition_of_done.evidence)
            )
        summary.trace = list(self.tracer.events)
        return summary

    async def emit_dod_status(self, emit: EventSink, dod: DefinitionOfDone) -> None:
        """Emit the latest definition-of-done status."""

        self.dod_store.save(dod)
        await emit(
            AgentEvent(
                type="dod_status",
                content=(
                    f"DoD: {dod.status} "
                    f"({len(dod.pending_items)} pending"
                    + (
                        f", last verification: {dod.last_verification_result}"
                        if dod.last_verification_result
                        else ""
                    )
                    + ")"
                ),
                dod_status=dod.status,
                pending_items_count=len(dod.pending_items),
                last_verification_result=dod.last_verification_result,
                definition_of_done=dod,
            )
        )


def _verification_result_provenance(
    dod: DefinitionOfDone,
    *,
    passed: bool,
) -> list[EvidenceProvenance]:
    entries: list[EvidenceProvenance] = []
    target_status = (
        EvidenceProvenanceStatus.SUPPORTS.value
        if passed
        else EvidenceProvenanceStatus.CONTRADICTS.value
    )
    for evidence in dod.evidence:
        if evidence.passed != passed:
            continue
        command = evidence.command or "verification"
        summary = (
            f"verification passed for `{command}`"
            if passed
            else f"verification failed for `{command}`"
        )
        detail = _verification_detail(evidence)
        entries.append(
            EvidenceProvenance(
                category="verification",
                source="dod.evidence",
                summary=summary,
                status=target_status,
                subject=command,
                detail=detail,
            )
        )
    if entries:
        observed_commands = {
            evidence.command for evidence in dod.evidence if evidence.command
        }
        if not passed:
            for command in dod.verification_commands:
                if not command or command in observed_commands:
                    continue
                entries.append(
                    EvidenceProvenance(
                        category="verification",
                        source="dod.verification_commands",
                        summary=(
                            "verification did not produce an observed result for "
                            f"`{command}`"
                        ),
                        status=EvidenceProvenanceStatus.MISSING.value,
                        subject=command,
                    )
                )
        return entries

    if not passed:
        for command in dod.verification_commands:
            if not command:
                continue
            entries.append(
                EvidenceProvenance(
                    category="verification",
                    source="dod.verification_commands",
                    summary=(
                        "verification did not produce an observed result for "
                        f"`{command}`"
                    ),
                    status=EvidenceProvenanceStatus.MISSING.value,
                    subject=command,
                )
            )
        if entries:
            return entries
        return [
            EvidenceProvenance(
                category="verification",
                source="dod.verification_commands",
                summary="verification commands were still missing at execution time",
                status=EvidenceProvenanceStatus.MISSING.value,
            )
        ]

    for command in dod.verification_commands:
        if not command:
            continue
        entries.append(
            EvidenceProvenance(
                category="verification",
                source="dod.verification_commands",
                summary=(
                    f"verification passed for `{command}`"
                    if passed
                    else f"verification failed for `{command}`"
                ),
                status=target_status,
                subject=command,
            )
        )
    return entries


def _missing_planned_artifact_labels(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> list[str]:
    labels: list[str] = []
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        exists = target.is_dir() if expect_directory else target.is_file()
        if exists:
            continue
        label = target.name or str(target)
        if expect_directory and not label.endswith("/"):
            label += "/"
        labels.append(f"`{label}`")
    return labels


def _first_missing_planned_artifact(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> tuple[Path, bool] | None:
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        exists = target.is_dir() if expect_directory else target.is_file()
        if not exists:
            return target, expect_directory
    return None


def _build_missing_artifact_recovery_nudge(
    missing_artifact: tuple[Path, bool] | None,
) -> str | None:
    if missing_artifact is None:
        return None

    target, expect_directory = missing_artifact
    label = target.name or str(target)
    if expect_directory and not label.endswith("/"):
        label += "/"

    if expect_directory:
        return (
            "Your prior completion claim was incorrect because "
            f"`{label}` does not exist yet. Do not summarize, mark completion, or "
            "write bookkeeping notes yet. Your next response should be one concrete "
            f"tool call that creates `{target}`. If a specific missing fact blocks "
            "that step, ask one precise question."
        )

    return (
        "Your prior completion claim was incorrect because "
        f"`{label}` does not exist yet. Do not summarize, mark completion, or "
        "write bookkeeping notes yet. Your next response should be one concrete "
        f"`write` or `edit`-style tool call that creates or updates `{target}`. "
        "If a specific missing fact blocks that step, ask one precise question."
    )


def _verification_result_observations(
    dod: DefinitionOfDone,
    *,
    passed: bool,
    attempt_id: str | None,
    attempt_number: int | None,
) -> list[VerificationObservation]:
    entries: list[VerificationObservation] = []
    target_status = (
        VerificationObservationStatus.PASSED.value
        if passed
        else VerificationObservationStatus.FAILED.value
    )
    observed_commands: set[str] = set()
    for evidence in dod.evidence:
        if evidence.passed != passed:
            continue
        command = evidence.command or "verification"
        observed_commands.add(command)
        entries.append(
            VerificationObservation(
                status=target_status,
                summary=(
                    f"verification passed for `{command}`"
                    if passed
                    else f"verification failed for `{command}`"
                ),
                command=evidence.command or None,
                kind=evidence.kind,
                exit_code=evidence.exit_code,
                detail=_verification_detail(evidence),
                attempt_id=attempt_id,
                attempt_number=attempt_number,
            )
        )

    if passed:
        return entries

    for command in dod.verification_commands:
        if not command or command in observed_commands:
            continue
        entries.append(
            VerificationObservation(
                status=VerificationObservationStatus.MISSING.value,
                summary=f"verification did not produce an observed result for `{command}`",
                command=command,
                kind=_classify_verification_kind(command),
                attempt_id=attempt_id,
                attempt_number=attempt_number,
            )
        )

    if entries:
        return entries

    return [
        VerificationObservation(
            status=VerificationObservationStatus.MISSING.value,
            summary="verification commands were still missing at execution time",
            attempt_id=attempt_id,
            attempt_number=attempt_number,
        )
    ]


def _verification_observation_from_evidence(
    evidence: VerificationEvidence,
    *,
    attempt_id: str | None,
    attempt_number: int | None,
) -> VerificationObservation:
    return VerificationObservation(
        status=(
            VerificationObservationStatus.SKIPPED.value
            if evidence.skipped
            else VerificationObservationStatus.PASSED.value
            if evidence.passed
            else VerificationObservationStatus.FAILED.value
        ),
        summary=_verification_timeline_reason_summary(evidence),
        command=evidence.command or None,
        kind=evidence.kind,
        exit_code=evidence.exit_code,
        detail=_verification_detail(evidence),
        attempt_id=attempt_id,
        attempt_number=attempt_number,
    )


def _verification_provenance_from_evidence(
    evidence: VerificationEvidence,
) -> list[EvidenceProvenance]:
    command = evidence.command or "verification"
    return [
        EvidenceProvenance(
            category="verification",
            source="dod.evidence",
            summary=_verification_timeline_reason_summary(evidence),
            status=(
                EvidenceProvenanceStatus.CONTEXT.value
                if evidence.skipped
                else EvidenceProvenanceStatus.SUPPORTS.value
                if evidence.passed
                else EvidenceProvenanceStatus.CONTRADICTS.value
            ),
            subject=command,
            detail=_verification_detail(evidence),
        )
    ]


def _missing_verification_observations(
    *,
    attempt_id: str | None,
    attempt_number: int | None,
) -> list[VerificationObservation]:
    return [
        VerificationObservation(
            status=VerificationObservationStatus.MISSING.value,
            summary="verification commands were still missing at execution time",
            attempt_id=attempt_id,
            attempt_number=attempt_number,
        )
    ]


def _missing_verification_provenance() -> list[EvidenceProvenance]:
    return [
        EvidenceProvenance(
            category="verification",
            source="dod.verification_commands",
            summary="verification commands were still missing at execution time",
            status=EvidenceProvenanceStatus.MISSING.value,
        )
    ]


def _pending_verification_observations(
    dod: DefinitionOfDone,
    *,
    attempt_id: str | None,
    attempt_number: int | None,
) -> list[VerificationObservation]:
    observations: list[VerificationObservation] = []
    for command in dod.verification_commands:
        observations.append(
            VerificationObservation(
                status=VerificationObservationStatus.PENDING.value,
                summary=f"verification pending for `{command}`",
                command=command,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
            )
        )
    return observations


def _pending_verification_provenance(
    dod: DefinitionOfDone,
) -> list[EvidenceProvenance]:
    provenance: list[EvidenceProvenance] = []
    for command in dod.verification_commands:
        provenance.append(
            EvidenceProvenance(
                category="verification",
                source="dod.verification_commands",
                summary=f"verification command pending: {command}",
                status=EvidenceProvenanceStatus.MISSING.value,
                subject=command,
            )
        )
    return provenance


def _verification_detail(evidence: VerificationEvidence) -> str | None:
    for candidate in (evidence.stdout, evidence.stderr, evidence.output):
        text = str(candidate).strip()
        if text:
            return text.splitlines()[0]
    return None


def _verification_timeline_reason_code(evidence: VerificationEvidence) -> str:
    if evidence.skipped:
        return "verification_command_skipped"
    if evidence.passed:
        return "verification_command_passed"
    return "verification_command_failed"


def _verification_timeline_reason_summary(evidence: VerificationEvidence) -> str:
    command = evidence.command or "verification"
    if evidence.skipped:
        return f"verification skipped for `{command}`"
    if evidence.passed:
        return f"verification passed for `{command}`"
    return f"verification failed for `{command}`"


def _maybe_mark_optional_verification_skip(
    evidence: VerificationEvidence,
) -> VerificationEvidence:
    detail = "\n".join(
        part for part in (evidence.stderr, evidence.output) if str(part).strip()
    ).lower()
    command = (evidence.command or "").lower()
    if (
        not evidence.passed
        and evidence.exit_code == 127
        and "command not found" in detail
        and "html5validator" in command
    ):
        evidence.skipped = True
    return evidence


def _verification_state_signature(dod: DefinitionOfDone) -> str:
    touched = "|".join(sorted(set(dod.touched_files)))
    commands = "|".join(sorted(set(dod.successful_commands)))
    return (
        f"lines={dod.line_changes}"
        f";touched={touched}"
        f";actions={len(dod.mutating_actions)}"
        f";commands={commands}"
    )


def _normalize_pending_statement(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _is_task_restatement_pending_item(item: str, task_statement: str) -> bool:
    normalized_item = _normalize_pending_statement(item)
    normalized_task = _normalize_pending_statement(task_statement)
    return bool(normalized_item and normalized_item == normalized_task)


def _response_declares_no_mutation_needed(candidate_response: str) -> bool:
    lowered = candidate_response.lower()
    return any(
        phrase in lowered
        for phrase in (
            "already correct",
            "already up to date",
            "already matches",
            "already complete",
            "no edit is needed",
            "no edits are needed",
            "no change is needed",
            "no changes are needed",
            "nothing to change",
            "no update is needed",
            "no updates are needed",
        )
    )


def _build_verification_repair_guidance(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> str:
    repair_targets = _extract_verification_repair_targets(dod.evidence)
    missing_planned_outputs = _extract_verification_missing_planned_outputs(
        dod,
        project_root=project_root,
        repair_targets=repair_targets,
    )
    fixes = _extract_verification_repairs(
        dod.evidence,
        repair_targets=repair_targets,
    )
    repair_source_paths = _existing_repair_source_paths(
        dod,
        repair_targets=repair_targets,
        project_root=project_root,
    )
    if missing_planned_outputs:
        missing_output_keys = {
            str(path.resolve(strict=False)) for path in missing_planned_outputs
        }
        lines = ["Repair focus:"]
        for path in missing_planned_outputs[:4]:
            lines.append(
                f"- Continue the declared output set by creating missing planned artifact `{path}`."
            )
        for target in repair_targets:
            normalized_expected = str(
                Path(target.expected_path).resolve(strict=False)
            )
            if normalized_expected not in missing_output_keys:
                continue
            lines.append(
                f"- Existing file `{target.artifact_path}` already references "
                f"`{target.failing_reference}` -> `{normalized_expected}`."
            )
        primary_missing = missing_planned_outputs[0]
        lines.extend(
            [
                f"- Immediate next step: write `{primary_missing}`.",
                "- Do not rewrite existing aggregate files to match the partial artifact "
                "set while these declared outputs are still missing.",
                *(
                    [
                        "- Use the existing artifact files as the source of truth while "
                        "continuing the declared output set: "
                        + ", ".join(f"`{path}`" for path in repair_source_paths[:6])
                        + (", ..." if len(repair_source_paths) > 6 else "")
                    ]
                    if repair_source_paths
                    else []
                ),
                "- After each new file write, continue with the next missing declared "
                "output or rerun verification once the declared output set exists.",
                "- Do not reread unrelated reference materials or restart discovery "
                "while this repair target is unresolved.",
            ]
        )
        return "\n".join(lines)
    if not fixes and not repair_targets:
        return (
            "Use the failed verification evidence directly, avoid rereading unrelated "
            "files, and fix the target file before retrying."
        )

    lines = ["Repair focus:"]
    lines.extend(f"- {item}" for item in fixes)
    primary_target = repair_targets[0] if repair_targets else None
    if primary_target is not None:
        lines.extend(
            [
                f"- Immediate next step: edit `{primary_target.artifact_path}`.",
                "- If the broken reference should remain, create "
                f"`{primary_target.expected_path}`; otherwise remove or replace "
                f"`{primary_target.failing_reference}`.",
                *(
                    [
                        "- Use the existing artifact files as the source of truth while "
                        "repairing this file: "
                        + ", ".join(f"`{path}`" for path in repair_source_paths[:6])
                        + (", ..." if len(repair_source_paths) > 6 else "")
                    ]
                    if repair_source_paths
                    else []
                ),
                "- Do not reread unrelated reference materials or restart discovery "
                "while this concrete repair target is unresolved.",
            ]
        )
    else:
        lines.append(
            "- Reuse these exact failures instead of restarting discovery from earlier "
            "chapters."
        )
    return "\n".join(lines)


def _extract_verification_repairs(
    evidence_items: list[VerificationEvidence],
    *,
    repair_targets: list[VerificationRepairTarget] | None = None,
) -> list[str]:
    fixes: list[str] = []
    target_map = {
        (target.artifact_path, target.failing_reference, target.expected_path): target
        for target in (repair_targets or _extract_verification_repair_targets(evidence_items))
    }
    for target in target_map.values():
        item = (
            f"Fix the broken local reference `{target.failing_reference}` in "
            f"`{target.artifact_path}`."
        )
        if item not in fixes:
            fixes.append(item)
    for evidence in evidence_items:
        for candidate in (evidence.stderr, evidence.output, evidence.stdout):
            for problem in _extract_missing_local_html_links(str(candidate)):
                parsed = _parse_missing_local_html_link(problem)
                if parsed is not None:
                    key = (
                        parsed.artifact_path,
                        parsed.failing_reference,
                        parsed.expected_path,
                    )
                    if key in target_map:
                        continue
                item = (
                    "Fix the missing local HTML link "
                    f"`{problem}` in the edited artifact set."
                )
                if item not in fixes:
                    fixes.append(item)
    return fixes


@dataclass(frozen=True)
class VerificationRepairTarget:
    """Structured repair target extracted from failed verification evidence."""

    artifact_path: str
    failing_reference: str
    expected_path: str


def _build_verification_failure_recovery_nudge(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> str | None:
    repair_targets = _extract_verification_repair_targets(dod.evidence)
    missing_planned_outputs = _extract_verification_missing_planned_outputs(
        dod,
        project_root=project_root,
        repair_targets=repair_targets,
    )
    repair_source_paths = _existing_repair_source_paths(
        dod,
        repair_targets=repair_targets,
        project_root=project_root,
    )
    if missing_planned_outputs:
        primary_missing = missing_planned_outputs[0]
        source_hint = ""
        if repair_source_paths:
            preview = ", ".join(f"`{path}`" for path in repair_source_paths[:4])
            if len(repair_source_paths) > 4:
                preview += ", ..."
            source_hint = (
                " Use the generated files already on disk as the source of truth: "
                f"{preview}."
            )
        return (
            "Verification failed because the current artifact set already references "
            "declared outputs that do not exist yet. "
            "Do not rewrite the existing aggregate files to match the partial artifact set. "
            "Your next response should be one concrete `write`-style tool call that creates "
            f"`{primary_missing}`. "
            "Continue one missing declared output at a time until the declared set exists."
            f"{source_hint}"
        )
    if repair_targets:
        primary_target = repair_targets[0]
        source_hint = ""
        if repair_source_paths:
            preview = ", ".join(f"`{path}`" for path in repair_source_paths[:4])
            if len(repair_source_paths) > 4:
                preview += ", ..."
            source_hint = (
                " Use the existing artifact files already on disk as the source of truth: "
                f"{preview}."
            )
        return (
            "Verification already identified the concrete repair target. "
            "Do not restart discovery or reread unrelated references. "
            "Your next response should be one concrete `edit` or `write`-style tool "
            f"call that updates `{primary_target.artifact_path}` to repair "
            f"`{primary_target.failing_reference}`. "
            f"If that reference should stay, create `{primary_target.expected_path}`; "
            "otherwise remove or replace the broken local reference."
            f"{source_hint}"
        )

    fixes = _extract_verification_repairs(dod.evidence, repair_targets=repair_targets)
    if not fixes:
        return None
    return (
        "Verification already identified a concrete failure in the active artifact set. "
        "Reuse that evidence directly, apply one concrete edit or patch, and do not "
        "restart discovery unless a specific missing fact blocks the repair."
    )


def _existing_repair_source_paths(
    dod: DefinitionOfDone,
    *,
    repair_targets: list[VerificationRepairTarget],
    project_root: Path,
) -> list[str]:
    if not repair_targets:
        return []

    candidate_dirs = {
        Path(target.expected_path).parent.resolve(strict=False)
        for target in repair_targets
        if str(target.expected_path).strip()
    }
    candidate_dirs.update(
        Path(target.artifact_path).parent.resolve(strict=False)
        for target in repair_targets
        if str(target.artifact_path).strip()
    )

    paths: list[str] = []
    seen: set[str] = set()
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=24,
    ):
        if expect_directory or not target.is_file():
            continue
        resolved = target.resolve(strict=False)
        if resolved.parent not in candidate_dirs:
            continue
        normalized = str(resolved)
        if normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


def _extract_verification_repair_targets(
    evidence_items: list[VerificationEvidence],
) -> list[VerificationRepairTarget]:
    targets: list[VerificationRepairTarget] = []
    seen: set[tuple[str, str, str]] = set()
    for evidence in evidence_items:
        for candidate in (evidence.stderr, evidence.output, evidence.stdout):
            for problem in _extract_missing_local_html_links(str(candidate)):
                parsed = _parse_missing_local_html_link(problem)
                if parsed is None:
                    continue
                key = (
                    parsed.artifact_path,
                    parsed.failing_reference,
                    parsed.expected_path,
                )
                if key in seen:
                    continue
                seen.add(key)
                targets.append(parsed)
    return targets


def _extract_verification_missing_planned_outputs(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    repair_targets: list[VerificationRepairTarget],
) -> list[Path]:
    if not repair_targets:
        return []

    planned_targets = collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=24,
    )
    missing_declared_outputs: set[str] = set()
    for target, expect_directory in planned_targets:
        normalized_target = target.resolve(strict=False)
        if expect_directory:
            for candidate in collect_missing_declared_html_output_files(
                target=normalized_target,
                project_root=project_root,
            ):
                missing_declared_outputs.add(str(candidate.resolve(strict=False)))
            continue
        if planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=False,
            project_root=project_root,
        ):
            continue
        missing_declared_outputs.add(str(normalized_target))

    missing_paths: list[Path] = []
    seen: set[str] = set()
    for repair_target in repair_targets:
        expected_path = Path(repair_target.expected_path).resolve(strict=False)
        normalized_expected = str(expected_path)
        if normalized_expected not in missing_declared_outputs:
            continue
        if normalized_expected in seen:
            continue
        seen.add(normalized_expected)
        missing_paths.append(expected_path)
    return missing_paths


def _parse_missing_local_html_link(problem: str) -> VerificationRepairTarget | None:
    if " -> " not in problem:
        return None
    broken_target, expected_path = problem.split(" -> ", 1)
    broken_target = broken_target.strip()
    expected_path = expected_path.strip()
    if not broken_target or not expected_path or ":" not in broken_target:
        return None
    artifact_path, failing_reference = broken_target.rsplit(":", 1)
    artifact_path = artifact_path.strip()
    failing_reference = failing_reference.strip()
    if not artifact_path or not failing_reference:
        return None
    return VerificationRepairTarget(
        artifact_path=artifact_path,
        failing_reference=failing_reference,
        expected_path=expected_path,
    )


def _extract_missing_local_html_links(text: str) -> list[str]:
    if "Missing local HTML links:" not in text:
        return []

    problems: list[str] = []
    capture = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "Missing local HTML links:":
            capture = True
            continue
        if not capture:
            continue
        if " -> " not in line:
            continue
        if line not in problems:
            problems.append(line)
    return problems


def _classify_verification_kind(command: str) -> str:
    """Classify the verification command into a summary kind."""

    command_lower = command.lower()
    if "lint" in command_lower or "ruff" in command_lower:
        return "lint"
    if "type" in command_lower or "mypy" in command_lower or "py_compile" in command_lower:
        return "typecheck"
    if "test" in command_lower or "pytest" in command_lower:
        return "test"
    if "build" in command_lower:
        return "build"
    return "runtime"


def merge_usage(target: dict[str, int], update: dict[str, int]) -> None:
    """Merge normalized usage into an existing usage accumulator."""

    for key, value in normalize_usage(update).items():
        target[key] = target.get(key, 0) + value
