"""Definition-of-done gating and turn finalization for the runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from .logging import get_runtime_logger
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
    derive_verification_commands,
    ensure_active_verification_attempt,
)
from .events import AgentEvent, TurnSummary
from .evidence_provenance import (
    EvidenceProvenance,
    EvidenceProvenanceStatus,
    summarize_evidence_provenance,
)
from .executor import ToolExecutor
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
        if implementation_item in dod.pending_items:
            dod.pending_items.remove(implementation_item)
            dod.completed_items.append(implementation_item)

        tracked_pending_items = [
            item for item in dod.pending_items if item != "Collect verification evidence"
        ]

        mutating_paths = [path for path in dod.touched_files if path]
        requires_verification = bool(mutating_paths or dod.mutating_actions)
        rlog = get_runtime_logger()
        rlog.completion_check(
            "dod_gate",
            "requires_verification" if requires_verification else "no_verification",
            reason=f"files={mutating_paths[:3]}, actions={len(dod.mutating_actions)}"
            if requires_verification else None,
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

        if not dod.verification_commands:
            dod.verification_commands = derive_verification_commands(
                dod,
                project_root=self.context.project_root,
                task_statement=dod.task_statement,
            )

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
                reason_code=(
                    "verification_command_passed"
                    if evidence.passed
                    else "verification_command_failed"
                ),
                reason_summary=(
                    f"verification passed for `{command}`"
                    if evidence.passed
                    else f"verification failed for `{command}`"
                ),
                evidence_provenance=provenance,
                verification_observations=[observation],
            )
            all_passed = all_passed and evidence.passed
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
    command = evidence.command or "verification"
    return VerificationObservation(
        status=(
            VerificationObservationStatus.PASSED.value
            if evidence.passed
            else VerificationObservationStatus.FAILED.value
        ),
        summary=(
            f"verification passed for `{command}`"
            if evidence.passed
            else f"verification failed for `{command}`"
        ),
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
            summary=(
                f"verification passed for `{command}`"
                if evidence.passed
                else f"verification failed for `{command}`"
            ),
            status=(
                EvidenceProvenanceStatus.SUPPORTS.value
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
