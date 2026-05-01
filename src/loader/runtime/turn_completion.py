"""No-tool text completion orchestration for the conversation runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..llm.base import Message, Role
from .completion_policy import CompletionPolicy
from .context import RuntimeContext
from .dod import (
    DefinitionOfDone,
    collect_planned_artifact_targets,
    infer_next_output_file,
    planned_artifact_target_satisfied,
)
from .events import AgentEvent, TurnSummary
from .evidence_provenance import EvidenceProvenance
from .executor import ToolExecutor
from .finalization import TurnFinalizer
from .phases import TurnPhase, TurnPhaseTracker
from .policy_timeline import (
    append_policy_timeline_entry,
    completion_timeline_kind,
)
from .repair import ResponseRepairer
from .rollback import RollbackPlan
from .verification_observations import VerificationObservation
from .workflow import (
    effective_pending_todo_items,
    infer_pending_todo_output_target,
    preferred_pending_todo_item,
)

EventSink = Callable[[AgentEvent], Awaitable[None]]

_SPECIAL_DOD_ITEMS = {
    "Complete the requested work",
    "Collect verification evidence",
}
_PROGRESS_INTENT_HINTS = (
    "i'll ",
    "i will ",
    "i am going to ",
    "i'm going to ",
    "let me ",
    "now i'll ",
    "next i'll ",
    "continue by ",
    "continue with ",
)
_COMPLETION_HINTS = (
    "done",
    "completed",
    "finished",
    "all set",
    "verified",
    "successfully completed",
    "everything is done",
)


class TurnCompletionAction(StrEnum):
    """What the runtime should do after evaluating one no-tool text response."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    FINALIZE = "finalize"


@dataclass(slots=True)
class TurnCompletionDecision:
    """Outcome of evaluating one no-tool text response."""

    action: TurnCompletionAction
    continuation_count: int
    finalize_reason_code: str | None = None
    finalize_reason_summary: str | None = None


class TurnCompletionController:
    """Owns no-tool completion policy and DoD gating."""

    def __init__(
        self,
        context: RuntimeContext,
        *,
        repairer: ResponseRepairer,
        completion_policy: CompletionPolicy,
        finalizer: TurnFinalizer,
        phase_tracker: TurnPhaseTracker,
    ) -> None:
        self.context = context
        self.repairer = repairer
        self.completion_policy = completion_policy
        self.finalizer = finalizer
        self.phase_tracker = phase_tracker

    def _record_completion_decision(
        self,
        *,
        summary: TurnSummary,
        decision_code: str,
        decision_summary: str,
    ) -> None:
        summary.completion_decision_code = decision_code
        summary.completion_decision_summary = decision_summary
        self.context.session.update_runtime_state(
            last_completion_decision_code=decision_code,
            last_completion_decision_summary=decision_summary,
        )
        summary.completion_trace = list(self.context.session.completion_trace)

    def _append_completion_trace_entry(
        self,
        *,
        summary: TurnSummary,
        stage: str,
        outcome: str,
        decision_code: str,
        decision_summary: str,
        evidence_summary: list[str] | None = None,
        evidence_provenance: list[EvidenceProvenance] | None = None,
        verification_observations: list[VerificationObservation] | None = None,
    ) -> None:
        append_policy_timeline_entry(
            self.context,
            summary,
            kind=completion_timeline_kind(stage=stage, outcome=outcome),
            reason_code=decision_code,
            reason_summary=decision_summary,
            policy_stage=stage,
            policy_outcome=outcome,
            evidence_summary=evidence_summary,
            evidence_provenance=evidence_provenance,
            verification_observations=verification_observations,
        )

    async def handle_text_response(
        self,
        *,
        content: str,
        response_content: str,
        task: str,
        effective_task: str,
        iterations: int,
        max_iterations: int,
        actions_taken: list[str],
        continuation_count: int,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
        rollback_plan: RollbackPlan | None,
    ) -> TurnCompletionDecision:
        """Handle a no-tool assistant response inside the main turn loop."""

        await self.phase_tracker.enter(
            TurnPhase.COMPLETION,
            emit,
            detail="Checking completion policy",
            reason_code="completion_gate",
        )
        text_loop_decision = await self.completion_policy.maybe_stop_for_text_loop(
            content=content,
            emit=emit,
            summary=summary,
        )
        if text_loop_decision.should_stop:
            self._append_completion_trace_entry(
                summary=summary,
                stage="text_loop",
                outcome="finalize",
                decision_code=text_loop_decision.decision_code,
                decision_summary=text_loop_decision.decision_summary,
            )
            self._record_completion_decision(
                summary=summary,
                decision_code=text_loop_decision.decision_code,
                decision_summary=text_loop_decision.decision_summary,
            )
            return TurnCompletionDecision(
                action=TurnCompletionAction.FINALIZE,
                continuation_count=continuation_count,
                finalize_reason_code="text_loop_bailout",
                finalize_reason_summary="Finalizing after text-loop bailout",
            )

        cfg = self.context.config.reasoning
        self.context.safeguards.record_response(content)
        if (
            cfg.completion_check
            and not dod.mutating_actions
        ):
            continuation_decision = (
                await self.completion_policy.maybe_continue_for_completion(
                    content=content,
                    response_content=response_content,
                    task=effective_task,
                    actions_taken=actions_taken,
                    continuation_count=continuation_count,
                    emit=emit,
                    dod=dod,
                )
            )
            self._append_completion_trace_entry(
                summary=summary,
                stage="continuation_check",
                outcome=(
                    "continue"
                    if continuation_decision.should_continue
                    else "finalize" if continuation_decision.should_finalize else "accept"
                ),
                decision_code=continuation_decision.decision_code,
                decision_summary=continuation_decision.decision_summary,
                evidence_summary=(
                    continuation_decision.completion_check.missing_evidence
                    if continuation_decision.completion_check is not None
                    and not continuation_decision.completion_check.is_complete
                    else None
                ),
                evidence_provenance=continuation_decision.evidence_provenance,
                verification_observations=continuation_decision.verification_observations,
            )
            if continuation_decision.should_continue:
                self._record_completion_decision(
                    summary=summary,
                    decision_code=continuation_decision.decision_code,
                    decision_summary=continuation_decision.decision_summary,
                )
                return TurnCompletionDecision(
                    action=TurnCompletionAction.CONTINUE,
                    continuation_count=continuation_count + 1,
                )
            if continuation_decision.should_finalize:
                final_response = continuation_decision.final_response
                summary.final_response = final_response
                if continuation_decision.failure:
                    summary.failures.append(continuation_decision.failure)
                final_message = Message(role=Role.ASSISTANT, content=final_response)
                self.context.session.append(final_message)
                summary.assistant_messages.append(final_message)
                self._record_completion_decision(
                    summary=summary,
                    decision_code=continuation_decision.decision_code,
                    decision_summary=continuation_decision.decision_summary,
                )
                await emit(
                    AgentEvent(
                        type="error",
                        content=continuation_decision.decision_summary,
                    )
                )
                await emit(AgentEvent(type="response", content=final_response))
                return TurnCompletionDecision(
                    action=TurnCompletionAction.FINALIZE,
                    continuation_count=continuation_count,
                    finalize_reason_code=continuation_decision.decision_code,
                    finalize_reason_summary=continuation_decision.decision_summary,
                )

        progress_messages = list(getattr(self.context.session, "messages", []) or [])
        progress_intent = _build_in_progress_continuation(
            content=content,
            dod=dod,
            project_root=self.context.project_root,
            messages=progress_messages,
        )
        if progress_intent is not None:
            assistant_message = Message(role=Role.ASSISTANT, content=response_content)
            self.context.session.append(assistant_message)
            summary.assistant_messages.append(assistant_message)
            if (
                progress_intent.target is not None
                and continuation_count == 0
                and not _recent_concrete_target_prompt(
                    progress_messages,
                    target=progress_intent.target,
                )
            ):
                self._append_completion_trace_entry(
                    summary=summary,
                    stage="continuation_check",
                    outcome="continue",
                    decision_code="in_progress_transition_continue",
                    decision_summary=(
                        "continued to let the assistant finish the concrete next "
                        "planned step without interrupting it yet"
                    ),
                )
                self._record_completion_decision(
                    summary=summary,
                    decision_code="in_progress_transition_continue",
                    decision_summary=(
                        "continued to let the assistant finish the concrete next "
                        "planned step without interrupting it yet"
                    ),
                )
                return TurnCompletionDecision(
                    action=TurnCompletionAction.CONTINUE,
                    continuation_count=continuation_count + 1,
                )

            self.context.session.append(
                Message(role=Role.USER, content=progress_intent.prompt)
            )
            self._append_completion_trace_entry(
                summary=summary,
                stage="continuation_check",
                outcome="continue",
                decision_code="in_progress_transition_continue",
                decision_summary=(
                    "continued because the assistant described the next planned step "
                    "without executing it yet"
                ),
            )
            self._record_completion_decision(
                summary=summary,
                decision_code="in_progress_transition_continue",
                decision_summary=(
                    "continued because the assistant described the next planned step "
                    "without executing it yet"
                ),
            )
            return TurnCompletionDecision(
                action=TurnCompletionAction.CONTINUE,
                continuation_count=continuation_count + 1,
            )

        final_response = self.completion_policy.finalize_response_text(
            content=content,
            actions_taken=actions_taken,
        )

        gate_result = await self.finalizer.run_definition_of_done_gate(
            dod=dod,
            candidate_response=final_response,
            emit=emit,
            summary=summary,
            executor=executor,
        )
        self._append_completion_trace_entry(
            summary=summary,
            stage="definition_of_done",
            outcome="continue" if gate_result.should_continue else "complete",
            decision_code=gate_result.reason_code,
            decision_summary=gate_result.reason_summary,
            evidence_provenance=gate_result.evidence_provenance,
            verification_observations=gate_result.verification_observations,
        )
        if gate_result.should_continue:
            self._record_completion_decision(
                summary=summary,
                decision_code=gate_result.reason_code,
                decision_summary=gate_result.reason_summary,
            )
            return TurnCompletionDecision(
                action=TurnCompletionAction.CONTINUE,
                continuation_count=continuation_count,
            )
        final_response = gate_result.final_response
        final_message = Message(role=Role.ASSISTANT, content=response_content)
        self.context.session.append(final_message)
        summary.assistant_messages.append(final_message)
        self._record_completion_decision(
            summary=summary,
            decision_code=gate_result.reason_code,
            decision_summary=gate_result.reason_summary,
        )

        if rollback_plan and rollback_plan.actions:
            await emit(
                AgentEvent(
                    type="rollback_summary",
                    content=f"Rollback plan: {len(rollback_plan.actions)} action(s) tracked",
                    rollback_plan=rollback_plan,
                )
            )

        summary.final_response = final_response
        await emit(AgentEvent(type="response", content=final_response))
        return TurnCompletionDecision(
            action=TurnCompletionAction.COMPLETE,
            continuation_count=continuation_count,
        )


@dataclass(frozen=True, slots=True)
class InProgressContinuation:
    prompt: str
    target: Path | None


def _build_in_progress_continuation(
    *,
    content: str,
    dod: DefinitionOfDone,
    project_root: Path,
    messages: list[object],
) -> InProgressContinuation | None:
    if not _looks_like_progress_intent(content):
        return None

    missing_artifact = _next_missing_planned_artifact(
        dod,
        project_root=project_root,
        messages=messages,
    )
    next_pending = preferred_pending_todo_item(
        dod,
        project_root=project_root,
        missing_artifact=missing_artifact,
    )
    if not next_pending and missing_artifact is None:
        return None

    target = _preferred_progress_target(
        dod,
        next_pending=next_pending,
        missing_artifact=missing_artifact,
        project_root=project_root,
        messages=messages,
    )
    if target is not None:
        return InProgressContinuation(
            prompt=(
                "[CONTINUE CURRENT STEP]\n"
                "You just described the next planned step, but the concrete output is not on disk yet. "
                f"Respond with one concrete `write` or `edit`-style tool call that creates or updates `{target}` now. "
                "Do not summarize, verify, or restart discovery first."
            ),
            target=target,
        )

    if next_pending:
        return InProgressContinuation(
            prompt=(
                "[CONTINUE CURRENT STEP]\n"
                "You just described the next planned step, but it has not been executed yet. "
                f"Continue with `{next_pending}` now by emitting one concrete tool call instead of another narration, summary, or verification claim."
            ),
            target=None,
        )
    return None


def _looks_like_progress_intent(content: str) -> bool:
    text = content.lower().strip()
    if not text or "?" in text:
        return False
    if any(marker in text for marker in _COMPLETION_HINTS):
        return False
    return any(marker in text for marker in _PROGRESS_INTENT_HINTS)


def _next_missing_planned_artifact(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    messages: list[object],
) -> tuple[Path, bool] | None:
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if not planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=expect_directory,
            project_root=project_root,
        ):
            return target, expect_directory

    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if not expect_directory or not target.is_dir():
            continue
        next_output_file, _ = infer_next_output_file(
            target=target,
            project_root=project_root,
            messages=list(messages or []),
        )
        if next_output_file is not None and not next_output_file.exists():
            return next_output_file, False
    return None


def _preferred_progress_target(
    dod: DefinitionOfDone,
    *,
    next_pending: str | None,
    missing_artifact: tuple[Path, bool] | None,
    project_root: Path,
    messages: list[object],
) -> Path | None:
    pending_items = [
        item
        for item in effective_pending_todo_items(
            dod,
            project_root=project_root,
        )
        if item not in _SPECIAL_DOD_ITEMS
    ]
    if next_pending and next_pending in pending_items:
        pending_target = infer_pending_todo_output_target(
            dod,
            next_pending,
            project_root=project_root,
        )
        if pending_target is not None and not pending_target.exists():
            return pending_target

    if missing_artifact is None:
        return None

    target, expect_directory = missing_artifact
    if not expect_directory:
        return target

    next_output_file, _ = infer_next_output_file(
        target=target,
        project_root=project_root,
        messages=list(messages or []),
    )
    if next_output_file is not None:
        return next_output_file
    return None


def _recent_concrete_target_prompt(
    messages: list[object],
    *,
    target: Path,
) -> bool:
    target = target.expanduser().resolve(strict=False)
    target_text = str(target)
    target_name = target.name
    for message in reversed(messages[-6:]):
        role = getattr(message, "role", None)
        if getattr(role, "value", role) != "user":
            continue
        content = str(getattr(message, "content", "") or "")
        if not content:
            continue
        if "[CONTINUE CURRENT STEP]" not in content and "[USER INTERRUPTION]" not in content and "[EMPTY ASSISTANT RESPONSE]" not in content:
            continue
        if target_text not in content and target_name not in content:
            continue
        if (
            "concrete mutation tool call" in content
            or "Resume by creating" in content
            or "Emit this tool shape now" in content
        ):
            return True
    return False
