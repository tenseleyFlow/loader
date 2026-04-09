"""No-tool text completion orchestration for the conversation runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from ..llm.base import Message, Role
from .completion_policy import CompletionPolicy
from .completion_trace import CompletionTraceEntry
from .context import RuntimeContext
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import TurnFinalizer
from .phases import TurnPhase, TurnPhaseTracker
from .policy_timeline import (
    append_policy_timeline_entry,
    completion_timeline_kind,
)
from .repair import ResponseRepairer
from .rollback import RollbackPlan

EventSink = Callable[[AgentEvent], Awaitable[None]]


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

    def _append_completion_trace_entry(
        self,
        *,
        summary: TurnSummary,
        stage: str,
        outcome: str,
        decision_code: str,
        decision_summary: str,
        evidence_summary: list[str] | None = None,
    ) -> None:
        entry = CompletionTraceEntry(
            stage=stage,
            outcome=outcome,
            decision_code=decision_code,
            decision_summary=decision_summary,
            evidence_summary=list(evidence_summary or []),
        )
        summary.completion_trace.append(entry)
        self.context.session.append_completion_trace_entry(entry)
        append_policy_timeline_entry(
            self.context,
            summary,
            kind=completion_timeline_kind(stage=stage, outcome=outcome),
            reason_code=decision_code,
            reason_summary=decision_summary,
            policy_stage=stage,
            policy_outcome=outcome,
            evidence_summary=evidence_summary,
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
                    else None
                ),
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

        final_response = self.completion_policy.finalize_response_text(
            content=content,
            actions_taken=actions_taken,
        )

        final_message = Message(role=Role.ASSISTANT, content=response_content)
        self.context.session.append(final_message)
        summary.assistant_messages.append(final_message)

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
