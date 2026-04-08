"""No-tool text completion orchestration for the conversation runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from ..llm.base import Message, Role
from .completion_policy import CompletionPolicy
from .context import RuntimeContext
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import TurnFinalizer
from .phases import TurnPhase, TurnPhaseTracker
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
            and continuation_count < cfg.max_continuation_prompts
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
            if continuation_decision.should_continue:
                return TurnCompletionDecision(
                    action=TurnCompletionAction.CONTINUE,
                    continuation_count=continuation_count + 1,
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
        if gate_result.should_continue:
            return TurnCompletionDecision(
                action=TurnCompletionAction.CONTINUE,
                continuation_count=continuation_count,
            )
        final_response = gate_result.final_response

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
