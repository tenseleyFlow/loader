"""Assistant-cycle orchestration for one conversation-loop iteration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ..llm.base import Message, Role
from .assistant_turns import AssistantTurnRequester
from .context import RuntimeContext
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import merge_usage
from .logging import get_runtime_logger
from .phases import TurnPhase, TurnPhaseTracker, TurnTransitionKind
from .policy_timeline import append_policy_timeline_entry
from .repair import ResponseRepairer
from .response_routing import (
    AssistantResponseRouter,
    ResponseRouteAction,
    ResponseRouteContext,
)
from .rollback import RollbackPlan
from .workflow_policy import WorkflowTimelineEntryKind

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


class TurnIterationAction(StrEnum):
    """What the conversation loop should do after one assistant-cycle."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    FINALIZE = "finalize"


@dataclass(slots=True)
class TurnIterationDecision:
    """Structured outcome of one assistant-cycle iteration."""

    action: TurnIterationAction
    continuation_count: int
    empty_retry_count: int
    extracted_iterations: int
    consecutive_errors: int
    new_actions_taken: list[str] = field(default_factory=list)
    finalize_reason_code: str | None = None
    finalize_reason_summary: str | None = None


class TurnIterationController:
    """Own assistant response handling, routing, and tool-batch continuation."""

    def __init__(
        self,
        context: RuntimeContext,
        *,
        phase_tracker: TurnPhaseTracker,
        turn_requester: AssistantTurnRequester,
        repairer: ResponseRepairer,
        response_router: AssistantResponseRouter,
    ) -> None:
        self.context = context
        self.phase_tracker = phase_tracker
        self.turn_requester = turn_requester
        self.repairer = repairer
        self.response_router = response_router

    async def run_iteration(
        self,
        *,
        task: str,
        effective_task: str,
        original_task: str | None,
        effective_max_tokens: int,
        iterations: int,
        max_iterations: int,
        actions_taken: list[str],
        continuation_count: int,
        empty_retry_count: int,
        max_empty_retries: int,
        extracted_iterations: int,
        max_extracted_iterations: int,
        consecutive_errors: int,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
        rollback_plan: RollbackPlan | None,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
        emit_confirmation,
    ) -> TurnIterationDecision:
        """Run one assistant-cycle and return the next loop decision."""

        await self.phase_tracker.enter(
            TurnPhase.ASSISTANT,
            emit,
            detail="Requesting assistant response",
            reason_code="request_assistant_response",
        )
        await emit(AgentEvent(type="thinking"))
        assistant_turn = await self.turn_requester.request_turn(
            emit=emit,
            max_tokens=effective_max_tokens,
        )
        merge_usage(summary.usage, assistant_turn.usage)

        response_content = assistant_turn.response_content
        tool_calls = list(assistant_turn.tool_calls)
        pending_tool_calls_seen = set(assistant_turn.pending_tool_calls_seen)

        rlog = get_runtime_logger()
        rlog.turn_response(
            iteration=iterations,
            content_len=len(assistant_turn.content),
            tool_call_count=len(tool_calls),
            tool_names=[tc.name for tc in tool_calls],
            usage=assistant_turn.usage,
        )

        if not assistant_turn.content.strip() and not tool_calls:
            return await self._handle_empty_response(
                task=task,
                original_task=original_task,
                empty_retry_count=empty_retry_count,
                max_empty_retries=max_empty_retries,
                extracted_iterations=extracted_iterations,
                continuation_count=continuation_count,
                consecutive_errors=consecutive_errors,
                emit=emit,
                summary=summary,
            )

        analysis = self.repairer.analyze_response(
            content=assistant_turn.content,
            response_content=response_content,
            tool_calls=tool_calls,
            extracted_iterations=extracted_iterations,
            max_extracted_iterations=max_extracted_iterations,
        )
        extracted_iterations = analysis.extracted_iterations
        if analysis.clear_stream:
            await self.phase_tracker.enter(
                TurnPhase.REPAIR,
                emit,
                detail="Repairing raw-text tool fallback",
                reason_code="repair_raw_text_tool_fallback",
                kind=TurnTransitionKind.REROUTE,
            )
            if (
                not analysis.should_stop
                and analysis.reason_code
                and analysis.reason_summary
            ):
                append_policy_timeline_entry(
                    self.context,
                    summary,
                    kind=WorkflowTimelineEntryKind.REPAIR_RETRY,
                    reason_code=analysis.reason_code,
                    reason_summary=analysis.reason_summary,
                    policy_stage="raw_text_tool_fallback",
                    policy_outcome="retry",
                )
            await emit(AgentEvent(type="clear_stream"))

        route_decision = await self.response_router.route_response(
            analysis=analysis,
            pending_tool_calls_seen=pending_tool_calls_seen,
            context=ResponseRouteContext(
                task=task,
                effective_task=effective_task,
                iterations=iterations,
                max_iterations=max_iterations,
                actions_taken=actions_taken,
                continuation_count=continuation_count,
                consecutive_errors=consecutive_errors,
                dod=dod,
                summary=summary,
                executor=executor,
                rollback_plan=rollback_plan,
            ),
            emit=emit,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            emit_confirmation=emit_confirmation,
        )
        if route_decision.action == ResponseRouteAction.CONTINUE:
            return TurnIterationDecision(
                action=TurnIterationAction.CONTINUE,
                continuation_count=route_decision.continuation_count,
                empty_retry_count=empty_retry_count,
                extracted_iterations=extracted_iterations,
                consecutive_errors=route_decision.consecutive_errors,
                new_actions_taken=route_decision.new_actions_taken,
            )
        if route_decision.action == ResponseRouteAction.FINALIZE:
            return TurnIterationDecision(
                action=TurnIterationAction.FINALIZE,
                continuation_count=route_decision.continuation_count,
                empty_retry_count=empty_retry_count,
                extracted_iterations=extracted_iterations,
                consecutive_errors=route_decision.consecutive_errors,
                new_actions_taken=route_decision.new_actions_taken,
                finalize_reason_code=route_decision.finalize_reason_code,
                finalize_reason_summary=route_decision.finalize_reason_summary,
            )
        return TurnIterationDecision(
            action=TurnIterationAction.COMPLETE,
            continuation_count=route_decision.continuation_count,
            empty_retry_count=empty_retry_count,
            extracted_iterations=extracted_iterations,
            consecutive_errors=route_decision.consecutive_errors,
            new_actions_taken=route_decision.new_actions_taken,
        )

    async def _handle_empty_response(
        self,
        *,
        task: str,
        original_task: str | None,
        empty_retry_count: int,
        max_empty_retries: int,
        extracted_iterations: int,
        continuation_count: int,
        consecutive_errors: int,
        emit: EventSink,
        summary: TurnSummary,
    ) -> TurnIterationDecision:
        await self.phase_tracker.enter(
            TurnPhase.REPAIR,
            emit,
            detail="Repairing empty assistant response",
            reason_code="repair_empty_response",
            kind=TurnTransitionKind.RETRY,
        )
        next_empty_retry_count = empty_retry_count + 1
        empty_decision = self.repairer.handle_empty_response(
            task=task,
            original_task=original_task,
            empty_retry_count=next_empty_retry_count,
            max_empty_retries=max_empty_retries,
        )
        if empty_decision.should_continue and empty_decision.retry_message:
            if empty_decision.reason_code and empty_decision.reason_summary:
                append_policy_timeline_entry(
                    self.context,
                    summary,
                    kind=WorkflowTimelineEntryKind.REPAIR_RETRY,
                    reason_code=empty_decision.reason_code,
                    reason_summary=empty_decision.reason_summary,
                    policy_stage="empty_response",
                    policy_outcome="retry",
                )
            self.context.session.append(
                Message(
                    role=Role.USER,
                    content=empty_decision.retry_message,
                )
            )
            return TurnIterationDecision(
                action=TurnIterationAction.CONTINUE,
                continuation_count=continuation_count,
                empty_retry_count=next_empty_retry_count,
                extracted_iterations=extracted_iterations,
                consecutive_errors=consecutive_errors,
            )

        final_response = empty_decision.final_response or ""
        summary.final_response = final_response
        if empty_decision.failure:
            summary.failures.append(empty_decision.failure)
        if empty_decision.reason_code and empty_decision.reason_summary:
            append_policy_timeline_entry(
                self.context,
                summary,
                kind=WorkflowTimelineEntryKind.REPAIR_FAIL,
                reason_code=empty_decision.reason_code,
                reason_summary=empty_decision.reason_summary,
                policy_stage="empty_response",
                policy_outcome="failed",
            )
        await emit(AgentEvent(type="response", content=final_response))
        return TurnIterationDecision(
            action=TurnIterationAction.COMPLETE,
            continuation_count=continuation_count,
            empty_retry_count=next_empty_retry_count,
            extracted_iterations=extracted_iterations,
            consecutive_errors=consecutive_errors,
        )
