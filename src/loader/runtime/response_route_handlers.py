"""Dedicated handlers for assistant response routes."""

from __future__ import annotations

from ..llm.base import Message, Role
from .context import RuntimeContext
from .events import AgentEvent
from .phases import TurnPhase, TurnPhaseTracker
from .repair import ToolCallAnalysis
from .response_route_types import (
    ConfirmationHandler,
    EventSink,
    ResponseRouteAction,
    ResponseRouteContext,
    ResponseRouteDecision,
    UserQuestionHandler,
)
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .turn_completion import TurnCompletionAction, TurnCompletionController


class FinalAnswerRouteHandler:
    """Own final-answer completion behavior."""

    def __init__(self, context: RuntimeContext, tracer: RuntimeTracer) -> None:
        self.context = context
        self.tracer = tracer

    async def handle(
        self,
        *,
        analysis: ToolCallAnalysis,
        context: ResponseRouteContext,
        emit: EventSink,
    ) -> ResponseRouteDecision:
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=analysis.response_content,
        )
        self.context.session.append(assistant_message)
        context.summary.assistant_messages.append(assistant_message)
        final_response = analysis.final_response or analysis.content
        context.summary.final_response = final_response
        self.tracer.record("turn.completed", reason="final_answer")
        await emit(AgentEvent(type="response", content=final_response))
        return ResponseRouteDecision(
            action=ResponseRouteAction.COMPLETE,
            continuation_count=context.continuation_count,
            consecutive_errors=context.consecutive_errors,
        )


class ToolBatchRouteHandler:
    """Own tool-batch routing after assistant-response classification."""

    def __init__(
        self,
        context: RuntimeContext,
        *,
        tracer: RuntimeTracer,
        phase_tracker: TurnPhaseTracker,
        tool_batches: ToolBatchRunner,
    ) -> None:
        self.context = context
        self.tracer = tracer
        self.phase_tracker = phase_tracker
        self.tool_batches = tool_batches

    async def handle(
        self,
        *,
        analysis: ToolCallAnalysis,
        pending_tool_calls_seen: set[str],
        context: ResponseRouteContext,
        emit: EventSink,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
        emit_confirmation,
    ) -> ResponseRouteDecision:
        if analysis.should_stop:
            assistant_message = Message(
                role=Role.ASSISTANT,
                content=analysis.response_content,
            )
            self.context.session.append(assistant_message)
            context.summary.assistant_messages.append(assistant_message)
            final_response = analysis.final_response or analysis.content
            context.summary.final_response = final_response
            if analysis.failure:
                context.summary.failures.append(analysis.failure)
            await emit(AgentEvent(type="response", content=final_response))
            return ResponseRouteDecision(
                action=ResponseRouteAction.COMPLETE,
                continuation_count=context.continuation_count,
                consecutive_errors=context.consecutive_errors,
            )

        await self.phase_tracker.enter(
            TurnPhase.TOOLS,
            emit,
            detail="Executing tool batch",
            reason_code="execute_tool_batch",
        )
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=analysis.response_content,
            tool_calls=list(analysis.tool_calls),
        )
        self.context.session.append(assistant_message)
        context.summary.assistant_messages.append(assistant_message)
        self.tracer.record(
            "assistant.tool_batch",
            tool_count=len(analysis.tool_calls),
            source=analysis.tool_source,
        )

        batch_result = await self.tool_batches.execute_batch(
            tool_calls=list(analysis.tool_calls),
            tool_source=analysis.tool_source,
            pending_tool_calls_seen=pending_tool_calls_seen,
            emit=emit,
            summary=context.summary,
            dod=context.dod,
            executor=context.executor,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            emit_confirmation=emit_confirmation,
            consecutive_errors=context.consecutive_errors,
        )
        if batch_result.halted:
            return ResponseRouteDecision(
                action=ResponseRouteAction.FINALIZE,
                continuation_count=context.continuation_count,
                consecutive_errors=batch_result.consecutive_errors,
                new_actions_taken=batch_result.actions_taken,
                finalize_reason_code="tool_batch_halted",
                finalize_reason_summary="Finalizing after halted tool batch",
            )
        return ResponseRouteDecision(
            action=ResponseRouteAction.CONTINUE,
            continuation_count=context.continuation_count,
            consecutive_errors=batch_result.consecutive_errors,
            new_actions_taken=batch_result.actions_taken,
        )


class TextCompletionRouteHandler:
    """Own text-only completion routing after assistant-response classification."""

    def __init__(self, turn_completion: TurnCompletionController) -> None:
        self.turn_completion = turn_completion

    async def handle(
        self,
        *,
        analysis: ToolCallAnalysis,
        context: ResponseRouteContext,
        emit: EventSink,
    ) -> ResponseRouteDecision:
        completion_decision = await self.turn_completion.handle_text_response(
            content=analysis.content,
            response_content=analysis.response_content,
            task=context.task,
            effective_task=context.effective_task,
            iterations=context.iterations,
            max_iterations=context.max_iterations,
            actions_taken=context.actions_taken,
            continuation_count=context.continuation_count,
            dod=context.dod,
            emit=emit,
            summary=context.summary,
            executor=context.executor,
            rollback_plan=context.rollback_plan,
        )
        if completion_decision.action == TurnCompletionAction.CONTINUE:
            return ResponseRouteDecision(
                action=ResponseRouteAction.CONTINUE,
                continuation_count=completion_decision.continuation_count,
                consecutive_errors=context.consecutive_errors,
            )
        if completion_decision.action == TurnCompletionAction.FINALIZE:
            return ResponseRouteDecision(
                action=ResponseRouteAction.FINALIZE,
                continuation_count=completion_decision.continuation_count,
                consecutive_errors=context.consecutive_errors,
                finalize_reason_code=completion_decision.finalize_reason_code,
                finalize_reason_summary=completion_decision.finalize_reason_summary,
            )
        return ResponseRouteDecision(
            action=ResponseRouteAction.COMPLETE,
            continuation_count=completion_decision.continuation_count,
            consecutive_errors=context.consecutive_errors,
        )
