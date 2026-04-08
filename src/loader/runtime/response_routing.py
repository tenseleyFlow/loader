"""Assistant-response routing for one turn-iteration cycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ..llm.base import Message, Role
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .phases import TurnPhase, TurnPhaseTracker
from .repair import ToolCallAnalysis
from .rollback import RollbackPlan
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .turn_completion import TurnCompletionAction, TurnCompletionController

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


class ResponseRouteAction(StrEnum):
    """What should happen after routing one assistant response."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    FINALIZE = "finalize"


@dataclass(slots=True)
class ResponseRouteDecision:
    """Structured response-routing outcome for one assistant cycle."""

    action: ResponseRouteAction
    continuation_count: int
    consecutive_errors: int
    new_actions_taken: list[str] = field(default_factory=list)
    finalize_reason_code: str | None = None
    finalize_reason_summary: str | None = None


@dataclass(slots=True)
class ResponseRouteContext:
    """Loop state needed to route a classified assistant response."""

    task: str
    effective_task: str
    iterations: int
    max_iterations: int
    actions_taken: list[str]
    continuation_count: int
    consecutive_errors: int
    dod: DefinitionOfDone
    summary: TurnSummary
    executor: ToolExecutor
    rollback_plan: RollbackPlan | None


class AssistantResponseRouter:
    """Own response-policy routing after repair/classification."""

    def __init__(
        self,
        agent,
        *,
        tracer: RuntimeTracer,
        phase_tracker: TurnPhaseTracker,
        tool_batches: ToolBatchRunner,
        turn_completion: TurnCompletionController,
    ) -> None:
        self.agent = agent
        self.tracer = tracer
        self.phase_tracker = phase_tracker
        self.tool_batches = tool_batches
        self.turn_completion = turn_completion

    async def route_response(
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
        """Route one classified assistant response through the next policy seam."""

        if analysis.is_final_answer:
            return await self._complete_final_answer(
                analysis=analysis,
                context=context,
                emit=emit,
            )

        if analysis.tool_calls:
            return await self._route_tool_batch(
                analysis=analysis,
                pending_tool_calls_seen=pending_tool_calls_seen,
                context=context,
                emit=emit,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                emit_confirmation=emit_confirmation,
            )

        return await self._route_text_completion(
            analysis=analysis,
            context=context,
            emit=emit,
        )

    async def _complete_final_answer(
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
        self.agent.session.append(assistant_message)
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

    async def _route_tool_batch(
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
            self.agent.session.append(assistant_message)
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
        self.agent.session.append(assistant_message)
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

    async def _route_text_completion(
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
