"""Assistant-response dispatch for one turn-iteration cycle."""

from __future__ import annotations

from .phases import TurnPhaseTracker
from .repair import ToolCallAnalysis
from .response_route_handlers import (
    FinalAnswerRouteHandler,
    TextCompletionRouteHandler,
    ToolBatchRouteHandler,
)
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
from .turn_completion import TurnCompletionController

__all__ = [
    "AssistantResponseRouter",
    "ResponseRouteAction",
    "ResponseRouteContext",
    "ResponseRouteDecision",
]


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
        self.final_answer_handler = FinalAnswerRouteHandler(agent, tracer)
        self.tool_batch_handler = ToolBatchRouteHandler(
            agent,
            tracer=tracer,
            phase_tracker=phase_tracker,
            tool_batches=tool_batches,
        )
        self.text_completion_handler = TextCompletionRouteHandler(turn_completion)

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
            return await self.final_answer_handler.handle(
                analysis=analysis,
                context=context,
                emit=emit,
            )

        if analysis.tool_calls:
            return await self.tool_batch_handler.handle(
                analysis=analysis,
                pending_tool_calls_seen=pending_tool_calls_seen,
                context=context,
                emit=emit,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                emit_confirmation=emit_confirmation,
            )

        return await self.text_completion_handler.handle(
            analysis=analysis,
            context=context,
            emit=emit,
        )
