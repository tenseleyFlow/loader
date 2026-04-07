"""Assistant-cycle orchestration for one conversation-loop iteration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from ..agent.reasoning import RollbackPlan
from ..llm.base import Message, Role, ToolCall
from .assistant_turns import AssistantTurnRequester
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import merge_usage
from .phases import TurnPhase, TurnPhaseTracker, TurnTransitionKind
from .repair import ResponseRepairer
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .turn_completion import TurnCompletionAction, TurnCompletionController

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
        agent,
        *,
        tracer: RuntimeTracer,
        phase_tracker: TurnPhaseTracker,
        turn_requester: AssistantTurnRequester,
        repairer: ResponseRepairer,
        tool_batches: ToolBatchRunner,
        turn_completion: TurnCompletionController,
    ) -> None:
        self.agent = agent
        self.tracer = tracer
        self.phase_tracker = phase_tracker
        self.turn_requester = turn_requester
        self.repairer = repairer
        self.tool_batches = tool_batches
        self.turn_completion = turn_completion

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

        content = assistant_turn.content
        response_content = assistant_turn.response_content
        tool_calls = list(assistant_turn.tool_calls)
        pending_tool_calls_seen = set(assistant_turn.pending_tool_calls_seen)

        if not content.strip():
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
            content=content,
            response_content=response_content,
            tool_calls=tool_calls,
            extracted_iterations=extracted_iterations,
            max_extracted_iterations=max_extracted_iterations,
        )
        content = analysis.content
        tool_calls = list(analysis.tool_calls)
        tool_source = analysis.tool_source
        extracted_iterations = analysis.extracted_iterations
        if analysis.clear_stream:
            await self.phase_tracker.enter(
                TurnPhase.REPAIR,
                emit,
                detail="Repairing raw-text tool fallback",
                reason_code="repair_raw_text_tool_fallback",
                kind=TurnTransitionKind.REROUTE,
            )
            await emit(AgentEvent(type="clear_stream"))

        if analysis.is_final_answer:
            assistant_message = Message(role=Role.ASSISTANT, content=response_content)
            self.agent.session.append(assistant_message)
            summary.assistant_messages.append(assistant_message)
            final_response = analysis.final_response or content
            summary.final_response = final_response
            self.tracer.record("turn.completed", reason="final_answer")
            await emit(AgentEvent(type="response", content=final_response))
            return TurnIterationDecision(
                action=TurnIterationAction.COMPLETE,
                continuation_count=continuation_count,
                empty_retry_count=empty_retry_count,
                extracted_iterations=extracted_iterations,
                consecutive_errors=consecutive_errors,
            )

        if tool_calls:
            if analysis.should_stop:
                assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                self.agent.session.append(assistant_message)
                summary.assistant_messages.append(assistant_message)
                final_response = analysis.final_response or content
                summary.final_response = final_response
                if analysis.failure:
                    summary.failures.append(analysis.failure)
                await emit(AgentEvent(type="response", content=final_response))
                return TurnIterationDecision(
                    action=TurnIterationAction.COMPLETE,
                    continuation_count=continuation_count,
                    empty_retry_count=empty_retry_count,
                    extracted_iterations=extracted_iterations,
                    consecutive_errors=consecutive_errors,
                )
            return await self._handle_tool_batch(
                response_content=response_content,
                tool_calls=tool_calls,
                tool_source=tool_source,
                pending_tool_calls_seen=pending_tool_calls_seen,
                extracted_iterations=extracted_iterations,
                continuation_count=continuation_count,
                empty_retry_count=empty_retry_count,
                consecutive_errors=consecutive_errors,
                dod=dod,
                emit=emit,
                summary=summary,
                executor=executor,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                emit_confirmation=emit_confirmation,
            )

        completion_decision = await self.turn_completion.handle_text_response(
            content=content,
            response_content=response_content,
            task=task,
            effective_task=effective_task,
            iterations=iterations,
            max_iterations=max_iterations,
            actions_taken=actions_taken,
            continuation_count=continuation_count,
            dod=dod,
            emit=emit,
            summary=summary,
            executor=executor,
            rollback_plan=rollback_plan,
        )
        if completion_decision.action == TurnCompletionAction.CONTINUE:
            return TurnIterationDecision(
                action=TurnIterationAction.CONTINUE,
                continuation_count=completion_decision.continuation_count,
                empty_retry_count=empty_retry_count,
                extracted_iterations=extracted_iterations,
                consecutive_errors=consecutive_errors,
            )
        if completion_decision.action == TurnCompletionAction.FINALIZE:
            return TurnIterationDecision(
                action=TurnIterationAction.FINALIZE,
                continuation_count=completion_decision.continuation_count,
                empty_retry_count=empty_retry_count,
                extracted_iterations=extracted_iterations,
                consecutive_errors=consecutive_errors,
                finalize_reason_code=completion_decision.finalize_reason_code,
                finalize_reason_summary=completion_decision.finalize_reason_summary,
            )
        return TurnIterationDecision(
            action=TurnIterationAction.COMPLETE,
            continuation_count=completion_decision.continuation_count,
            empty_retry_count=empty_retry_count,
            extracted_iterations=extracted_iterations,
            consecutive_errors=consecutive_errors,
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
        if empty_decision.should_continue and empty_decision.retry_prompt:
            self.agent.session.append(
                Message(
                    role=Role.ASSISTANT,
                    content=empty_decision.retry_prompt,
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
        await emit(AgentEvent(type="response", content=final_response))
        return TurnIterationDecision(
            action=TurnIterationAction.COMPLETE,
            continuation_count=continuation_count,
            empty_retry_count=next_empty_retry_count,
            extracted_iterations=extracted_iterations,
            consecutive_errors=consecutive_errors,
        )

    async def _handle_tool_batch(
        self,
        *,
        response_content: str,
        tool_calls: list[ToolCall],
        tool_source: str,
        pending_tool_calls_seen: set[str],
        extracted_iterations: int,
        continuation_count: int,
        empty_retry_count: int,
        consecutive_errors: int,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
        emit_confirmation,
    ) -> TurnIterationDecision:
        await self.phase_tracker.enter(
            TurnPhase.TOOLS,
            emit,
            detail="Executing tool batch",
            reason_code="execute_tool_batch",
        )
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=response_content,
            tool_calls=tool_calls,
        )
        self.agent.session.append(assistant_message)
        summary.assistant_messages.append(assistant_message)
        self.tracer.record(
            "assistant.tool_batch",
            tool_count=len(tool_calls),
            source=tool_source,
        )

        batch_result = await self.tool_batches.execute_batch(
            tool_calls=tool_calls,
            tool_source=tool_source,
            pending_tool_calls_seen=pending_tool_calls_seen,
            emit=emit,
            summary=summary,
            dod=dod,
            executor=executor,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            emit_confirmation=emit_confirmation,
            consecutive_errors=consecutive_errors,
        )
        if batch_result.halted:
            return TurnIterationDecision(
                action=TurnIterationAction.FINALIZE,
                continuation_count=continuation_count,
                empty_retry_count=empty_retry_count,
                extracted_iterations=extracted_iterations,
                consecutive_errors=batch_result.consecutive_errors,
                new_actions_taken=batch_result.actions_taken,
                finalize_reason_code="tool_batch_halted",
                finalize_reason_summary="Finalizing after halted tool batch",
            )
        return TurnIterationDecision(
            action=TurnIterationAction.CONTINUE,
            continuation_count=continuation_count,
            empty_retry_count=empty_retry_count,
            extracted_iterations=extracted_iterations,
            consecutive_errors=batch_result.consecutive_errors,
            new_actions_taken=batch_result.actions_taken,
        )
