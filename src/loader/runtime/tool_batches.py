"""Tool-batch execution and recovery bookkeeping for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..agent.recovery import RecoveryContext, format_failure_message, format_recovery_prompt
from ..llm.base import Message, Role, ToolCall
from .context import RuntimeContext
from .dod import DefinitionOfDone, DefinitionOfDoneStore, record_successful_tool_call
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutionState, ToolExecutor
from .workflow import sync_todos_to_definition_of_done

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


@dataclass
class ToolBatchResult:
    """Outcome of running one assistant-proposed tool batch."""

    actions_taken: list[str] = field(default_factory=list)
    consecutive_errors: int = 0
    halted: bool = False
    final_response: str = ""


class ToolBatchRunner:
    """Owns tool-batch execution, recovery, and post-tool bookkeeping."""

    def __init__(
        self,
        context: RuntimeContext,
        dod_store: DefinitionOfDoneStore,
    ) -> None:
        self.context = context
        self.dod_store = dod_store

    async def execute_batch(
        self,
        *,
        tool_calls: list[ToolCall],
        tool_source: str,
        pending_tool_calls_seen: set[str],
        emit: EventSink,
        summary: TurnSummary,
        dod: DefinitionOfDone,
        executor: ToolExecutor,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
        emit_confirmation,
        consecutive_errors: int,
    ) -> ToolBatchResult:
        """Run one assistant tool batch through the shared executor seam."""

        result = ToolBatchResult(consecutive_errors=consecutive_errors)

        for tool_call in tool_calls:
            cfg = self.context.config.reasoning

            if cfg.confidence_scoring:
                should_skip = await self._handle_confidence_gate(
                    tool_call=tool_call,
                    emit=emit,
                )
                if should_skip:
                    continue

            if tool_call.id not in pending_tool_calls_seen:
                await emit(
                    AgentEvent(
                        type="tool_call",
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments,
                        phase="assistant",
                    )
                )

            result.actions_taken.append(
                f"{tool_call.name}: {str(tool_call.arguments)[:100]}"
            )

            outcome = await executor.execute_tool_call(
                tool_call,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                emit_confirmation=emit_confirmation,
                source=tool_source,
            )
            if (
                outcome.rollback_action is not None
                and self.context.config.reasoning.show_rollback_plan
            ):
                await emit(
                    AgentEvent(
                        type="rollback",
                        content=(
                            f"Rollback tracked: {outcome.rollback_action.description}"
                        ),
                        rollback_action=outcome.rollback_action,
                    )
                )

            if (
                outcome.state == ToolExecutionState.EXECUTED
                and outcome.is_error
                and self.context.config.auto_recover
            ):
                recovery_result = await self._handle_recovery(tool_call, outcome, emit)
                if recovery_result is not None:
                    summary.tool_result_messages.append(recovery_result)
                    self.context.session.append(recovery_result)
                    continue

            if outcome.state == ToolExecutionState.EXECUTED and not outcome.is_error:
                loop_response = await self._record_successful_execution(
                    tool_call=tool_call,
                    outcome=outcome,
                    dod=dod,
                    emit=emit,
                    summary=summary,
                )
                if loop_response is not None:
                    result.halted = True
                    result.final_response = loop_response
                    return result

            if outcome.is_error:
                result.consecutive_errors += 1
            else:
                result.consecutive_errors = 0

            await emit(
                AgentEvent(
                    type="tool_result",
                    content=outcome.event_content,
                    tool_name=tool_call.name,
                    is_error=outcome.is_error,
                    phase="assistant",
                )
            )

            should_continue = await self._run_post_tool_verification(
                tool_call=tool_call,
                outcome=outcome,
                emit=emit,
            )
            if should_continue:
                continue

            self.context.session.append(outcome.message)
            summary.tool_result_messages.append(outcome.message)

        if result.consecutive_errors >= 3:
            final_response = (
                "I ran into some issues. "
                "Let me know if you'd like me to try a different approach."
            )
            summary.final_response = final_response
            summary.failures.append("three consecutive tool errors")
            await emit(AgentEvent(type="response", content=final_response))
            result.halted = True
            result.final_response = final_response

        return result

    async def _handle_confidence_gate(
        self,
        *,
        tool_call: ToolCall,
        emit: EventSink,
    ) -> bool:
        """Emit confidence scoring and optionally skip low-confidence actions."""

        cfg = self.context.config.reasoning
        context = "\n".join(
            message.content[:500]
            for message in self.context.messages[-5:]
            if message.content
        )
        confidence = await self.context.legacy.assess_confidence(
            tool_call.name,
            tool_call.arguments,
            context,
        )
        await emit(
            AgentEvent(
                type="confidence",
                content=f"Confidence: {confidence.level.name} ({confidence.score}/5)",
                confidence=confidence,
                tool_name=tool_call.name,
            )
        )
        if confidence.score >= cfg.min_confidence_for_action:
            return False

        low_confidence_message = (
            "[LOW CONFIDENCE WARNING] The planned action has low confidence "
            f"({confidence.level.name}).\n"
            f"Reasoning: {confidence.reasoning}\n"
            f"Risks: {', '.join(confidence.risks)}\n"
            "Consider an alternative approach or gather more information first."
        )
        self.context.session.append(Message(role=Role.USER, content=low_confidence_message))
        return True

    async def _record_successful_execution(
        self,
        *,
        tool_call: ToolCall,
        outcome,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
    ) -> str | None:
        """Update DoD and loop safeguards after a successful tool execution."""

        record_successful_tool_call(dod, tool_call)
        if tool_call.name == "TodoWrite" and outcome.registry_result is not None:
            new_todos = outcome.registry_result.metadata.get("new_todos", [])
            if isinstance(new_todos, list):
                sync_todos_to_definition_of_done(dod, new_todos)
        self.dod_store.save(dod)
        self.context.legacy.set_recovery_context(None)

        is_loop, loop_description = self.context.safeguards.detect_loop()
        if not is_loop:
            return None

        final_response = (
            "I noticed I was repeating the same actions. "
            "Let me know what you'd like me to do differently."
        )
        summary.final_response = final_response
        summary.failures.append(loop_description)
        loop_message = Message(role=Role.ASSISTANT, content=final_response)
        self.context.session.append(loop_message)
        summary.assistant_messages.append(loop_message)
        await emit(
            AgentEvent(
                type="error",
                content=(
                    f"Loop detected: {loop_description}. "
                    "Stopping to prevent repetitive behavior."
                ),
            )
        )
        await emit(AgentEvent(type="response", content=final_response))
        return final_response

    async def _run_post_tool_verification(
        self,
        *,
        tool_call: ToolCall,
        outcome,
        emit: EventSink,
    ) -> bool:
        """Run optional post-tool verification and return whether to continue."""

        cfg = self.context.config.reasoning
        if not (
            cfg.verification
            and outcome.state == ToolExecutionState.EXECUTED
            and not outcome.is_error
        ):
            return False

        verification = await self.context.legacy.verify_action(
            tool_call.name,
            tool_call.arguments,
            outcome.result_output,
        )
        await emit(
            AgentEvent(
                type="verification",
                content=f"Verified: {verification.verified}",
                verification=verification,
                tool_name=tool_call.name,
            )
        )
        if not verification.verified or not verification.needs_correction:
            return False

        correction_message = (
            "[VERIFICATION FAILED] The action did not "
            "produce expected results.\n"
            f"Discrepancies: {', '.join(verification.discrepancies)}\n"
            f"Suggestion: {verification.correction_suggestion}"
        )
        self.context.session.append(Message(role=Role.USER, content=correction_message))
        return True

    async def _handle_recovery(
        self,
        tool_call: ToolCall,
        outcome,
        emit: EventSink,
    ) -> Message | None:
        """Generate a recovery follow-up after an executed tool failure."""

        recovery_context = self.context.legacy.get_recovery_context()
        if recovery_context is None:
            recovery_context = RecoveryContext(
                original_tool=tool_call.name,
                original_args=tool_call.arguments,
                max_retries=self.context.config.max_recovery_attempts,
            )
            self.context.legacy.set_recovery_context(recovery_context)

        if recovery_context.is_similar_attempt(
            tool_call.name,
            tool_call.arguments,
        ):
            await emit(
                AgentEvent(
                    type="error",
                    content=(
                        "Loop detected: already tried a similar command. "
                        "Try a DIFFERENT approach (e.g., read a config file first)."
                    ),
                    tool_name=tool_call.name,
                )
            )
        else:
            recovery_context.add_attempt(
                tool_call.name,
                tool_call.arguments,
                outcome.result_output,
            )

        if recovery_context.can_retry():
            attempt_number = len(recovery_context.attempts)
            await emit(
                AgentEvent(
                    type="recovery",
                    content=(
                        "Tool failed, attempting recovery "
                        f"({attempt_number}/{recovery_context.max_retries})"
                    ),
                    tool_name=tool_call.name,
                    recovery_attempt=attempt_number,
                )
            )
            recovery_prompt = format_recovery_prompt(
                recovery_context,
                tool_call.name,
                tool_call.arguments,
                outcome.result_output,
            )
            return Message.tool_result_message(
                tool_call_id=tool_call.id,
                display_content=recovery_prompt,
                result_content=recovery_prompt,
                is_error=True,
            )

        failure_message = format_failure_message(recovery_context)
        await emit(
            AgentEvent(
                type="error",
                content=failure_message,
                tool_name=tool_call.name,
            )
        )
        self.context.legacy.set_recovery_context(None)
        return Message.tool_result_message(
            tool_call_id=tool_call.id,
            display_content=(f"Observation [{tool_call.name}]: Error: {failure_message}"),
            result_content=failure_message,
            is_error=True,
        )
