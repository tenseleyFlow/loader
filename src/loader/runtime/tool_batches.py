"""Tool-batch execution and recovery bookkeeping for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..llm.base import ToolCall
from .context import RuntimeContext
from .dod import DefinitionOfDone, DefinitionOfDoneStore, record_successful_tool_call
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutionState, ToolExecutor
from .tool_batch_checks import ToolBatchConfidenceGate, ToolBatchVerificationGate
from .tool_batch_recovery import ToolBatchRecoveryController
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
        *,
        confidence_gate: ToolBatchConfidenceGate | None = None,
        recovery_controller: ToolBatchRecoveryController | None = None,
        verification_gate: ToolBatchVerificationGate | None = None,
    ) -> None:
        self.context = context
        self.dod_store = dod_store
        self.confidence_gate = confidence_gate or ToolBatchConfidenceGate(context)
        self.recovery_controller = recovery_controller or ToolBatchRecoveryController(context)
        self.verification_gate = verification_gate or ToolBatchVerificationGate(context)

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
                should_skip = await self.confidence_gate.should_skip(
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
                recovery_result = await self.recovery_controller.build_follow_up(
                    tool_call=tool_call,
                    outcome=outcome,
                    emit=emit,
                )
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

            should_continue = await self.verification_gate.should_continue(
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

    async def _record_successful_execution(
        self,
        *,
        tool_call: ToolCall,
        outcome,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
    ) -> str | None:
        """Update DoD bookkeeping after a successful tool execution."""

        record_successful_tool_call(dod, tool_call)
        if tool_call.name == "TodoWrite" and outcome.registry_result is not None:
            new_todos = outcome.registry_result.metadata.get("new_todos", [])
            if isinstance(new_todos, list):
                sync_todos_to_definition_of_done(dod, new_todos)
        self.dod_store.save(dod)
        self.context.recovery_context = None
        return None
