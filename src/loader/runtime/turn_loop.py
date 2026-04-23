"""Main turn-loop orchestration for the conversation runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .context import RuntimeContext
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .logging import get_runtime_logger
from .rollback import RollbackPlan
from .turn_iteration import TurnIterationAction, TurnIterationController
from .turn_preamble import TurnPreludeController

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = (
    Callable[[str, str, str, dict[str, Any] | None], Awaitable[bool]] | None
)
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


@dataclass(slots=True)
class TurnLoopExit:
    """Structured reason for leaving the main turn loop."""

    reason_code: str
    reason_summary: str


@dataclass(slots=True)
class TurnLoopState:
    """Mutable iteration bookkeeping for one runtime turn."""

    iterations: int = 0
    actions_taken: list[str] = field(default_factory=list)
    continuation_count: int = 0
    empty_retry_count: int = 0
    extracted_iterations: int = 0
    consecutive_errors: int = 0
    max_empty_retries: int = 2
    max_extracted_iterations: int = 3


class TurnLoopController:
    """Own the main iteration loop that coordinates prelude and assistant cycles."""

    def __init__(
        self,
        context: RuntimeContext,
        *,
        turn_preamble: TurnPreludeController,
        turn_iteration: TurnIterationController,
    ) -> None:
        self.context = context
        self.turn_preamble = turn_preamble
        self.turn_iteration = turn_iteration

    async def run_loop(
        self,
        *,
        task: str,
        effective_task: str,
        original_task: str | None,
        effective_max_tokens: int,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        executor: ToolExecutor,
        rollback_plan: RollbackPlan | None,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
        emit_confirmation,
    ) -> TurnLoopExit:
        """Run the bounded main turn loop and report how it finished."""

        state = TurnLoopState()
        rlog = get_runtime_logger()
        while state.iterations < self.context.config.max_iterations:
            state.iterations += 1
            rlog.turn_start(
                iteration=state.iterations,
                message_count=len(self.context.session.messages),
                task=effective_task,
            )
            prelude_decision = await self.turn_preamble.prepare_iteration(
                task=task,
                original_task=original_task,
                iterations=state.iterations,
                dod=dod,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                executor=executor,
            )
            if prelude_decision.should_continue:
                continue

            iteration_decision = await self.turn_iteration.run_iteration(
                task=task,
                effective_task=effective_task,
                original_task=original_task,
                effective_max_tokens=effective_max_tokens,
                iterations=state.iterations,
                max_iterations=self.context.config.max_iterations,
                actions_taken=state.actions_taken,
                continuation_count=state.continuation_count,
                empty_retry_count=state.empty_retry_count,
                max_empty_retries=state.max_empty_retries,
                extracted_iterations=state.extracted_iterations,
                max_extracted_iterations=state.max_extracted_iterations,
                consecutive_errors=state.consecutive_errors,
                dod=dod,
                emit=emit,
                summary=summary,
                executor=executor,
                rollback_plan=rollback_plan,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                emit_confirmation=emit_confirmation,
            )
            state.continuation_count = iteration_decision.continuation_count
            state.empty_retry_count = iteration_decision.empty_retry_count
            state.extracted_iterations = iteration_decision.extracted_iterations
            state.consecutive_errors = iteration_decision.consecutive_errors
            state.actions_taken.extend(iteration_decision.new_actions_taken)
            rlog.turn_decision(
                iteration=state.iterations,
                action=iteration_decision.action.value,
                continuation_count=state.continuation_count,
                consecutive_errors=state.consecutive_errors,
                reason=iteration_decision.finalize_reason_code,
            )
            if iteration_decision.action == TurnIterationAction.CONTINUE:
                continue
            if iteration_decision.action == TurnIterationAction.FINALIZE:
                exit = TurnLoopExit(
                    reason_code=iteration_decision.finalize_reason_code
                    or "turn_complete",
                    reason_summary=iteration_decision.finalize_reason_summary
                    or "Finalizing completed turn",
                )
                rlog.loop_exit(state.iterations, exit.reason_code, exit.reason_summary)
                return exit
            break
        else:
            # Loop exhausted max_iterations without breaking — notify user
            await emit(
                AgentEvent(
                    type="error",
                    content=(
                        f"Reached iteration limit ({self.context.config.max_iterations}). "
                        "Stopping — the work above may be incomplete."
                    ),
                )
            )
            exit = TurnLoopExit(
                reason_code="max_iterations_reached",
                reason_summary=f"Stopped after {state.iterations} iterations (limit reached)",
            )
            rlog.loop_exit(state.iterations, exit.reason_code, exit.reason_summary)
            return exit

        exit = TurnLoopExit(
            reason_code="turn_complete",
            reason_summary="Finalizing completed turn",
        )
        rlog.loop_exit(state.iterations, exit.reason_code, exit.reason_summary)
        return exit
