"""Per-iteration bookkeeping and prelude control for conversation turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..llm.base import Message, Role
from .context import RuntimeContext
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .tracing import RuntimeTracer
from .workflow_recovery import WorkflowRecoveryController

EventSink = Callable[[AgentEvent], Awaitable[None]]
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


@dataclass(slots=True)
class TurnPreludeDecision:
    """Outcome of per-iteration prelude handling."""

    should_continue: bool = False


class TurnPreludeController:
    """Own iteration bookkeeping, steering drainage, and drift gating."""

    def __init__(
        self,
        context: RuntimeContext,
        *,
        tracer: RuntimeTracer,
        workflow_recovery: WorkflowRecoveryController,
    ) -> None:
        self.context = context
        self.tracer = tracer
        self.workflow_recovery = workflow_recovery

    async def prepare_iteration(
        self,
        *,
        task: str,
        original_task: str | None,
        iterations: int,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        on_user_question: UserQuestionHandler,
        executor,
    ) -> TurnPreludeDecision:
        """Handle iteration bookkeeping before requesting the assistant turn."""

        summary.iterations = iterations
        self.tracer.record("turn.iteration_started", iteration=iterations)

        steering_messages = self.context.drain_steering_messages()
        for steering_message in steering_messages:
            await emit(AgentEvent(type="steering", content=steering_message))
            self.context.session.append(
                Message(
                    role=Role.USER,
                    content=f"[USER INTERRUPTION]: {steering_message}",
                )
            )

        if await self.workflow_recovery.maybe_refresh_plan_for_drift(
            task=original_task or task,
            dod=dod,
            emit=emit,
            summary=summary,
            on_user_question=on_user_question,
            executor=executor,
        ):
            return TurnPreludeDecision(should_continue=True)

        return TurnPreludeDecision()
