"""Per-iteration bookkeeping and prelude control for conversation turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..llm.base import Message, Role
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .tracing import RuntimeTracer
from .workflow_recovery import WorkflowRecoveryController

EventSink = Callable[[AgentEvent], Awaitable[None]]
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None

_ACTION_KEYWORDS = (
    "create",
    "write",
    "make",
    "run",
    "execute",
    "build",
    "install",
    "delete",
    "remove",
    "add",
    "edit",
    "modify",
    "update",
    "fix",
)


@dataclass(slots=True)
class TurnPreludeDecision:
    """Outcome of per-iteration prelude handling."""

    should_continue: bool = False


class TurnPreludeController:
    """Own iteration bookkeeping, steering drainage, and drift gating."""

    def __init__(
        self,
        agent,
        *,
        tracer: RuntimeTracer,
        workflow_recovery: WorkflowRecoveryController,
    ) -> None:
        self.agent = agent
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

        if self._should_seed_action_bracket(task=task, iterations=iterations):
            self.agent.session.append(Message(role=Role.ASSISTANT, content="["))

        steering_messages = self.agent._drain_steering_queue()
        for steering_message in steering_messages:
            await emit(AgentEvent(type="steering", content=steering_message))
            self.agent.session.append(
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

    def _should_seed_action_bracket(self, *, task: str, iterations: int) -> bool:
        """Return whether to preserve the legacy action-seed hint."""

        if iterations != 1 or len(self.agent.messages) != 1:
            return False
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in _ACTION_KEYWORDS)
