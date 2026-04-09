"""Runtime-owned task decomposition orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..llm.base import Message, Role
from .bootstrap import RuntimeBootstrapSource
from .conversation import ConfirmationHandler, EventSink, UserQuestionHandler
from .deliberation import DECOMPOSITION_PROMPT, parse_decomposition
from .events import AgentEvent

RunTaskCallback = Callable[
    [
        str,
        EventSink,
        ConfirmationHandler,
        UserQuestionHandler,
        str | None,
        str | None,
    ],
    Awaitable[str],
]


class DecompositionTurnRunner:
    """Own runtime-managed task decomposition and subtask execution."""

    def __init__(
        self,
        source: RuntimeBootstrapSource,
        *,
        run_task: RunTaskCallback,
    ) -> None:
        self.source = source
        self.run_task = run_task

    async def run(
        self,
        task: str,
        emit: EventSink,
        *,
        on_confirmation: ConfirmationHandler = None,
        on_user_question: UserQuestionHandler = None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> str:
        """Run one decomposition flow or fall back to the direct task path."""

        await emit(AgentEvent(type="thinking", content="Analyzing task complexity..."))
        decomposition = await self._decompose_task(task)

        if len(decomposition.subtasks) <= 1:
            self.source.session.append(Message(role=Role.USER, content=task))
            return await self.run_task(
                task,
                emit,
                on_confirmation,
                on_user_question,
                requested_mode,
                original_task,
            )

        await emit(
            AgentEvent(
                type="decomposition",
                content=decomposition.to_prompt(),
                decomposition=decomposition,
            )
        )

        while not decomposition.is_complete() and not decomposition.has_failures():
            subtask = decomposition.next_subtask()
            if subtask is None:
                break

            subtask.status = "in_progress"
            await emit(
                AgentEvent(
                    type="subtask",
                    content=f"{decomposition.progress_str()} {subtask.description}",
                    subtask=subtask,
                )
            )

            self.source.session.append(
                Message(
                    role=Role.USER,
                    content=(
                        f"Execute this subtask: {subtask.description}\n\n"
                        f"Verification: {subtask.verification}"
                    ),
                )
            )
            subtask_response = await self.run_task(
                subtask.description,
                emit,
                on_confirmation,
                on_user_question,
                None,
                original_task,
            )

            if "error" in subtask_response.lower() or "failed" in subtask_response.lower():
                decomposition.mark_failed(subtask.id, subtask_response)
                if decomposition.can_retry(subtask.id):
                    decomposition.reset_for_retry(subtask.id)
                    await emit(
                        AgentEvent(
                            type="subtask",
                            content=f"Retrying subtask: {subtask.description}",
                            subtask=subtask,
                        )
                    )
            else:
                decomposition.mark_completed(subtask.id, subtask_response)

        if not decomposition.is_complete():
            return f"Task partially completed. {decomposition.to_prompt()}"

        summary_prompt = (
            f"All subtasks completed for: {task}\n\n"
            f"{decomposition.to_prompt()}\n\n"
            "Provide a brief summary of what was accomplished."
        )
        self.source.session.append(Message(role=Role.USER, content=summary_prompt))
        return await self.run_task(
            summary_prompt,
            emit,
            on_confirmation,
            on_user_question,
            None,
            original_task,
        )

    async def _decompose_task(self, task: str):
        """Request one structured decomposition from the active backend."""

        prompt = DECOMPOSITION_PROMPT.format(task=task)
        response = await self.source.backend.complete(
            messages=[
                self.source.session.system_message_factory(),
                Message(role=Role.USER, content=prompt),
            ],
            tools=None,
            temperature=0.3,
            max_tokens=1000,
        )
        return parse_decomposition(response.content, task)
