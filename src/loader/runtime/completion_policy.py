"""Completion-policy helpers for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..llm.base import Message, Role
from .context import RuntimeContext
from .events import AgentEvent, TurnSummary

EventSink = Callable[[AgentEvent], Awaitable[None]]


@dataclass(slots=True)
class TextLoopDecision:
    """Decision returned when checking for text loops."""

    should_stop: bool
    final_response: str = ""
    failure: str | None = None


class CompletionPolicy:
    """Owns loop bailout and final response cleanup."""

    def __init__(self, context: RuntimeContext) -> None:
        self.context = context

    async def maybe_stop_for_text_loop(
        self,
        *,
        content: str,
        emit: EventSink,
        summary: TurnSummary,
    ) -> TextLoopDecision:
        """Stop the turn when the assistant starts repeating textually."""

        is_text_loop, loop_description = self.context.safeguards.detect_text_loop(content)
        if not is_text_loop:
            return TextLoopDecision(should_stop=False)

        final_response = (
            "I seem to be repeating myself. "
            "Let me know if you'd like me to try a different approach."
        )
        summary.final_response = final_response
        summary.failures.append(loop_description)
        final_message = Message(role=Role.ASSISTANT, content=final_response)
        self.context.session.append(final_message)
        summary.assistant_messages.append(final_message)
        await emit(
            AgentEvent(
                type="error",
                content=f"Text loop detected: {loop_description}. Stopping.",
            )
        )
        await emit(AgentEvent(type="response", content=final_response))
        return TextLoopDecision(
            should_stop=True,
            final_response=final_response,
            failure=loop_description,
        )

    @staticmethod
    def finalize_response_text(
        *,
        content: str,
        actions_taken: list[str],
    ) -> str:
        """Return the assistant response without a synthetic follow-up suffix."""

        _ = actions_taken
        return content
