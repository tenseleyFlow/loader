"""Completion-policy helpers for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..agent.reasoning import (
    TaskCompletionCheck,
    detect_premature_completion,
    get_continuation_prompt,
    should_self_critique,
)
from ..llm.base import Message, Role
from .context import RuntimeContext
from .events import AgentEvent, TurnSummary

EventSink = Callable[[AgentEvent], Awaitable[None]]


@dataclass(slots=True)
class CritiqueDecision:
    """Decision returned from self-critique handling."""

    should_continue: bool


@dataclass(slots=True)
class TextLoopDecision:
    """Decision returned when checking for text loops."""

    should_stop: bool
    final_response: str = ""
    failure: str | None = None


@dataclass(slots=True)
class ContinuationDecision:
    """Decision returned from the non-mutating completion nudge."""

    should_continue: bool


class CompletionPolicy:
    """Owns critique, loop bailout, and non-mutating completion nudges."""

    def __init__(self, context: RuntimeContext) -> None:
        self.context = context

    async def maybe_self_critique(
        self,
        *,
        content: str,
        response_content: str,
        task: str,
        emit: EventSink,
    ) -> CritiqueDecision:
        """Run self-critique when enabled and revision would be useful."""

        cfg = self.context.config.reasoning
        if not (cfg.self_critique and len(content) > 100):
            return CritiqueDecision(should_continue=False)

        is_code_response = "```" in content or any(
            keyword in content.lower()
            for keyword in ["def ", "function ", "class ", "import "]
        )
        if not should_self_critique(content, is_code=is_code_response):
            return CritiqueDecision(should_continue=False)

        critique = await self.context.legacy.self_critique(content, task)
        await emit(
            AgentEvent(
                type="critique",
                content=f"Self-critique: {len(critique.issues_found)} issues found",
                critique=critique,
            )
        )
        if not critique.can_revise():
            return CritiqueDecision(should_continue=False)

        revision_message = (
            "[SELF-CRITIQUE] Review your response:\n"
            f"Issues found: {', '.join(critique.issues_found)}\n"
            f"Suggestions: {', '.join(critique.suggestions)}\n\n"
            "Please provide an improved response addressing these issues."
        )
        self.context.session.append(Message(role=Role.ASSISTANT, content=response_content))
        self.context.session.append(Message(role=Role.USER, content=revision_message))
        critique.revision_count += 1
        return CritiqueDecision(should_continue=True)

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

    async def maybe_continue_for_completion(
        self,
        *,
        content: str,
        response_content: str,
        task: str,
        actions_taken: list[str],
        continuation_count: int,
        emit: EventSink,
    ) -> ContinuationDecision:
        """Nudge non-mutating tasks to continue when completion looks premature."""

        cfg = self.context.config.reasoning
        if continuation_count >= cfg.max_continuation_prompts:
            return ContinuationDecision(should_continue=False)

        is_premature = (
            detect_premature_completion(task, content, actions_taken)
            if cfg.use_quick_completion
            else False
        )
        if not is_premature:
            return ContinuationDecision(should_continue=False)

        continuation_prompt = get_continuation_prompt(
            task,
            actions_taken,
            content,
        )
        await emit(
            AgentEvent(
                type="completion_check",
                content=f"Task may be incomplete ({len(actions_taken)} actions taken)",
                completion_check=TaskCompletionCheck(
                    original_task=task,
                    is_complete=False,
                    accomplished=[action.split(":")[0] for action in actions_taken],
                    continuation_prompt=continuation_prompt,
                ),
            )
        )
        self.context.session.append(Message(role=Role.ASSISTANT, content=response_content))
        self.context.session.append(Message(role=Role.USER, content=continuation_prompt))
        return ContinuationDecision(should_continue=True)

    @staticmethod
    def finalize_response_text(
        *,
        content: str,
        actions_taken: list[str],
    ) -> str:
        """Return the assistant response without a synthetic follow-up suffix."""

        _ = actions_taken
        return content
