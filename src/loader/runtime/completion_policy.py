"""Completion-policy helpers for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..llm.base import Message, Role
from .context import RuntimeContext
from .events import AgentEvent, TurnSummary
from .reasoning_types import TaskCompletionCheck
from .task_completion import assess_completion_follow_through, detect_premature_completion

EventSink = Callable[[AgentEvent], Awaitable[None]]

@dataclass(slots=True)
class TextLoopDecision:
    """Decision returned when checking for text loops."""

    should_stop: bool
    decision_code: str
    decision_summary: str
    final_response: str = ""
    failure: str | None = None


@dataclass(slots=True)
class ContinuationDecision:
    """Decision returned from the non-mutating completion nudge."""

    should_continue: bool
    decision_code: str
    decision_summary: str
    completion_check: TaskCompletionCheck | None = None
    should_finalize: bool = False
    final_response: str = ""
    failure: str | None = None


class CompletionPolicy:
    """Owns loop bailout, non-mutating completion nudges, and response cleanup."""

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
            return TextLoopDecision(
                should_stop=False,
                decision_code="text_loop_not_detected",
                decision_summary="accepted the response because no text loop was detected",
            )

        final_response = (
            "I stopped because I was repeating myself and couldn't make further progress."
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
            decision_code="text_loop_bailout",
            decision_summary="stopped after detecting a repeated text loop",
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
        completion_check = (
            assess_completion_follow_through(
                task=task,
                response=content,
                actions_taken=actions_taken,
            )
            if cfg.use_quick_completion
            else None
        )
        is_premature = (
            detect_premature_completion(task, content, actions_taken)
            if cfg.use_quick_completion
            else False
        )
        if not is_premature:
            return ContinuationDecision(
                should_continue=False,
                decision_code="completion_response_accepted",
                decision_summary="accepted the response because completion heuristics found no missing follow-through",
                completion_check=completion_check,
            )

        if continuation_count >= cfg.max_continuation_prompts:
            await emit(
                AgentEvent(
                    type="completion_check",
                    content=f"Task may be incomplete ({len(actions_taken)} actions taken)",
                    completion_check=completion_check,
                )
            )
            return ContinuationDecision(
                should_continue=False,
                should_finalize=True,
                decision_code="continuation_budget_exhausted",
                decision_summary=(
                    "stopped because the continuation budget was exhausted while "
                    "follow-through evidence was still missing"
                ),
                completion_check=completion_check,
                final_response=self._format_budget_exhausted_response(completion_check),
                failure="missing follow-through evidence after continuation budget exhaustion",
            )

        await emit(
            AgentEvent(
                type="completion_check",
                content=f"Task may be incomplete ({len(actions_taken)} actions taken)",
                completion_check=completion_check,
            )
        )
        self.context.session.append(Message(role=Role.ASSISTANT, content=response_content))
        self.context.session.append(
            Message(role=Role.USER, content=completion_check.continuation_prompt)
        )
        return ContinuationDecision(
            should_continue=True,
            decision_code="premature_completion_nudge",
            decision_summary="requested one continuation because the non-mutating response looked incomplete",
            completion_check=completion_check,
        )

    @staticmethod
    def finalize_response_text(
        *,
        content: str,
        actions_taken: list[str],
    ) -> str:
        """Return the assistant response without synthetic follow-up phrasing."""

        _ = actions_taken
        return content

    @staticmethod
    def _format_budget_exhausted_response(
        completion_check: TaskCompletionCheck | None,
    ) -> str:
        if completion_check is None or not completion_check.missing_evidence:
            return (
                "I stopped because the continuation budget was exhausted before I could "
                "show enough evidence that the task was complete."
            )
        missing = "; ".join(completion_check.missing_evidence[:2])
        return (
            "I stopped because I still could not show enough evidence that the task "
            f"was complete. Missing evidence: {missing}."
        )
