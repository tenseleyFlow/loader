"""Completion-policy helpers for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..llm.base import Message, Role
from .context import RuntimeContext
from .dod import DefinitionOfDone
from .logging import get_runtime_logger
from .events import AgentEvent, TurnSummary
from .evidence_provenance import EvidenceProvenance
from .reasoning_types import TaskCompletionCheck
from .task_completion import assess_completion_follow_through_with_provenance
from .verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
    describe_verification_attempt,
)

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
    evidence_provenance: list[EvidenceProvenance] = field(default_factory=list)
    verification_observations: list[VerificationObservation] = field(default_factory=list)
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
        rlog = get_runtime_logger()
        rlog.completion_check("text_loop", "detected" if is_text_loop else "clear",
                              reason=loop_description if is_text_loop else None)
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
        dod: DefinitionOfDone | None = None,
    ) -> ContinuationDecision:
        """Nudge non-mutating tasks to continue when completion looks premature."""

        cfg = self.context.config.reasoning
        assessment = (
            assess_completion_follow_through_with_provenance(
                task=task,
                response=content,
                actions_taken=actions_taken,
                dod=dod,
            )
            if cfg.use_quick_completion
            else None
        )
        completion_check = assessment.check if assessment is not None else None
        is_premature = bool(completion_check is not None and not completion_check.is_complete)
        rlog = get_runtime_logger()
        rlog.completion_check(
            "follow_through",
            "premature" if is_premature else "accepted",
            reason=(
                "; ".join(completion_check.missing_evidence[:2])
                if is_premature and completion_check
                else None
            ),
        )
        if not is_premature:
            return ContinuationDecision(
                should_continue=False,
                decision_code="completion_response_accepted",
                decision_summary=(
                    "accepted the response because completion heuristics found "
                    "no missing follow-through"
                ),
                completion_check=completion_check,
                evidence_provenance=(
                    list(assessment.evidence_provenance) if assessment is not None else []
                ),
                verification_observations=(
                    list(assessment.verification_observations)
                    if assessment is not None
                    else []
                ),
            )

        if continuation_count >= cfg.max_continuation_prompts:
            verification_observations = (
                list(assessment.verification_observations)
                if assessment is not None
                else []
            )
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
                decision_summary=self._budget_exhausted_summary(
                    completion_check=completion_check,
                    verification_observations=verification_observations,
                ),
                completion_check=completion_check,
                evidence_provenance=(
                    list(assessment.evidence_provenance) if assessment is not None else []
                ),
                verification_observations=verification_observations,
                final_response=self._format_budget_exhausted_response(
                    completion_check,
                    verification_observations=verification_observations,
                ),
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
            decision_summary=(
                "requested one continuation because the non-mutating response "
                "looked incomplete"
            ),
            completion_check=completion_check,
            evidence_provenance=(
                list(assessment.evidence_provenance) if assessment is not None else []
            ),
            verification_observations=(
                list(assessment.verification_observations)
                if assessment is not None
                else []
            ),
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
        *,
        verification_observations: list[VerificationObservation] | None = None,
    ) -> str:
        observed = CompletionPolicy._observed_verification_summary(
            verification_observations or []
        )
        if observed:
            return (
                "I stopped because the continuation budget was exhausted and observed "
                f"verification still showed: {observed}."
            )
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

    @staticmethod
    def _budget_exhausted_summary(
        *,
        completion_check: TaskCompletionCheck | None,
        verification_observations: list[VerificationObservation],
    ) -> str:
        observed = CompletionPolicy._observed_verification_summary(
            verification_observations
        )
        if observed:
            return (
                "stopped because the continuation budget was exhausted while "
                f"observed verification still showed {observed}"
            )
        if completion_check is None or not completion_check.missing_evidence:
            return (
                "stopped because the continuation budget was exhausted before "
                "follow-through evidence was established"
            )
        return (
            "stopped because the continuation budget was exhausted while "
            "follow-through evidence was still missing"
        )

    @staticmethod
    def _observed_verification_summary(
        verification_observations: list[VerificationObservation],
    ) -> str | None:
        if not verification_observations:
            return None
        for entry in verification_observations:
            if entry.status == VerificationObservationStatus.FAILED.value:
                return CompletionPolicy._render_observation(entry)
        for entry in verification_observations:
            if entry.status == VerificationObservationStatus.STALE.value:
                return CompletionPolicy._render_observation(entry)
        for entry in verification_observations:
            if entry.status == VerificationObservationStatus.MISSING.value:
                return CompletionPolicy._render_observation(entry)
        for entry in verification_observations:
            if entry.status == VerificationObservationStatus.PASSED.value:
                return CompletionPolicy._render_observation(entry)
        return CompletionPolicy._render_observation(verification_observations[0])

    @staticmethod
    def _render_observation(entry: VerificationObservation) -> str:
        summary = entry.summary.strip()
        details: list[str] = []
        if entry.detail:
            detail = entry.detail.strip()
            if detail and detail not in summary:
                details.append(detail)
        attempt = describe_verification_attempt(entry)
        if attempt and attempt not in summary:
            details.append(attempt)
        if details:
            return f"{summary} [{'; '.join(details)}]"
        return summary
