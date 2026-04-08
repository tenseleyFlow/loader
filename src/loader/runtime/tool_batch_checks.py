"""Confidence and verification helpers for tool-batch execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..llm.base import Message, Role, ToolCall
from .context import RuntimeContext
from .events import AgentEvent
from .executor import ToolExecutionOutcome, ToolExecutionState

EventSink = Callable[[AgentEvent], Awaitable[None]]


class ToolBatchConfidenceGate:
    """Own low-confidence gating before a tool call executes."""

    def __init__(self, context: RuntimeContext) -> None:
        self.context = context

    async def should_skip(
        self,
        *,
        tool_call: ToolCall,
        emit: EventSink,
    ) -> bool:
        """Emit confidence results and decide whether to skip execution."""

        cfg = self.context.config.reasoning
        context = "\n".join(
            message.content[:500]
            for message in self.context.messages[-5:]
            if message.content
        )
        confidence = await self.context.assess_confidence(
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


class ToolBatchVerificationGate:
    """Own post-tool verification and correction prompting."""

    def __init__(self, context: RuntimeContext) -> None:
        self.context = context

    async def should_continue(
        self,
        *,
        tool_call: ToolCall,
        outcome: ToolExecutionOutcome,
        emit: EventSink,
    ) -> bool:
        """Run post-tool verification and decide whether the loop should continue."""

        cfg = self.context.config.reasoning
        if not (
            cfg.verification
            and outcome.state == ToolExecutionState.EXECUTED
            and not outcome.is_error
        ):
            return False

        verification = await self.context.verify_action(
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
