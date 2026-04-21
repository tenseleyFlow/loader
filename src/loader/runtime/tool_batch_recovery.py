"""Recovery helpers for failed tool-batch executions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..llm.base import Message, ToolCall
from .compaction import infer_preferred_next_step, summarize_confirmed_facts
from .context import RuntimeContext
from .events import AgentEvent
from .executor import ToolExecutionOutcome
from .recovery import RecoveryContext, format_failure_message, format_recovery_prompt

EventSink = Callable[[AgentEvent], Awaitable[None]]


class ToolBatchRecoveryController:
    """Own recovery follow-up generation for failed tool executions."""

    def __init__(self, context: RuntimeContext) -> None:
        self.context = context

    async def build_follow_up(
        self,
        *,
        tool_call: ToolCall,
        outcome: ToolExecutionOutcome,
        emit: EventSink,
    ) -> Message | None:
        """Generate a recovery prompt or final failure message after a tool error."""

        recovery_context = self.context.recovery_context
        if recovery_context is None or not recovery_context.is_related_failure(
            tool_call.name,
            tool_call.arguments,
            outcome.result_output,
        ):
            recovery_context = RecoveryContext(
                original_tool=tool_call.name,
                original_args=tool_call.arguments,
                max_retries=self.context.config.max_recovery_attempts,
            )
            self.context.recovery_context = recovery_context

        if recovery_context.is_similar_attempt(
            tool_call.name,
            tool_call.arguments,
        ):
            await emit(
                AgentEvent(
                    type="error",
                    content=(
                        "Loop detected: already tried a similar command. "
                        "Try a DIFFERENT approach (e.g., read a config file first)."
                    ),
                    tool_name=tool_call.name,
                )
            )
        else:
            recovery_context.add_attempt(
                tool_call.name,
                tool_call.arguments,
                outcome.result_output,
            )

        if recovery_context.can_retry():
            attempt_number = len(recovery_context.attempts)
            await emit(
                AgentEvent(
                    type="recovery",
                    content=(
                        "Tool failed, attempting recovery "
                        f"({attempt_number}/{recovery_context.max_retries})"
                    ),
                    tool_name=tool_call.name,
                    recovery_attempt=attempt_number,
                )
            )
            recovery_prompt = format_recovery_prompt(
                recovery_context,
                tool_call.name,
                tool_call.arguments,
                outcome.result_output,
            )
            recovery_prompt = self._augment_recovery_prompt(recovery_prompt)
            return Message.tool_result_message(
                tool_call_id=tool_call.id,
                display_content=recovery_prompt,
                result_content=recovery_prompt,
                is_error=True,
            )

        failure_message = format_failure_message(recovery_context)
        await emit(
            AgentEvent(
                type="error",
                content=failure_message,
                tool_name=tool_call.name,
            )
        )
        self.context.recovery_context = None
        return Message.tool_result_message(
            tool_call_id=tool_call.id,
            display_content=(f"Observation [{tool_call.name}]: Error: {failure_message}"),
            result_content=failure_message,
            is_error=True,
        )

    def _augment_recovery_prompt(self, prompt: str) -> str:
        """Append transcript-aware recovery guidance when recent facts exist."""

        session = self.context.session
        current_task = getattr(session, "current_task", None)
        confirmed_facts = summarize_confirmed_facts(session.messages)
        preferred_next_step = infer_preferred_next_step(
            session.messages,
            current_task=current_task,
        )
        if not confirmed_facts and not preferred_next_step and not current_task:
            return prompt

        lines = [prompt, "", "## CONTINUE FROM KNOWN STATE"]
        if current_task:
            lines.append(f"- Current task: {current_task}")
        if confirmed_facts:
            lines.append(f"- Confirmed facts: {confirmed_facts}")
        if preferred_next_step:
            lines.append(f"- Preferred next step: {preferred_next_step}")
        lines.append(
            "- Preserve progress: do not restart by rereading already-confirmed files "
            "unless you need genuinely new evidence."
        )
        return "\n".join(lines)
