"""Recovery helpers for failed tool-batch executions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from pathlib import Path
import re

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
            recovery_prompt = self._augment_recovery_prompt(
                recovery_prompt,
                tool_call=tool_call,
                outcome=outcome,
            )
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

    def _augment_recovery_prompt(
        self,
        prompt: str,
        *,
        tool_call: ToolCall,
        outcome: ToolExecutionOutcome,
    ) -> str:
        """Append transcript-aware recovery guidance when recent facts exist."""

        session = self.context.session
        current_task = getattr(session, "current_task", None)
        confirmed_facts = summarize_confirmed_facts(session.messages)
        preferred_next_step = infer_preferred_next_step(
            session.messages,
            current_task=current_task,
        )
        actionable_known_state = bool(confirmed_facts and preferred_next_step)
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
        if actionable_known_state:
            lines.extend(
                [
                    "",
                    "## ACTION BIAS FOR THIS RECOVERY",
                    "- The confirmed findings above are already enough to keep moving.",
                    "- Prefer edit/write/patch on the target file over rereading the same files.",
                    "- Only inspect one more file if a specific filename, href, or title is still unknown.",
                    "- Treat the preferred next step as the default path forward.",
                ]
            )
        candidate_lines = self._file_not_found_candidate_lines(tool_call, outcome)
        if candidate_lines:
            lines.extend(["", "## LIKELY FILE CANDIDATES", *candidate_lines])
        return "\n".join(lines)

    def _file_not_found_candidate_lines(
        self,
        tool_call: ToolCall,
        outcome: ToolExecutionOutcome,
    ) -> list[str]:
        if tool_call.name not in {"read", "write", "edit", "patch"}:
            return []
        if "not found" not in outcome.result_output.lower():
            return []

        missing_path = self._canonicalize_path(
            str(
                tool_call.arguments.get("file_path")
                or tool_call.arguments.get("path")
                or ""
            ).strip()
        )
        if not missing_path:
            return []

        candidates = self._rank_known_file_candidates(missing_path)
        if not candidates:
            return []

        names = ", ".join(f"`{Path(candidate).name}`" for candidate in candidates[:3])
        return [
            f"- Requested file does not exist: `{missing_path}`",
            f"- Closest known files in the same directory: {names}",
            "- Prefer one of those exact filenames instead of retrying the missing path.",
        ]

    def _rank_known_file_candidates(self, missing_path: str) -> list[str]:
        missing_parent = str(Path(missing_path).parent)
        missing_name = Path(missing_path).name
        missing_prefix = missing_name.split("-", 1)[0]

        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for candidate in self._known_file_paths():
            if candidate == missing_path:
                continue
            if str(Path(candidate).parent) != missing_parent:
                continue
            name = Path(candidate).name
            if name in seen:
                continue
            seen.add(name)

            score = SequenceMatcher(None, missing_name, name).ratio()
            if missing_prefix and name.startswith(f"{missing_prefix}-"):
                score += 1.0
            ranked.append((score, candidate))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, candidate in ranked]

    def _known_file_paths(self) -> list[str]:
        pattern = re.compile(r"(?:~|/)[^\s`\"']+\.html")
        discovered: list[str] = []
        seen: set[str] = set()
        for message in self.context.session.messages:
            for raw_path in pattern.findall(message.content):
                candidate = self._canonicalize_path(raw_path)
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                discovered.append(candidate)
        return discovered

    def _canonicalize_path(self, raw_path: str) -> str:
        if not raw_path:
            return ""
        try:
            return str(Path(raw_path).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            return str(Path(raw_path).expanduser())
