"""Recovery helpers for failed tool-batch executions."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..llm.base import Message, Role, ToolCall
from .compaction import (
    extract_key_files,
    infer_preferred_next_step,
    summarize_confirmed_facts,
)
from .context import RuntimeContext
from .events import AgentEvent
from .executor import ToolExecutionOutcome
from .recovery import (
    RecoveryContext,
    detect_missing_mutation_payload,
    format_failure_message,
    format_recovery_prompt,
)
from .repair_focus import ActiveRepairContext, extract_active_repair_context

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
                        "Try a different next step using the files and facts you already have "
                        "(for example, make the specific edit, verify the current result, or "
                        "inspect one concrete unresolved target)."
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
        active_repair = self._active_repair_context()
        effective_task = current_task
        if active_repair is not None and active_repair.artifact_path:
            effective_task = (
                "Repair the current artifact using the failed verification evidence: "
                f"{active_repair.artifact_path}"
            )
            focus_path = active_repair.artifact_path
            preferred_next_step = (
                f"Update `{active_repair.artifact_path}` to resolve the current "
                "verification failures."
            )
        else:
            focus_path = self._preferred_focus_path(
                tool_call=tool_call,
                current_task=current_task,
            )
            preferred_next_step = infer_preferred_next_step(
                session.messages,
                current_task=effective_task,
                focus_path=focus_path or None,
            )
        confirmed_facts = summarize_confirmed_facts(session.messages)
        lines = [prompt]
        candidate_lines = self._file_not_found_candidate_lines(
            tool_call,
            outcome,
            active_repair=active_repair,
        )
        actionable_known_state = bool(
            active_repair or current_task or confirmed_facts or preferred_next_step or candidate_lines
        )
        if active_repair is not None:
            lines.extend(["", "## ACTIVE REPAIR TARGET"])
            lines.append(
                "- Verification already failed on the current artifact set. "
                "Stay on this repair until the broken local references are fixed."
            )
            lines.extend(active_repair.repair_lines)
            drifted_path = self._canonicalize_path(
                str(
                    tool_call.arguments.get("file_path")
                    or tool_call.arguments.get("path")
                    or ""
                ).strip()
            )
            if (
                drifted_path
                and active_repair.artifact_path
                and drifted_path != active_repair.artifact_path
            ):
                lines.append(
                    f"- The failed tool call drifted to `{drifted_path}`. "
                    f"Return to `{active_repair.artifact_path}` instead of reopening "
                    "the original discovery task."
                )
            lines.append(
                "- Treat this repair as higher priority than the original discovery "
                "prompt until verification passes."
            )
        if active_repair or confirmed_facts or preferred_next_step or current_task:
            lines.extend(["", "## CONTINUE FROM KNOWN STATE"])
            if active_repair is not None and active_repair.artifact_path:
                lines.append(f"- Active repair target: `{active_repair.artifact_path}`")
            elif current_task:
                lines.append(f"- Current task: {current_task}")
            if confirmed_facts:
                lines.append(f"- Confirmed facts: {confirmed_facts}")
            if preferred_next_step:
                lines.append(f"- Preferred next step: {preferred_next_step}")
            lines.append(
                "- Preserve progress: do not restart by rereading already-confirmed files "
                "unless you need genuinely new evidence."
            )
            if active_repair is not None:
                lines.append(
                    "- Do not go back to the original reference guide or invent alternate "
                    "paths while this repair target is unresolved."
                )
            if actionable_known_state:
                target_line = (
                    f"- Prefer edit/write/patch on `{active_repair.artifact_path}` over "
                    "rereading the same files."
                    if active_repair is not None and active_repair.artifact_path
                    else "- Prefer edit/write/patch on the target file over rereading the same files."
                )
                lines.extend(
                    [
                        "",
                        "## ACTION BIAS FOR THIS RECOVERY",
                        "- The confirmed findings above are already enough to keep moving.",
                        target_line,
                        "- Only inspect one more file if a specific filename, href, or title is still unknown.",
                        "- Treat the preferred next step as the default path forward.",
                    ]
                )
        if candidate_lines:
            lines.extend(["", "## LIKELY FILE CANDIDATES", *candidate_lines])
        target_excerpt_lines = self._target_excerpt_lines(tool_call)
        if target_excerpt_lines:
            lines.extend(["", "## CURRENT TARGET EXCERPT", *target_excerpt_lines])
        payload_fix_lines = self._missing_payload_fix_lines(tool_call, outcome)
        if payload_fix_lines:
            lines.extend(["", "## PAYLOAD FORMAT FIX", *payload_fix_lines])
        return "\n".join(lines)

    def _missing_payload_fix_lines(
        self,
        tool_call: ToolCall,
        outcome: ToolExecutionOutcome,
    ) -> list[str]:
        fix = detect_missing_mutation_payload(
            tool_call.name,
            tool_call.arguments,
            outcome.result_output,
        )
        if fix is None:
            return []

        target = fix["file_path"]
        invalid_fields = ", ".join(f"`{field}`" for field in fix["invalid_fields"])
        required_fields = "`, `".join(fix["required_fields"])
        if tool_call.name == "write":
            target_line = (
                f"- The failed call for `{target}` omitted the required `content` payload."
                if target
                else "- The failed call omitted the required `content` payload."
            )
            return [
                target_line,
                f"- {invalid_fields} are summary counters, not valid write inputs.",
                "- Resend one concrete `write(file_path=..., content='...')` call now instead of rereading more files.",
            ]

        if tool_call.name == "edit":
            target_line = (
                f"- The failed call for `{target}` omitted the required `{required_fields}` payload."
                if target
                else f"- The failed call omitted the required `{required_fields}` payload."
            )
            return [
                target_line,
                f"- {invalid_fields} are summary counters, not valid edit inputs.",
                "- Resend one concrete `edit(file_path=..., old_string='...', new_string='...')` call now instead of rereading more files.",
            ]

        if tool_call.name == "patch":
            target_line = (
                f"- The failed call for `{target}` omitted the required patch body."
                if target
                else "- The failed call omitted the required patch body."
            )
            return [
                target_line,
                f"- {invalid_fields} are summary counters, not valid patch inputs.",
                "- Resend one concrete `patch(file_path=..., patch='...')` or `patch(..., hunks=[...])` call now instead of rereading more files.",
            ]

        return []

    def _preferred_focus_path(
        self,
        *,
        tool_call: ToolCall,
        current_task: str | None,
    ) -> str:
        raw_path = str(
            tool_call.arguments.get("file_path")
            or tool_call.arguments.get("path")
            or ""
        ).strip()
        if not raw_path:
            return ""
        if tool_call.name in {"write", "edit", "patch"} or not current_task:
            return raw_path

        primary_target = self._primary_task_target_path(current_task)
        if not primary_target:
            return raw_path

        candidate = self._canonicalize_path(raw_path)
        target = self._canonicalize_path(primary_target)
        if not candidate or not target or candidate == target:
            return raw_path

        candidate_path = Path(candidate)
        target_path = Path(target)
        if (
            tool_call.name == "read"
            and candidate_path.suffix == ".html"
            and candidate_path.parent == target_path.parent / "chapters"
        ):
            return target

        return raw_path

    def _primary_task_target_path(self, current_task: str) -> str | None:
        paths = extract_key_files(
            [Message(role=Role.USER, content=current_task)],
            limit=6,
        )
        for path in paths:
            normalized = self._canonicalize_path(path)
            if not normalized:
                continue
            if normalized.endswith(".html") and "/chapters/" not in normalized:
                return normalized
        for path in paths:
            normalized = self._canonicalize_path(path)
            if normalized:
                return normalized
        return None

    def _file_not_found_candidate_lines(
        self,
        tool_call: ToolCall,
        outcome: ToolExecutionOutcome,
        *,
        active_repair: ActiveRepairContext | None = None,
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
            if active_repair is not None and active_repair.artifact_path:
                return [
                    f"- Requested file does not exist: `{missing_path}`",
                    f"- Active repair target is `{active_repair.artifact_path}`.",
                    "- Repair the known target instead of inventing a new path.",
                ]
            return []

        names = ", ".join(self._describe_candidate(candidate) for candidate in candidates[:3])
        lines = [
            f"- Requested file does not exist: `{missing_path}`",
            f"- Closest known files in the same directory: {names}",
            "- Prefer one of those exact filenames instead of retrying the missing path.",
        ]
        if active_repair is not None and active_repair.artifact_path:
            lines.append(
                f"- Keep the repair centered on `{active_repair.artifact_path}` rather than "
                "switching back to broad discovery."
            )
        return lines

    def _rank_known_file_candidates(self, missing_path: str) -> list[str]:
        missing_parent = str(Path(missing_path).parent)
        missing_name = Path(missing_path).name
        missing_prefix = missing_name.split("-", 1)[0]

        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()
        for candidate in self._known_file_paths(missing_path):
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

    def _known_file_paths(self, missing_path: str | None = None) -> list[str]:
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
        if missing_path:
            missing = Path(missing_path)
            parent = missing.parent
            if parent.is_dir():
                sibling_candidates = sorted(
                    child.resolve(strict=False)
                    for child in parent.iterdir()
                    if child.is_file()
                    and child.name != missing.name
                    and (
                        not missing.suffix
                        or child.suffix == missing.suffix
                    )
                )
                for child in sibling_candidates:
                    candidate = str(child)
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    discovered.append(candidate)
        return discovered

    def _describe_candidate(self, candidate: str) -> str:
        path = Path(candidate)
        return f"`{path.name}`"

    def _target_excerpt_lines(self, tool_call: ToolCall) -> list[str]:
        if tool_call.name not in {"edit", "patch"}:
            return []

        raw_path = str(
            tool_call.arguments.get("file_path")
            or tool_call.arguments.get("path")
            or ""
        ).strip()
        target_path = self._canonicalize_path(raw_path)
        if not target_path:
            return []

        path = Path(target_path)
        if not path.is_file():
            return []

        try:
            content = path.read_text()
        except Exception:
            return []

        file_lines = content.splitlines()
        if not file_lines:
            return [
                f"- Target file: `{target_path}`",
                "- The file is currently empty.",
                "- Use the exact on-disk state above when preparing the next mutation.",
            ]

        start, end, label = self._excerpt_window_for_tool_call(
            file_lines=file_lines,
            content=content,
            tool_call=tool_call,
        )
        excerpt = self._format_excerpt_lines(file_lines, start, end)
        if not excerpt:
            return []

        return [
            f"- Target file: `{target_path}`",
            f"- {label}",
            *excerpt,
            "- Use the exact on-disk text above when preparing the next mutation.",
            "- If several adjacent lines are wrong, replace the containing block in one edit instead of retrying a smaller substitution.",
        ]

    def _excerpt_window_for_tool_call(
        self,
        *,
        file_lines: list[str],
        content: str,
        tool_call: ToolCall,
    ) -> tuple[int, int, str]:
        if tool_call.name == "edit":
            window = self._edit_excerpt_window(
                file_lines=file_lines,
                content=content,
                arguments=tool_call.arguments,
            )
            if window is not None:
                return window
        if tool_call.name == "patch":
            window = self._patch_excerpt_window(
                file_lines=file_lines,
                arguments=tool_call.arguments,
            )
            if window is not None:
                return window
        return self._bounded_window(
            file_lines=file_lines,
            start=0,
            length=min(10, len(file_lines)),
            label="Current file contents:",
        )

    def _edit_excerpt_window(
        self,
        *,
        file_lines: list[str],
        content: str,
        arguments: dict[str, Any],
    ) -> tuple[int, int, str] | None:
        old_string = str(arguments.get("old_string") or "")
        new_string = str(arguments.get("new_string") or "")

        if old_string:
            exact_window = self._exact_string_window(
                content=content,
                file_lines=file_lines,
                needle=old_string,
                label="Current file contents for the requested edit:",
            )
            if exact_window is not None:
                return exact_window

        anchor = old_string or new_string
        approximate_window = self._approximate_string_window(
            file_lines=file_lines,
            needle=anchor,
            label="Closest on-disk block to the requested edit:",
        )
        if approximate_window is not None:
            return approximate_window
        return None

    def _patch_excerpt_window(
        self,
        *,
        file_lines: list[str],
        arguments: dict[str, Any],
    ) -> tuple[int, int, str] | None:
        hunks = arguments.get("hunks")
        if not isinstance(hunks, list) or not hunks:
            return None

        first_hunk = hunks[0]
        if not isinstance(first_hunk, dict):
            return None

        anchor_lines: list[str] = []
        raw_lines = first_hunk.get("lines")
        if isinstance(raw_lines, list):
            for raw_line in raw_lines:
                if not isinstance(raw_line, str) or not raw_line:
                    continue
                if raw_line[0] in {" ", "-"}:
                    anchor_lines.append(raw_line[1:])

        anchor = "\n".join(anchor_lines).strip()
        approximate_window = self._approximate_string_window(
            file_lines=file_lines,
            needle=anchor,
            label="Closest on-disk block to the requested patch:",
        )
        if approximate_window is not None:
            return approximate_window

        old_start = first_hunk.get("old_start", 1)
        old_lines = first_hunk.get("old_lines", len(anchor_lines) or 1)
        try:
            start = max(0, int(old_start) - 1)
        except (TypeError, ValueError):
            start = 0
        try:
            length = max(1, int(old_lines))
        except (TypeError, ValueError):
            length = max(1, len(anchor_lines) or 1)
        return self._bounded_window(
            file_lines=file_lines,
            start=start,
            length=length,
            label="Current file contents near the requested patch location:",
        )

    def _exact_string_window(
        self,
        *,
        content: str,
        file_lines: list[str],
        needle: str,
        label: str,
    ) -> tuple[int, int, str] | None:
        if not needle:
            return None
        index = content.find(needle)
        if index == -1:
            return None
        start_line = content[:index].count("\n")
        block_length = max(1, len(needle.splitlines()))
        return self._bounded_window(
            file_lines=file_lines,
            start=start_line,
            length=block_length,
            label=label,
        )

    def _approximate_string_window(
        self,
        *,
        file_lines: list[str],
        needle: str,
        label: str,
    ) -> tuple[int, int, str] | None:
        normalized_needle = self._normalize_match_text(needle)
        if not normalized_needle:
            return None

        needle_lines = [line for line in needle.splitlines() if line.strip()]
        if not needle_lines:
            needle_lines = [needle.strip()]

        min_window = 1
        max_window = min(len(file_lines), max(1, len(needle_lines) + 2))
        best_score = 0.0
        best_start = 0
        best_length = min(max_window, max(1, len(needle_lines)))
        for window_length in range(min_window, max_window + 1):
            for start in range(0, len(file_lines) - window_length + 1):
                candidate = "\n".join(file_lines[start : start + window_length])
                score = SequenceMatcher(
                    None,
                    normalized_needle,
                    self._normalize_match_text(candidate),
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_start = start
                    best_length = window_length

        if best_score < 0.25:
            return None

        return self._bounded_window(
            file_lines=file_lines,
            start=best_start,
            length=best_length,
            label=label,
        )

    def _bounded_window(
        self,
        *,
        file_lines: list[str],
        start: int,
        length: int,
        label: str,
    ) -> tuple[int, int, str]:
        context_before = 2
        context_after = 2
        start_index = max(0, start - context_before)
        end_index = min(len(file_lines), start + max(1, length) + context_after)
        return start_index, end_index, label

    def _format_excerpt_lines(
        self,
        file_lines: list[str],
        start: int,
        end: int,
    ) -> list[str]:
        if start >= end:
            return []
        width = len(str(end))
        return [
            f"  {line_number:>{width}} | {file_lines[line_number - 1]}"
            for line_number in range(start + 1, end + 1)
        ]

    def _normalize_match_text(self, text: str) -> str:
        return " ".join(str(text or "").split())

    def _active_repair_context(self) -> ActiveRepairContext | None:
        return extract_active_repair_context(self.context.session.messages)

    def _canonicalize_path(self, raw_path: str) -> str:
        if not raw_path:
            return ""
        try:
            return str(Path(raw_path).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            return str(Path(raw_path).expanduser())
