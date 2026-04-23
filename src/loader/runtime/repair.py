"""Assistant-response repair and fallback helpers for the typed runtime."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.base import ToolCall
from .context import RuntimeContext
from .dod import (
    DefinitionOfDone,
    collect_planned_artifact_targets,
    infer_next_declared_html_output_file,
    planned_artifact_target_satisfied,
)
from .parsing import parse_tool_calls
from .workflow import effective_pending_todo_items, reconcile_aggregate_completion_steps

_SPECIAL_DOD_ITEMS = {
    "Complete the requested work",
    "Collect verification evidence",
}
_LATE_STAGE_EMPTY_RETRY_EXTRA = 2
_WORKING_NOTE_TOOL_NAMES = (
    "notepad_write_working",
    "notepad_append",
    "notepad_write_priority",
    "notepad_write_manual",
)
_MUTATION_TODO_HINTS = (
    "create",
    "creating",
    "update",
    "updating",
    "edit",
    "editing",
    "write",
    "writing",
    "fix",
    "fixing",
    "modify",
    "modifying",
    "change",
    "changing",
    "patch",
    "patching",
    "replace",
    "replacing",
    "correct",
    "correcting",
    "rewrite",
    "rewriting",
)
_CONSISTENCY_REVIEW_HINTS = (
    "consistent",
    "consistently",
    "formatted",
    "link",
    "linked",
    "navigation",
    "work properly",
    "all files",
    "every file",
)


@dataclass(slots=True)
class EmptyResponseDecision:
    """Decision for an empty assistant response."""

    should_continue: bool
    reason_code: str | None = None
    reason_summary: str | None = None
    retry_message: str | None = None
    final_response: str | None = None
    failure: str | None = None


@dataclass(slots=True)
class ToolCallAnalysis:
    """Normalized assistant-output analysis for tool execution."""

    content: str
    response_content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_source: str = "native"
    clear_stream: bool = False
    is_final_answer: bool = False
    should_stop: bool = False
    reason_code: str | None = None
    reason_summary: str | None = None
    final_response: str | None = None
    failure: str | None = None
    extracted_iterations: int = 0


class ResponseRepairer:
    """Owns response-repair heuristics that used to live inline in the loop."""

    def __init__(self, context: RuntimeContext) -> None:
        self.context = context

    def handle_empty_response(
        self,
        *,
        task: str,
        original_task: str | None,
        empty_retry_count: int,
        max_empty_retries: int,
        dod: DefinitionOfDone | None = None,
    ) -> EmptyResponseDecision:
        """Return the next action when the assistant responds with empty content."""

        _ = task, original_task
        effective_max_empty_retries = self._effective_max_empty_retries(
            dod,
            base_max_empty_retries=max_empty_retries,
        )
        if empty_retry_count <= effective_max_empty_retries:
            return EmptyResponseDecision(
                should_continue=True,
                reason_code="empty_response_retry",
                reason_summary=(
                    "retried after the assistant returned an empty response"
                ),
                retry_message=self._build_empty_response_retry_message(
                    dod,
                    retry_number=empty_retry_count,
                    max_empty_retries=effective_max_empty_retries,
                ),
            )

        return EmptyResponseDecision(
            should_continue=False,
            reason_code="empty_response_retry_exhausted",
            reason_summary="stopped after the assistant returned empty responses repeatedly",
            final_response=(
                "I didn't get a usable response from the model after "
                f"retrying {effective_max_empty_retries} times. Please try again or "
                "switch to a different backend/model."
            ),
            failure="assistant returned empty output repeatedly",
        )

    def analyze_response(
        self,
        *,
        content: str,
        response_content: str,
        tool_calls: list[ToolCall],
        extracted_iterations: int,
        max_extracted_iterations: int,
    ) -> ToolCallAnalysis:
        """Normalize assistant output into final-answer, tool, or repair outcomes."""

        normalized_content = content
        normalized_tool_calls = list(tool_calls)
        tool_source = "native"

        if self.context.use_react:
            parsed = parse_tool_calls(content)
            normalized_tool_calls = parsed.tool_calls
            normalized_content = parsed.content

            if parsed.is_final_answer and not normalized_tool_calls:
                return ToolCallAnalysis(
                    content=normalized_content,
                    response_content=response_content,
                    is_final_answer=True,
                    final_response=normalized_content,
                )

        clear_stream = False
        next_extracted_iterations = extracted_iterations
        if not normalized_tool_calls:
            raw_tool_calls = self._extract_raw_tool_calls(response_content)
            if raw_tool_calls:
                normalized_tool_calls = raw_tool_calls
                tool_source = "raw_text"
                clear_stream = True

        if normalized_tool_calls and tool_source == "raw_text":
            next_extracted_iterations += 1
            if next_extracted_iterations > max_extracted_iterations:
                return ToolCallAnalysis(
                    content=normalized_content,
                    response_content=response_content,
                    tool_calls=normalized_tool_calls,
                    tool_source=tool_source,
                    clear_stream=clear_stream,
                    extracted_iterations=next_extracted_iterations,
                    should_stop=True,
                    reason_code="raw_text_tool_recovery_exhausted",
                    reason_summary=(
                        "stopped after raw-text tool recovery budget was exhausted"
                    ),
                    final_response=(
                        "I couldn't safely continue because the model kept emitting "
                        "raw-text tool calls instead of proper tool invocations. "
                        "Please try again or switch to a different backend/model."
                    ),
                    failure="raw-text tool recovery budget exhausted",
                )

        return ToolCallAnalysis(
            content=normalized_content,
            response_content=response_content,
            tool_calls=normalized_tool_calls,
            tool_source=tool_source,
            clear_stream=clear_stream,
            reason_code=(
                "raw_text_tool_recovered" if tool_source == "raw_text" else None
            ),
            reason_summary=(
                "recovered raw-text tool calls into executable tool invocations"
                if tool_source == "raw_text"
                else None
            ),
            extracted_iterations=next_extracted_iterations,
        )

    def _extract_raw_tool_calls(self, response_content: str) -> list[ToolCall]:
        """Recover raw-text tool calls from the runtime parser and registry."""

        allowed_tool_names = [
            tool.name for tool in self.context.registry.list_tools()
        ]
        parsed = parse_tool_calls(
            response_content,
            allowed_tool_names=allowed_tool_names,
        )
        return parsed.tool_calls

    def _build_empty_response_retry_message(
        self,
        dod: DefinitionOfDone | None,
        *,
        retry_number: int,
        max_empty_retries: int,
    ) -> str:
        progress_lines: list[str] = []
        if dod is not None:
            reconcile_aggregate_completion_steps(
                dod,
                project_root=self.context.project_root,
            )
            latest_working_note = self._latest_working_note()
            if latest_working_note:
                progress_lines.append(
                    "Latest working note: " + latest_working_note
                )

            planned_lines = self._planned_artifact_progress_lines(dod)
            progress_lines.extend(planned_lines)
            progress_lines.extend(
                self._next_step_resume_lines(
                    dod,
                    retry_number=retry_number,
                )
            )

            touched = [
                f"`{Path(path).name or path}`"
                for path in dod.touched_files[-3:]
                if str(path).strip()
            ]
            if touched:
                progress_lines.append(
                    "Confirmed touched files: " + ", ".join(touched)
                )

            completed = [
                item
                for item in dod.completed_items
                if item not in _SPECIAL_DOD_ITEMS
            ]
            if completed:
                progress_lines.append(
                    "Confirmed completed work: " + "; ".join(completed[-2:])
                )

            next_pending = next(
                (
                    item
                    for item in dod.pending_items
                    if item not in _SPECIAL_DOD_ITEMS
                ),
                None,
            )
            if next_pending:
                progress_lines.append(f"Next pending item: {next_pending}")
            todo_refresh = self._todo_refresh_retry_line(dod)
            if todo_refresh:
                progress_lines.append(todo_refresh)

        if not progress_lines:
            return (
                "[EMPTY ASSISTANT RESPONSE]\n"
                f"Your last response was empty (retry {retry_number}/{max_empty_retries}). "
                "Respond directly to the task "
                "or call tools if needed. Do not return an empty response."
            )

        return "\n".join(
            [
                "[EMPTY ASSISTANT RESPONSE]",
                (
                    "Your last response was empty "
                    f"(retry {retry_number}/{max_empty_retries}). Continue from the "
                    "confirmed progress below instead of restarting."
                ),
                *[f"- {line}" for line in progress_lines],
                "",
                "Respond directly to the task or call tools if needed. Do not return an empty response.",
            ]
        )

    def _todo_refresh_retry_line(self, dod: DefinitionOfDone) -> str | None:
        non_special_pending = [
            item for item in dod.pending_items if item not in _SPECIAL_DOD_ITEMS
        ]
        non_special_completed = [
            item for item in dod.completed_items if item not in _SPECIAL_DOD_ITEMS
        ]
        if len(dod.touched_files) < 2 and (len(non_special_pending) + len(non_special_completed)) < 3:
            return None
        return (
            "If the tracked steps are stale, refresh `TodoWrite` alongside the next "
            "concrete mutation instead of spending a full turn on bookkeeping alone."
        )

    def _effective_max_empty_retries(
        self,
        dod: DefinitionOfDone | None,
        *,
        base_max_empty_retries: int,
    ) -> int:
        if dod is None:
            return base_max_empty_retries
        completed_artifacts, missing_artifacts = self._planned_artifact_counts(dod)
        if completed_artifacts < 3 or missing_artifacts == 0:
            return base_max_empty_retries
        return base_max_empty_retries + _LATE_STAGE_EMPTY_RETRY_EXTRA

    def _planned_artifact_counts(self, dod: DefinitionOfDone) -> tuple[int, int]:
        completed = 0
        missing = 0
        for target, expect_directory in collect_planned_artifact_targets(
            dod,
            project_root=self.context.project_root,
            max_paths=12,
        ):
            if planned_artifact_target_satisfied(
                dod,
                target=target,
                expect_directory=expect_directory,
                project_root=self.context.project_root,
            ):
                completed += 1
            else:
                missing += 1
        return completed, missing

    def _planned_artifact_progress_lines(self, dod: DefinitionOfDone) -> list[str]:
        targets = collect_planned_artifact_targets(
            dod,
            project_root=self.context.project_root,
            max_paths=12,
        )
        if not targets:
            return []

        missing_labels = [
            self._format_artifact_label(target, expect_directory=expect_directory)
            for target, expect_directory in targets
            if not planned_artifact_target_satisfied(
                dod,
                target=target,
                expect_directory=expect_directory,
                project_root=self.context.project_root,
            )
        ]
        if not missing_labels:
            return []

        lines = [f"Next missing planned artifact: {missing_labels[0]}"]
        first_missing_target, first_missing_is_directory = next(
            (
                (target, expect_directory)
                for target, expect_directory in targets
                if not planned_artifact_target_satisfied(
                    dod,
                    target=target,
                    expect_directory=expect_directory,
                    project_root=self.context.project_root,
                )
            ),
            (None, False),
        )
        if first_missing_target is not None and first_missing_is_directory:
            next_output_file = infer_next_declared_html_output_file(
                target=first_missing_target,
                project_root=self.context.project_root,
            )
            if next_output_file is not None:
                lines.append(
                    "Next declared output under "
                    f"{self._format_artifact_label(first_missing_target, expect_directory=True)}: "
                    f"{self._format_artifact_label(next_output_file, expect_directory=False)}"
                )
        if len(missing_labels) > 1:
            preview = ", ".join(missing_labels[:3])
            if len(missing_labels) > 3:
                preview += ", ..."
            lines.append("Remaining planned artifacts: " + preview)
        return lines

    def _next_step_resume_lines(
        self,
        dod: DefinitionOfDone,
        *,
        retry_number: int,
    ) -> list[str]:
        completed_artifacts, _ = self._planned_artifact_counts(dod)
        next_pending = next(
            (
                item
                for item in effective_pending_todo_items(
                    dod,
                    project_root=self.context.project_root,
                )
                if item not in _SPECIAL_DOD_ITEMS
            ),
            None,
        )
        if (
            completed_artifacts == 0
            and next_pending
            and not _todo_is_mutation_step(next_pending)
            and not _todo_is_consistency_review_step(next_pending)
        ):
            lines = [f"Resume with this exact next step: advance `{next_pending}`."]
            lines.append(
                "Make the next response one concrete evidence-gathering tool call that "
                "directly advances that step."
            )
            lines.append(
                "Do not jump ahead to later artifact creation, verification, or a "
                "completion summary until that discovery step is satisfied."
            )
            if retry_number >= 2:
                lines.append(
                    "Do not restart from scratch or return another working note; emit the "
                    "next evidence-gathering tool call now."
                )
            else:
                lines.append(
                    "Do not restart from scratch unless one specific missing fact blocks "
                    "that discovery step."
                )
            return lines

        for target, expect_directory in collect_planned_artifact_targets(
            dod,
            project_root=self.context.project_root,
            max_paths=12,
        ):
            if planned_artifact_target_satisfied(
                dod,
                target=target,
                expect_directory=expect_directory,
                project_root=self.context.project_root,
            ):
                continue
            label = self._format_artifact_label(
                target,
                expect_directory=expect_directory,
            )
            if expect_directory:
                next_output_file = infer_next_declared_html_output_file(
                    target=target,
                    project_root=self.context.project_root,
                )
                if next_output_file is not None:
                    next_output_label = self._format_artifact_label(
                        next_output_file,
                        expect_directory=False,
                    )
                    if next_pending and _todo_is_mutation_step(next_pending):
                        lines = [
                            "Resume with this exact next step: continue "
                            f"`{next_pending}` by creating {next_output_label}."
                        ]
                    else:
                        lines = [
                            "Resume with this exact next step: create "
                            f"{next_output_label}."
                        ]
                    lines.append(
                        f"It is the next missing declared output under {label}."
                    )
                    lines.append(
                        f"Prefer one `write` call for `{next_output_file}` before more research."
                    )
                    if not next_output_file.parent.exists():
                        lines.append(
                            "The `write` tool can create that file's parent directories "
                            "automatically, so do the write in one step instead of stopping "
                            "for a separate mkdir."
                        )
                    if retry_number >= 2:
                        lines.append(
                            "Do not restart discovery; emit the next mutation tool call now."
                        )
                    else:
                        lines.append(
                            "Do not restart discovery unless one specific missing fact blocks this step."
                        )
                    return lines
            if expect_directory and target.is_dir():
                if next_pending and _todo_is_mutation_step(next_pending):
                    lines = [
                        "Resume with this exact next step: continue "
                        f"`{next_pending}` by creating the next output file under {label}."
                    ]
                else:
                    lines = [
                        "Resume with this exact next step: create the next output file "
                        f"under {label}."
                    ]
                lines.append(
                    f"Prefer one concrete `write` call for a file inside `{target}` before more research."
                )
            else:
                lines = [f"Resume with this exact next step: create {label}."]
            if expect_directory and not target.is_dir():
                lines.append(
                    f"Prefer one concrete directory-creation step for `{target}` before more research."
                )
            elif not expect_directory:
                lines.append(
                    f"Prefer one `write` call for `{target}` before any more reference reads."
                )
                if not target.parent.exists():
                    lines.append(
                        "The `write` tool can create that file's parent directories "
                        "automatically, so do the write in one step instead of stopping "
                        "for a separate mkdir."
                    )
                lines.append(
                    "Shape the next response as one concrete `write(file_path=..., "
                    "content=...)` tool call for that exact path."
                )
            if completed_artifacts >= 3:
                lines.append(
                    "Follow the same one-file-at-a-time mutation pattern that already "
                    "created the confirmed planned artifacts."
                )
            lines.append(
                "Your next response should be the concrete mutation tool call itself, "
                "not TodoWrite alone, verification, or a completion summary."
            )
            if retry_number >= 2:
                lines.append(
                    "Do not restart discovery; emit the next mutation tool call now."
                )
            else:
                lines.append(
                    "Do not restart discovery unless one specific missing fact blocks this step."
                )
            return lines
        return []

    @staticmethod
    def _format_artifact_label(path: Path, *, expect_directory: bool) -> str:
        label = path.name or str(path)
        if expect_directory and not label.endswith("/"):
            label += "/"
        return f"`{label}`"

    def _latest_working_note(self) -> str | None:
        messages = list(getattr(self.context.session, "messages", []) or [])
        for message in reversed(messages):
            content = str(getattr(message, "content", "") or "").strip()
            if not content:
                continue
            for tool_name in _WORKING_NOTE_TOOL_NAMES:
                prefix = f"Observation [{tool_name}]: Result:"
                if prefix not in content:
                    continue
                note = content.split(prefix, 1)[1].strip()
                if not note:
                    continue
                first_line = next(
                    (line.strip() for line in note.splitlines() if line.strip()),
                    "",
                )
                if not first_line:
                    continue
                first_line = re.sub(r"^-\s*\[[^\]]+\]\s*", "", first_line).strip()
                return first_line or None
        return None


def _todo_is_mutation_step(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in _MUTATION_TODO_HINTS)


def _todo_is_consistency_review_step(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in _CONSISTENCY_REVIEW_HINTS)
