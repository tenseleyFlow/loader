"""Assistant-response repair and fallback helpers for the typed runtime."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.base import ToolCall
from .context import RuntimeContext
from .dod import (
    DefinitionOfDone,
    collect_planned_artifact_targets,
    infer_next_output_file,
    planned_artifact_target_satisfied,
)
from .parsing import parse_tool_calls
from .path_display import display_runtime_path
from .recovery import detect_missing_mutation_payload
from .workflow import (
    infer_output_outline_label,
    infer_pending_todo_output_target,
    preferred_pending_todo_item,
    reconcile_aggregate_completion_steps,
    todo_describes_aggregate_mutation,
    todo_describes_broad_setup_step,
    todo_file_candidates,
)

_SPECIAL_DOD_ITEMS = {
    "Complete the requested work",
    "Collect verification evidence",
}
_FIRST_FILE_EMPTY_RETRY_EXTRA = 2
_LATE_STAGE_EMPTY_RETRY_EXTRA = 2
_MULTI_FILE_OUTPUT_EMPTY_RETRY_EXTRA = 2
_SUMMARY_ARTIFACT_NAMES = {
    "index.html",
    "index.htm",
    "readme",
    "readme.md",
    "readme.rst",
    "readme.txt",
}
_WORKING_NOTE_TOOL_NAMES = (
    "notepad_write_working",
    "notepad_append",
    "notepad_write_priority",
    "notepad_write_manual",
)
_MUTATION_TODO_HINTS = (
    "create",
    "creating",
    "develop",
    "developing",
    "build",
    "building",
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
        if dod is not None:
            minimal_retry_message = self._build_early_concrete_write_retry_message(
                dod,
                retry_number=retry_number,
                max_empty_retries=max_empty_retries,
            )
            if minimal_retry_message is not None:
                return minimal_retry_message
        if dod is not None and self._should_compact_empty_retry_message(dod):
            compact_lines: list[str] = []
            compact_lines.extend(self._compact_planned_artifact_lines(dod))
            compact_lines.extend(self._payload_retry_lines(dod))
            compact_lines.extend(
                self._next_step_resume_lines(
                    dod,
                    retry_number=retry_number,
                )
            )
            return "\n".join(
                [
                    "[EMPTY ASSISTANT RESPONSE]",
                    (
                        "Your last response was empty "
                        f"(retry {retry_number}/{max_empty_retries}). Continue from the "
                        "exact next step below."
                    ),
                    *[f"- {line}" for line in compact_lines],
                    "",
                    "Respond with that concrete mutation tool call now. Do not return an empty response.",
                ]
            )

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
            progress_lines.extend(self._payload_retry_lines(dod))
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

            preferred_missing_artifact = self._preferred_resume_missing_artifact(dod)
            next_pending = self._preferred_resume_pending_item(
                dod,
                missing_artifact=preferred_missing_artifact,
            )
            resume_already_names_pending = bool(
                next_pending
                and any(f"`{next_pending}`" in line for line in progress_lines)
            )
            if next_pending and not resume_already_names_pending:
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

    def _build_early_concrete_write_retry_message(
        self,
        dod: DefinitionOfDone,
        *,
        retry_number: int,
        max_empty_retries: int,
    ) -> str | None:
        if retry_number < 3:
            return None
        if not self._has_confirmed_output_file_progress(dod):
            return None
        if self._has_confirmed_substantive_output_file_progress(dod):
            return None

        next_missing_artifact = self._preferred_resume_missing_artifact(dod)
        next_pending = self._preferred_resume_pending_item(
            dod,
            missing_artifact=next_missing_artifact,
        )
        inferred_pending_target = (
            self._infer_pending_item_output_target(dod, next_pending)
            if next_pending
            else None
        )
        concrete_target: Path | None = None
        if inferred_pending_target is not None and not inferred_pending_target.exists():
            concrete_target = inferred_pending_target.expanduser().resolve(strict=False)
        elif next_missing_artifact is not None and not next_missing_artifact[1]:
            concrete_target = next_missing_artifact[0].expanduser().resolve(strict=False)
        if concrete_target is None or not concrete_target.suffix:
            return None

        outline_label = infer_output_outline_label(
            dod,
            concrete_target,
            project_root=self.context.project_root,
            todo_label=next_pending or "",
        )
        if next_pending and _todo_is_mutation_step(next_pending):
            first_line = (
                f"Continue `{next_pending}` by creating `{concrete_target.name}`."
            )
        else:
            first_line = f"Create `{concrete_target.name}` now."
        compact_retry = retry_number >= 4

        lines = [
            first_line,
            self._mutation_tool_scaffold(concrete_target, tool_name="write"),
        ]
        if outline_label:
            lines.append(
                f"Use the existing outline label `{outline_label}` for that file so it matches the current guide structure."
            )
        html_scaffold_line = self._known_existing_html_scaffold_line(
            concrete_target,
            require_first_substantive_output=True,
        )
        if html_scaffold_line:
            lines.append(html_scaffold_line)
        reference_line = self._known_reference_structure_line(
            concrete_target,
            require_first_substantive_output=True,
        )
        if reference_line and not compact_retry:
            lines.append(reference_line)
        reference_cues_line = self._known_reference_cues_line(
            concrete_target,
            require_first_substantive_output=True,
            retry_number=retry_number,
        )
        if reference_cues_line and not compact_retry:
            lines.append(reference_cues_line)
        html_starter_line = self._known_html_starter_shape_line(
            concrete_target,
            require_first_substantive_output=True,
            retry_number=retry_number,
            outline_label=outline_label,
        )
        if html_starter_line:
            lines.append(html_starter_line)
        if (
            not compact_retry
            and _should_encourage_initial_version(
                target=concrete_target,
                has_confirmed_output_file_progress=True,
                has_confirmed_substantive_output_file_progress=False,
            )
        ):
            lines.append(
                "Write a compact but real initial version of this file now, then refine or expand it in later edits."
            )
        lines.append(
            "No narration, no TodoWrite, no rereads, and no empty response; emit the mutation tool call now."
        )
        return "\n".join(
            [
                "[EMPTY ASSISTANT RESPONSE]",
                (
                    "Your last response was empty "
                    f"(retry {retry_number}/{max_empty_retries}). Emit the exact next mutation now."
                ),
                *[f"- {line}" for line in lines],
            ]
        )

    def _payload_retry_lines(self, dod: DefinitionOfDone | None) -> list[str]:
        recovery_context = self.context.recovery_context
        if recovery_context is None or not recovery_context.attempts:
            return []
        attempt = recovery_context.attempts[-1]
        fix = detect_missing_mutation_payload(
            attempt.tool_name,
            attempt.arguments,
            attempt.error,
        )
        if fix is None:
            return []

        target = fix["file_path"] or self._preferred_retry_target(dod)
        invalid = ", ".join(f"`{field}`" for field in fix["invalid_fields"])
        display_target = display_runtime_path(target) if target else None
        if fix.get("kind") == "missing_target":
            if attempt.tool_name == "write":
                target_line = (
                    f"Last tool failure: resend `write` for `{display_target}` with a valid `file_path` and real `content`."
                    if display_target
                    else "Last tool failure: resend `write` with a valid `file_path` and real `content`."
                )
                return [
                    target_line,
                    "Do not leave `file_path` empty; point it at the concrete next output file.",
                    self._mutation_tool_scaffold(
                        Path(target),
                        tool_name="write",
                    )
                    if target
                    else "Emit the `write(file_path=..., content=\"...\")` call with the real target path now.",
                ]
            if attempt.tool_name == "edit":
                target_line = (
                    f"Last tool failure: resend `edit` for `{display_target}` with a valid `file_path` plus real `old_string`/`new_string`."
                    if display_target
                    else "Last tool failure: resend `edit` with a valid `file_path` plus real `old_string`/`new_string`."
                )
                return [
                    target_line,
                    "Do not leave `file_path` empty; point it at the concrete file you already know needs the edit.",
                    self._mutation_tool_scaffold(
                        Path(target),
                        tool_name="edit",
                    )
                    if target
                    else "Emit the `edit(file_path=..., old_string=\"...\", new_string=\"...\")` call with the real target path now.",
                ]
            if attempt.tool_name == "patch":
                target_line = (
                    f"Last tool failure: resend `patch` for `{display_target}` with a valid `file_path` and real patch text or `hunks`."
                    if display_target
                    else "Last tool failure: resend `patch` with a valid `file_path` and real patch text or `hunks`."
                )
                return [
                    target_line,
                    "Do not leave `file_path` empty; point it at the concrete file you already know needs the patch.",
                    self._mutation_tool_scaffold(
                        Path(target),
                        tool_name="patch",
                    )
                    if target
                    else "Emit the `patch(file_path=..., patch=\"...\")` call with the real target path now.",
                ]
        if attempt.tool_name == "write":
            lines = [
                (
                    f"Last tool failure: resend `write` for `{display_target}` with real `content`, not just summary fields."
                    if display_target
                    else "Last tool failure: resend `write` with real `content`, not just summary fields."
                ),
            ]
            lines.append(f"Do not use {invalid} in place of the actual file body.")
            if target:
                lines.append(
                    self._mutation_tool_scaffold(
                        Path(target),
                        tool_name="write",
                    )
                )
            return lines
        if attempt.tool_name == "edit":
            lines = [
                (
                    f"Last tool failure: resend `edit` for `{display_target}` with the real text payload."
                    if display_target
                    else "Last tool failure: resend `edit` with the real text payload."
                ),
                f"Do not use {invalid} in place of `old_string`/`new_string`.",
            ]
            if target:
                lines.append(
                    self._mutation_tool_scaffold(
                        Path(target),
                        tool_name="edit",
                    )
                )
            return lines
        if attempt.tool_name == "patch":
            lines = [
                (
                    f"Last tool failure: resend `patch` for `{display_target}` with real patch text or structured hunks."
                    if display_target
                    else "Last tool failure: resend `patch` with real patch text or structured hunks."
                ),
                f"Do not use {invalid} in place of the real patch payload.",
            ]
            if target:
                lines.append(
                    self._mutation_tool_scaffold(
                        Path(target),
                        tool_name="patch",
                    )
                )
            return lines
        return []

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
        if completed_artifacts >= 3 and missing_artifacts > 0:
            return base_max_empty_retries + _LATE_STAGE_EMPTY_RETRY_EXTRA
        if self._has_concrete_next_output_step(dod):
            extra_retries = _LATE_STAGE_EMPTY_RETRY_EXTRA
            if self._has_confirmed_substantive_output_file_progress(dod):
                extra_retries += _MULTI_FILE_OUTPUT_EMPTY_RETRY_EXTRA
            elif completed_artifacts > 0:
                extra_retries += _FIRST_FILE_EMPTY_RETRY_EXTRA
            return base_max_empty_retries + extra_retries
        return base_max_empty_retries

    def _should_compact_empty_retry_message(self, dod: DefinitionOfDone) -> bool:
        completed_artifacts, missing_artifacts = self._planned_artifact_counts(dod)
        if completed_artifacts >= 3:
            return missing_artifacts > 0
        return self._has_concrete_next_output_step(dod)

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

    def _has_concrete_next_output_step(self, dod: DefinitionOfDone) -> bool:
        next_missing_artifact = next(
            (
                artifact
                for artifact in collect_planned_artifact_targets(
                    dod,
                    project_root=self.context.project_root,
                    max_paths=12,
                )
                if not planned_artifact_target_satisfied(
                    dod,
                    target=artifact[0],
                    expect_directory=artifact[1],
                    project_root=self.context.project_root,
                )
            ),
            None,
        )
        next_pending = self._preferred_resume_pending_item(
            dod,
            missing_artifact=next_missing_artifact,
        )
        if next_pending and self._infer_pending_item_output_target(dod, next_pending):
            return True
        if next_missing_artifact is None:
            return False
        target, expect_directory = next_missing_artifact
        if not expect_directory:
            return True
        next_output_file, _ = infer_next_output_file(
            target=target,
            project_root=self.context.project_root,
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        return next_output_file is not None

    def _has_confirmed_output_file_progress(self, dod: DefinitionOfDone) -> bool:
        return any(
            not expect_directory
            and planned_artifact_target_satisfied(
                dod,
                target=target,
                expect_directory=False,
                project_root=self.context.project_root,
            )
            for target, expect_directory in collect_planned_artifact_targets(
                dod,
                project_root=self.context.project_root,
                max_paths=12,
            )
        )

    def _has_confirmed_substantive_output_file_progress(
        self,
        dod: DefinitionOfDone,
    ) -> bool:
        for raw_path in dod.touched_files:
            if not str(raw_path).strip():
                continue
            path = Path(raw_path).expanduser().resolve(strict=False)
            if not path.suffix or _is_summary_artifact_path(path) or not path.is_file():
                continue
            return True
        return any(
            not expect_directory
            and not _is_summary_artifact_path(target)
            and planned_artifact_target_satisfied(
                dod,
                target=target,
                expect_directory=False,
                project_root=self.context.project_root,
            )
            for target, expect_directory in collect_planned_artifact_targets(
                dod,
                project_root=self.context.project_root,
                max_paths=12,
            )
        )

    def _planned_artifact_progress_lines(self, dod: DefinitionOfDone) -> list[str]:
        targets = collect_planned_artifact_targets(
            dod,
            project_root=self.context.project_root,
            max_paths=12,
        )
        if not targets:
            return []

        preferred_missing_artifact = self._preferred_resume_missing_artifact(dod)
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

        if preferred_missing_artifact is not None:
            preferred_label = self._format_artifact_label(
                preferred_missing_artifact[0],
                expect_directory=preferred_missing_artifact[1],
            )
            ordered_labels = [preferred_label, *missing_labels]
            missing_labels = list(dict.fromkeys(ordered_labels))

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
        detail_target = (
            preferred_missing_artifact
            if preferred_missing_artifact is not None
            else (
                (first_missing_target, first_missing_is_directory)
                if first_missing_target is not None
                else None
            )
        )
        if detail_target is not None and detail_target[1]:
            detail_path = detail_target[0]
            next_output_file, next_output_source = infer_next_output_file(
                target=detail_path,
                project_root=self.context.project_root,
                messages=list(getattr(self.context.session, "messages", []) or []),
            )
            if next_output_file is not None:
                next_output_detail = (
                    "Next declared output under "
                    if next_output_source == "declared"
                    else "Next observed output pattern under "
                )
                lines.append(
                    next_output_detail
                    + f"{self._format_artifact_label(detail_path, expect_directory=True)}: "
                    f"{self._format_artifact_label(next_output_file, expect_directory=False)}"
                )
        if len(missing_labels) > 1:
            preview = ", ".join(missing_labels[:3])
            if len(missing_labels) > 3:
                preview += ", ..."
            lines.append("Remaining planned artifacts: " + preview)
        return lines

    def _compact_planned_artifact_lines(self, dod: DefinitionOfDone) -> list[str]:
        lines = self._planned_artifact_progress_lines(dod)
        if self._confirmed_output_file_count(dod) < 2:
            return lines[:1]
        return lines[:2]

    def _confirmed_output_file_count(self, dod: DefinitionOfDone) -> int:
        return sum(
            1
            for target, expect_directory in collect_planned_artifact_targets(
                dod,
                project_root=self.context.project_root,
                max_paths=12,
            )
            if not expect_directory
            and planned_artifact_target_satisfied(
                dod,
                target=target,
                expect_directory=False,
                project_root=self.context.project_root,
            )
        )

    def _next_step_resume_lines(
        self,
        dod: DefinitionOfDone,
        *,
        retry_number: int,
    ) -> list[str]:
        completed_artifacts, _ = self._planned_artifact_counts(dod)
        has_confirmed_output_file_progress = self._has_confirmed_output_file_progress(dod)
        has_confirmed_substantive_output_file_progress = (
            self._has_confirmed_substantive_output_file_progress(dod)
        )
        next_missing_artifact = self._preferred_resume_missing_artifact(dod)
        next_pending = self._preferred_resume_pending_item(
            dod,
            missing_artifact=next_missing_artifact,
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

        inferred_pending_target = (
            self._infer_pending_item_output_target(dod, next_pending)
            if next_pending
            else None
        )
        if (
            next_pending
            and inferred_pending_target is None
            and next_missing_artifact is not None
            and not next_missing_artifact[1]
            and todo_describes_aggregate_mutation(next_pending)
            and not todo_describes_broad_setup_step(next_pending)
        ):
            concrete_target = next_missing_artifact[0]
            outline_label = infer_output_outline_label(
                dod,
                concrete_target,
                project_root=self.context.project_root,
                todo_label=next_pending,
            )
            lines = [
                f"Resume with this exact next step: create `{concrete_target.name}`.",
                f"It is the next concrete output needed to continue `{next_pending}`.",
                "Prefer one `write(content=...)` call for "
                f"`{display_runtime_path(concrete_target)}` before more research.",
                self._mutation_tool_scaffold(
                    concrete_target,
                    tool_name="write",
                ),
            ]
            if not concrete_target.parent.exists():
                lines.append(
                    "The `write` tool can create that file's parent directories "
                    "automatically, so do the write in one step instead of stopping "
                    "for a separate mkdir."
                )
            if outline_label:
                lines.append(
                    f"Use the existing outline label `{outline_label}` for that file so it matches the current guide structure."
                )
            reference_line = self._known_reference_structure_line(
                concrete_target,
                require_first_substantive_output=(
                    has_confirmed_output_file_progress
                    and not has_confirmed_substantive_output_file_progress
                ),
            )
            if reference_line:
                lines.append(reference_line)
            reference_cues_line = self._known_reference_cues_line(
                concrete_target,
                require_first_substantive_output=(
                    has_confirmed_output_file_progress
                    and not has_confirmed_substantive_output_file_progress
                ),
                retry_number=retry_number,
            )
            if reference_cues_line:
                lines.append(reference_cues_line)
            html_scaffold_line = self._known_existing_html_scaffold_line(
                concrete_target,
                require_first_substantive_output=(
                    has_confirmed_output_file_progress
                    and not has_confirmed_substantive_output_file_progress
                ),
            )
            if html_scaffold_line:
                lines.append(html_scaffold_line)
            html_starter_line = self._known_html_starter_shape_line(
                concrete_target,
                require_first_substantive_output=(
                    has_confirmed_output_file_progress
                    and not has_confirmed_substantive_output_file_progress
                ),
                retry_number=retry_number,
                outline_label=outline_label,
            )
            if html_starter_line:
                lines.append(html_starter_line)
            if _should_encourage_initial_version(
                target=concrete_target,
                has_confirmed_output_file_progress=has_confirmed_output_file_progress,
                has_confirmed_substantive_output_file_progress=has_confirmed_substantive_output_file_progress,
            ):
                lines.append(
                    "Do not wait to perfect the entire multi-file output before this write. "
                    "Write a compact but real initial version of this file now, then refine "
                    "or expand it in later edits."
                )
            if has_confirmed_substantive_output_file_progress:
                lines.append(
                    "Follow the same full-payload one-file-at-a-time write pattern that "
                    "already created the confirmed output files."
                )
            if retry_number >= 2:
                lines.append(
                    "Do not return another working note or empty response; emit the "
                    "concrete mutation tool call now."
                )
            else:
                lines.append(
                    "Do not restart discovery unless one specific missing fact blocks "
                    "that file write."
                )
            return lines
        if next_pending and inferred_pending_target is not None:
            inferred_is_directory = not bool(inferred_pending_target.suffix)
            inferred_label = self._format_artifact_label(
                inferred_pending_target,
                expect_directory=inferred_is_directory,
            )
            outline_label = infer_output_outline_label(
                dod,
                inferred_pending_target,
                project_root=self.context.project_root,
                todo_label=next_pending,
            )
            lines = [
                "Resume with this exact next step: continue "
                f"`{next_pending}` by creating {inferred_label}."
            ]
            if inferred_is_directory:
                lines.append(
                    "Prefer one concrete directory-creation step for "
                    f"`{display_runtime_path(inferred_pending_target)}` before more research."
                )
                lines.append(
                    self._directory_creation_scaffold(inferred_pending_target)
                )
            else:
                lines.append(
                    "Prefer one `write(content=...)` call for "
                    f"`{display_runtime_path(inferred_pending_target)}` before more research."
                )
                lines.append(
                    self._mutation_tool_scaffold(
                        inferred_pending_target,
                        tool_name="write",
                    )
                )
            if outline_label:
                lines.append(
                    f"Use the existing outline label `{outline_label}` for that file so it matches the current guide structure."
                )
            self._append_concrete_html_write_cues(
                lines,
                target=inferred_pending_target,
                outline_label=outline_label,
                retry_number=retry_number,
                has_confirmed_output_file_progress=has_confirmed_output_file_progress,
                has_confirmed_substantive_output_file_progress=has_confirmed_substantive_output_file_progress,
            )
            if todo_describes_aggregate_mutation(next_pending):
                lines.insert(
                    1,
                    f"It is the next concrete output needed to continue `{next_pending}`.",
                )
            if (
                not inferred_is_directory
                and _should_encourage_initial_version(
                    target=inferred_pending_target,
                    has_confirmed_output_file_progress=has_confirmed_output_file_progress,
                    has_confirmed_substantive_output_file_progress=has_confirmed_substantive_output_file_progress,
                )
            ):
                lines.append(
                    "Do not wait to perfect the entire multi-file output before this write. "
                    "Write a compact but real initial version of this file now, then refine "
                    "or expand it in later edits."
                )
            if has_confirmed_substantive_output_file_progress:
                lines.append(
                    "Follow the same full-payload one-file-at-a-time write pattern that "
                    "already created the confirmed output files."
                )
            if retry_number >= 2:
                lines.append(
                    "Do not return another working note or empty response; emit the "
                    "concrete mutation tool call now."
                )
            else:
                lines.append(
                    "Do not restart discovery unless one specific missing fact blocks "
                    "that file write."
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
                next_output_file, next_output_source = infer_next_output_file(
                    target=target,
                    project_root=self.context.project_root,
                    messages=list(getattr(self.context.session, "messages", []) or []),
                )
                if next_output_file is not None:
                    next_output_label = self._format_artifact_label(
                        next_output_file,
                        expect_directory=False,
                    )
                    outline_label = infer_output_outline_label(
                        dod,
                        next_output_file,
                        project_root=self.context.project_root,
                        todo_label=next_pending or "",
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
                        if next_output_source == "declared"
                        else (
                            "It mirrors the observed filename pattern from another "
                            f"{label} directory you already inspected."
                        )
                    )
                    lines.append(
                        "Prefer one `write` call for "
                        f"`{display_runtime_path(next_output_file)}` before more research."
                    )
                    lines.append(
                        self._mutation_tool_scaffold(
                            next_output_file,
                            tool_name="write",
                        )
                    )
                    if outline_label:
                        lines.append(
                            f"Use the existing outline label `{outline_label}` for that file so it matches the current guide structure."
                        )
                    self._append_concrete_html_write_cues(
                        lines,
                        target=next_output_file,
                        outline_label=outline_label,
                        retry_number=retry_number,
                        has_confirmed_output_file_progress=has_confirmed_output_file_progress,
                        has_confirmed_substantive_output_file_progress=has_confirmed_substantive_output_file_progress,
                    )
                    if _should_encourage_initial_version(
                        target=next_output_file,
                        has_confirmed_output_file_progress=has_confirmed_output_file_progress,
                        has_confirmed_substantive_output_file_progress=has_confirmed_substantive_output_file_progress,
                    ):
                        lines.append(
                            "Do not wait to perfect the entire multi-file output before this write. "
                            "Write a compact but real initial version of this file now, then refine "
                            "or expand it in later edits."
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
                    "Prefer one concrete `write` call for a file inside "
                    f"`{display_runtime_path(target)}` before more research."
                )
            else:
                lines = [f"Resume with this exact next step: create {label}."]
            if expect_directory and not target.is_dir():
                lines.append(
                    "Prefer one concrete directory-creation step for "
                    f"`{display_runtime_path(target)}` before more research."
                )
            elif not expect_directory:
                lines.append(
                    "Prefer one `write` call for "
                    f"`{display_runtime_path(target)}` before any more reference reads."
                )
                if not target.parent.exists():
                    lines.append(
                        "The `write` tool can create that file's parent directories "
                        "automatically, so do the write in one step instead of stopping "
                        "for a separate mkdir."
                    )
                if _should_encourage_initial_version(
                    target=target,
                    has_confirmed_output_file_progress=has_confirmed_output_file_progress,
                    has_confirmed_substantive_output_file_progress=has_confirmed_substantive_output_file_progress,
                ):
                    lines.append(
                        "Do not wait to perfect the entire multi-file output before this write. "
                        "Write a compact but real initial version of this file now, then refine "
                        "or expand it in later edits."
                    )
                lines.append(
                    self._mutation_tool_scaffold(
                        target,
                        tool_name="write",
                    )
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

    def _append_concrete_html_write_cues(
        self,
        lines: list[str],
        *,
        target: Path,
        outline_label: str | None,
        retry_number: int,
        has_confirmed_output_file_progress: bool,
        has_confirmed_substantive_output_file_progress: bool,
    ) -> None:
        first_substantive_output = (
            has_confirmed_output_file_progress
            and not has_confirmed_substantive_output_file_progress
        )
        reference_line = self._known_reference_structure_line(
            target,
            require_first_substantive_output=first_substantive_output,
        )
        if reference_line:
            lines.append(reference_line)
        reference_cues_line = self._known_reference_cues_line(
            target,
            require_first_substantive_output=first_substantive_output,
            retry_number=retry_number,
        )
        if reference_cues_line:
            lines.append(reference_cues_line)
        html_scaffold_line = self._known_existing_html_scaffold_line(
            target,
            require_first_substantive_output=first_substantive_output,
        )
        if html_scaffold_line:
            lines.append(html_scaffold_line)
        sibling_scaffold_line = self._known_existing_html_sibling_scaffold_line(
            target,
            outline_label=outline_label,
            require_existing_substantive_output=has_confirmed_substantive_output_file_progress,
        )
        if sibling_scaffold_line:
            lines.append(sibling_scaffold_line)
        html_starter_line = self._known_html_starter_shape_line(
            target,
            require_first_substantive_output=(
                first_substantive_output
                or (
                    has_confirmed_substantive_output_file_progress
                    and retry_number >= 4
                )
            ),
            retry_number=retry_number,
            outline_label=outline_label,
        )
        if html_starter_line:
            lines.append(html_starter_line)

    def _infer_pending_item_output_target(
        self,
        dod: DefinitionOfDone,
        item: str,
    ) -> Path | None:
        return infer_pending_todo_output_target(
            dod,
            item,
            project_root=self.context.project_root,
        )

    def _preferred_resume_pending_item(
        self,
        dod: DefinitionOfDone,
        *,
        missing_artifact: tuple[Path, bool] | None,
    ) -> str | None:
        preferred = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
            missing_artifact=missing_artifact,
        )
        if preferred:
            return preferred

        explicit_file_items = [
            item
            for item in dod.pending_items
            if item not in _SPECIAL_DOD_ITEMS
            and _todo_is_mutation_step(item)
            and todo_file_candidates(item)
        ]
        if explicit_file_items:
            return explicit_file_items[0]

        return next(
            (item for item in dod.pending_items if item not in _SPECIAL_DOD_ITEMS),
            None,
        )

    def _preferred_resume_missing_artifact(
        self,
        dod: DefinitionOfDone,
    ) -> tuple[Path, bool] | None:
        planned_targets = collect_planned_artifact_targets(
            dod,
            project_root=self.context.project_root,
            max_paths=12,
        )
        first_missing = next(
            (
                artifact
                for artifact in planned_targets
                if not planned_artifact_target_satisfied(
                    dod,
                    target=artifact[0],
                    expect_directory=artifact[1],
                    project_root=self.context.project_root,
                )
            ),
            None,
        )
        if first_missing is None:
            return None

        next_pending = self._preferred_resume_pending_item(
            dod,
            missing_artifact=first_missing,
        )
        if next_pending is None:
            return self._concretize_directory_missing_artifact(
                dod,
                first_missing,
                planned_targets=planned_targets,
            )

        inferred_target = self._infer_pending_item_output_target(dod, next_pending)
        if inferred_target is None or inferred_target.exists():
            return self._concretize_directory_missing_artifact(
                dod,
                first_missing,
                planned_targets=planned_targets,
            )

        normalized_target = inferred_target.expanduser().resolve(strict=False)
        for planned_target, expect_directory in planned_targets:
            normalized_planned = planned_target.expanduser().resolve(strict=False)
            if expect_directory:
                try:
                    normalized_target.relative_to(normalized_planned)
                except ValueError:
                    continue
                return normalized_target, False
            if normalized_planned == normalized_target:
                return normalized_target, False
        return first_missing

    def _preferred_retry_target(self, dod: DefinitionOfDone | None) -> str:
        if dod is None:
            return ""

        missing_artifact = self._preferred_resume_missing_artifact(dod)
        next_pending = self._preferred_resume_pending_item(
            dod,
            missing_artifact=missing_artifact,
        )
        if next_pending:
            pending_target = self._infer_pending_item_output_target(dod, next_pending)
            if pending_target is not None and not pending_target.exists():
                return str(pending_target)

        if missing_artifact is None:
            return ""

        target, expect_directory = missing_artifact
        if not expect_directory:
            return str(target)

        next_output_file, _ = infer_next_output_file(
            target=target,
            project_root=self.context.project_root,
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        if next_output_file is not None:
            return str(next_output_file)
        return str(target)

    def _concretize_directory_missing_artifact(
        self,
        dod: DefinitionOfDone,
        missing_artifact: tuple[Path, bool],
        *,
        planned_targets: list[tuple[Path, bool]],
    ) -> tuple[Path, bool]:
        target, expect_directory = missing_artifact
        if not expect_directory:
            return missing_artifact
        if any(
            not expect_dir
            and not planned_artifact_target_satisfied(
                dod,
                target=planned_target,
                expect_directory=expect_dir,
                project_root=self.context.project_root,
            )
            for planned_target, expect_dir in planned_targets
        ):
            return missing_artifact
        next_output_file, _ = infer_next_output_file(
            target=target,
            project_root=self.context.project_root,
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        if next_output_file is None or next_output_file.exists():
            return missing_artifact
        return next_output_file, False

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

    def _known_reference_structure_line(
        self,
        target: Path,
        *,
        require_first_substantive_output: bool,
    ) -> str | None:
        if not require_first_substantive_output:
            return None
        reference = self._best_known_reference_path(target)
        if reference is None:
            return None
        return (
            f"You already read `{display_runtime_path(reference)}`; reuse its overall "
            "structure as the starting pattern for this new file, then adapt the content "
            "to the current target."
        )

    def _known_reference_cues_line(
        self,
        target: Path,
        *,
        require_first_substantive_output: bool,
        retry_number: int,
    ) -> str | None:
        if not require_first_substantive_output or retry_number < 2:
            return None
        reference = self._best_known_reference_path(target)
        if reference is None:
            return None
        cues = self._reference_content_cues(reference)
        if not cues:
            return None
        return f"Reference cues from `{display_runtime_path(reference)}`: {cues}"

    def _known_existing_html_scaffold_line(
        self,
        target: Path,
        *,
        require_first_substantive_output: bool,
    ) -> str | None:
        if not require_first_substantive_output:
            return None
        if target.suffix.lower() not in {".html", ".htm"}:
            return None
        scaffold = self._best_known_root_html_scaffold(target)
        if scaffold is None:
            return None
        return (
            f"Reuse the existing `{display_runtime_path(scaffold)}` head/style/container "
            "pattern for this chapter so the guide stays visually consistent; only adapt "
            "the title, heading, and chapter body content."
        )

    def _known_existing_html_sibling_scaffold_line(
        self,
        target: Path,
        *,
        outline_label: str | None,
        require_existing_substantive_output: bool,
    ) -> str | None:
        if not require_existing_substantive_output:
            return None
        if target.suffix.lower() not in {".html", ".htm"}:
            return None
        sibling = self._best_known_existing_html_sibling(target)
        if sibling is None:
            return None
        label = outline_label.strip() if outline_label and outline_label.strip() else target.stem
        return (
            f"Reuse the overall structure and navigation pattern from "
            f"`{display_runtime_path(sibling)}` as the starting pattern for `{label}`; "
            "adapt the title, heading, and body content to the new chapter."
        )

    def _known_html_starter_shape_line(
        self,
        target: Path,
        *,
        require_first_substantive_output: bool,
        retry_number: int,
        outline_label: str | None,
    ) -> str | None:
        if not require_first_substantive_output or retry_number < 1:
            return None
        if target.suffix.lower() not in {".html", ".htm"}:
            return None
        label = outline_label.strip() if outline_label and outline_label.strip() else "this chapter"
        return (
            f"If you get stuck, start with `<title>{label}</title>`, "
            f"`<h1>{label}</h1>`, one introductory paragraph, a couple of `<h2>` "
            "sections with short body text, and a back link to `../index.html`."
        )

    def _best_known_root_html_scaffold(self, target: Path) -> Path | None:
        normalized_target = target.expanduser().resolve(strict=False)
        if normalized_target.suffix.lower() not in {".html", ".htm"}:
            return None
        candidate = normalized_target.parent.parent / "index.html"
        if candidate == normalized_target or not candidate.exists():
            return None
        return candidate

    def _best_known_existing_html_sibling(self, target: Path) -> Path | None:
        normalized_target = target.expanduser().resolve(strict=False)
        if normalized_target.suffix.lower() not in {".html", ".htm"}:
            return None
        try:
            siblings = [
                candidate
                for candidate in normalized_target.parent.iterdir()
                if candidate.is_file()
                and candidate != normalized_target
                and candidate.suffix.lower() == normalized_target.suffix.lower()
                and not _is_summary_artifact_path(candidate)
            ]
        except OSError:
            return None
        if not siblings:
            return None
        siblings.sort(
            key=lambda candidate: (
                candidate.stat().st_mtime if candidate.exists() else 0.0,
                str(candidate),
            ),
            reverse=True,
        )
        return siblings[0]

    def _best_known_reference_path(self, target: Path) -> Path | None:
        normalized_target = target.expanduser().resolve(strict=False)
        target_tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", normalized_target.stem.lower())
            if token
        }
        target_number = _leading_numeric_prefix(normalized_target.stem)
        messages = list(getattr(self.context.session, "messages", []) or [])
        candidates: list[tuple[int, str, Path]] = []

        for message in messages:
            for tool_call in getattr(message, "tool_calls", []) or []:
                if getattr(tool_call, "name", "") != "read":
                    continue
                raw_path = str(tool_call.arguments.get("file_path") or "").strip()
                if not raw_path:
                    continue
                candidate = Path(raw_path).expanduser().resolve(strict=False)
                if candidate == normalized_target or not candidate.suffix:
                    continue
                if candidate.suffix.lower() != normalized_target.suffix.lower():
                    continue
                score = 0
                if candidate.name.lower() == normalized_target.name.lower():
                    score += 8
                if candidate.parent.name.lower() == normalized_target.parent.name.lower():
                    score += 2
                if target_number and _leading_numeric_prefix(candidate.stem) == target_number:
                    score += 3
                candidate_tokens = {
                    token
                    for token in re.split(r"[^a-z0-9]+", candidate.stem.lower())
                    if token
                }
                score += min(3, len(target_tokens & candidate_tokens))
                if score <= 0:
                    continue
                candidates.append((score, str(candidate), candidate))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _reference_content_cues(self, reference: Path) -> str | None:
        try:
            content = reference.read_text()
        except OSError:
            return None

        suffix = reference.suffix.lower()
        cues: list[str] = []
        if suffix in {".html", ".htm"}:
            for raw_line in content.splitlines():
                stripped = " ".join(raw_line.strip().split())
                if not stripped:
                    continue
                lowered = stripped.lower()
                if not any(
                    token in lowered
                    for token in ("<title", "<h1", "<h2", "<p", "<li", "<a ")
                ):
                    continue
                cues.append(_truncate_reference_cue(stripped))
                if len(cues) >= 3:
                    break
        if not cues:
            for raw_line in content.splitlines():
                stripped = " ".join(raw_line.strip().split())
                if not stripped:
                    continue
                if sum(ch.isalpha() for ch in stripped) < 6:
                    continue
                cues.append(_truncate_reference_cue(stripped))
                if len(cues) >= 3:
                    break
        if not cues:
            return None
        return " | ".join(cues)

    @staticmethod
    def _mutation_tool_scaffold(path: Path, *, tool_name: str) -> str:
        normalized_path = json.dumps(display_runtime_path(path))
        if tool_name == "edit":
            signature = (
                f"edit(file_path={normalized_path}, old_string=\"...\", "
                'new_string="...")'
            )
        elif tool_name == "patch":
            signature = f"patch(file_path={normalized_path}, patch=\"...\")"
        else:
            signature = f"write(file_path={normalized_path}, content=\"...\")"
        return f"Emit this tool shape now: `{signature}`."

    @staticmethod
    def _directory_creation_scaffold(path: Path) -> str:
        command = f"mkdir -p {shlex.quote(display_runtime_path(path))}"
        return f"Emit this tool shape now: `bash(command={json.dumps(command)})`."


def _todo_is_mutation_step(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in _MUTATION_TODO_HINTS)


def _todo_is_consistency_review_step(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in _CONSISTENCY_REVIEW_HINTS)


def _is_summary_artifact_path(path: Path) -> bool:
    return path.name.lower() in _SUMMARY_ARTIFACT_NAMES


def _should_encourage_initial_version(
    *,
    target: Path,
    has_confirmed_output_file_progress: bool,
    has_confirmed_substantive_output_file_progress: bool,
) -> bool:
    if not has_confirmed_output_file_progress:
        return True
    if _is_summary_artifact_path(target):
        return False
    return not has_confirmed_substantive_output_file_progress


def _leading_numeric_prefix(stem: str) -> str:
    match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else ""


def _truncate_reference_cue(value: str, *, max_chars: int = 96) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."
