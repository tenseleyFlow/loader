"""Tool-batch execution and recovery bookkeeping for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.base import ToolCall
from .compaction import infer_preferred_next_step, summarize_confirmed_facts
from .context import RuntimeContext
from .dod import (
    DefinitionOfDone,
    DefinitionOfDoneStore,
    all_planned_artifacts_exist,
    begin_new_verification_attempt,
    collect_planned_artifact_targets,
    derive_verification_commands,
    ensure_active_verification_attempt,
    infer_next_declared_html_output_file,
    is_state_mutating_tool_call,
    planned_artifact_target_satisfied,
    record_successful_tool_call,
    synthesize_todo_items,
)
from .events import AgentEvent, TurnSummary
from .evidence_provenance import EvidenceProvenance, EvidenceProvenanceStatus
from .executor import ToolExecutionState, ToolExecutor
from .logging import get_runtime_logger
from .policy_timeline import append_verification_timeline_entry
from .repair_focus import extract_active_repair_context
from .safeguard_services import extract_shell_text_rewrite_target
from .tool_batch_checks import ToolBatchConfidenceGate, ToolBatchVerificationGate
from .tool_batch_recovery import ToolBatchRecoveryController
from .verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
)
from .workflow import (
    advance_todos_from_tool_call,
    effective_pending_todo_items,
    reconcile_aggregate_completion_steps,
    sync_todos_to_definition_of_done,
)

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = (
    Callable[[str, str, str, dict[str, Any] | None], Awaitable[bool]] | None
)
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None

_VERIFY_ITEM = "Collect verification evidence"
_TODO_NUDGE_EXCLUDED_ITEMS = {
    "Complete the requested work",
    _VERIFY_ITEM,
}
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
    "ensure",
)
_BOOKKEEPING_NOTE_TOOL_NAMES = {
    "notepad_write_working",
    "notepad_append",
    "notepad_write_priority",
    "notepad_write_manual",
}


@dataclass
class ToolBatchResult:
    """Outcome of running one assistant-proposed tool batch."""

    actions_taken: list[str] = field(default_factory=list)
    consecutive_errors: int = 0
    halted: bool = False
    final_response: str = ""


class ToolBatchRunner:
    """Owns tool-batch execution, recovery, and post-tool bookkeeping."""

    def __init__(
        self,
        context: RuntimeContext,
        dod_store: DefinitionOfDoneStore,
        *,
        confidence_gate: ToolBatchConfidenceGate | None = None,
        recovery_controller: ToolBatchRecoveryController | None = None,
        verification_gate: ToolBatchVerificationGate | None = None,
    ) -> None:
        self.context = context
        self.dod_store = dod_store
        self.confidence_gate = confidence_gate or ToolBatchConfidenceGate(context)
        self.recovery_controller = recovery_controller or ToolBatchRecoveryController(context)
        self.verification_gate = verification_gate or ToolBatchVerificationGate(context)

    async def execute_batch(
        self,
        *,
        tool_calls: list[ToolCall],
        tool_source: str,
        pending_tool_calls_seen: set[str],
        emit: EventSink,
        summary: TurnSummary,
        dod: DefinitionOfDone,
        executor: ToolExecutor,
        on_confirmation: ConfirmationHandler,
        on_user_question: UserQuestionHandler,
        emit_confirmation,
        consecutive_errors: int,
    ) -> ToolBatchResult:
        """Run one assistant tool batch through the shared executor seam."""

        result = ToolBatchResult(consecutive_errors=consecutive_errors)

        # Pre-populate planned items for the entire batch so the todo
        # widget shows what's coming, not just what's done.
        planned_labels = _batch_planned_labels(tool_calls)
        completed_labels: list[str] = []

        async def _emit_batch_todos() -> None:
            """Emit a todo update combining DoD state with batch progress."""
            items = synthesize_todo_items(dod)
            for label in planned_labels:
                if label in completed_labels:
                    continue
                # Don't duplicate items already in DoD
                if any(item["content"] == label for item in items):
                    continue
                items.append({"content": label, "status": "in_progress", "active_form": label})
            if items:
                await emit(AgentEvent(type="todo_update", todo_items=items))

        await _emit_batch_todos()

        for tool_call in tool_calls:
            cfg = self.context.config.reasoning

            if cfg.confidence_scoring:
                should_skip = await self.confidence_gate.should_skip(
                    tool_call=tool_call,
                    emit=emit,
                )
                if should_skip:
                    continue

            if tool_call.id not in pending_tool_calls_seen:
                await emit(
                    AgentEvent(
                        type="tool_call",
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        tool_args=tool_call.arguments,
                        phase="assistant",
                    )
                )

            result.actions_taken.append(
                f"{tool_call.name}: {str(tool_call.arguments)[:100]}"
            )

            outcome = await executor.execute_tool_call(
                tool_call,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                emit_confirmation=emit_confirmation,
                source=tool_source,
            )
            executed_tool_call = outcome.tool_call
            if (
                outcome.rollback_action is not None
                and self.context.config.reasoning.show_rollback_plan
            ):
                await emit(
                    AgentEvent(
                        type="rollback",
                        content=(
                            f"Rollback tracked: {outcome.rollback_action.description}"
                        ),
                        rollback_action=outcome.rollback_action,
                    )
                )

            if (
                outcome.state == ToolExecutionState.EXECUTED
                and outcome.is_error
                and self.context.config.auto_recover
            ):
                recovery_result = await self.recovery_controller.build_follow_up(
                    tool_call=executed_tool_call,
                    outcome=outcome,
                    emit=emit,
                )
                if recovery_result is not None:
                    summary.tool_result_messages.append(recovery_result)
                    self.context.session.append(recovery_result)
                    continue

            if outcome.state == ToolExecutionState.EXECUTED and not outcome.is_error:
                loop_response = await self._record_successful_execution(
                    tool_call=executed_tool_call,
                    outcome=outcome,
                    dod=dod,
                    emit=emit,
                    summary=summary,
                )
                # Mark this tool's label as completed and emit live progress
                label = _tool_call_label(executed_tool_call)
                if label:
                    completed_labels.append(label)
                await _emit_batch_todos()
                if loop_response is not None:
                    result.halted = True
                    result.final_response = loop_response
                    return result

            if outcome.is_error:
                result.consecutive_errors += 1
            else:
                result.consecutive_errors = 0

            await emit(
                AgentEvent(
                    type="tool_result",
                    content=outcome.event_content,
                    tool_name=executed_tool_call.name,
                    tool_call_id=outcome.tool_call.id,
                    tool_metadata=(
                        outcome.registry_result.metadata
                        if outcome.registry_result is not None
                        else None
                    ),
                    is_error=outcome.is_error,
                    phase="assistant",
                )
            )

            # Always append tool results to the session so the model sees
            # its own output.  The verification gate may inject a correction
            # prompt, but the original result must still be in context —
            # otherwise the model operates blind and loops.
            self.context.session.append(outcome.message)
            summary.tool_result_messages.append(outcome.message)
            if outcome.state == ToolExecutionState.DUPLICATE:
                self._queue_duplicate_observation_nudge(tool_call, dod=dod)
            elif outcome.state == ToolExecutionState.BLOCKED:
                self._queue_blocked_active_repair_nudge(outcome.event_content)
                self._queue_blocked_active_repair_mutation_nudge(outcome.event_content)
                self._queue_blocked_completed_artifact_scope_nudge(
                    outcome.event_content,
                    dod=dod,
                )
                self._queue_blocked_late_reference_drift_nudge(
                    outcome.event_content,
                    dod=dod,
                )
                self._queue_blocked_shell_rewrite_nudge(tool_call)
                self._queue_blocked_html_edit_nudge(tool_call, outcome.event_content)

            should_continue = await self.verification_gate.should_continue(
                tool_call=tool_call,
                outcome=outcome,
                emit=emit,
            )

            rlog = get_runtime_logger()
            rlog.tool_exec(
                name=tool_call.name,
                state=outcome.state.value,
                is_error=outcome.is_error,
                result_preview=outcome.event_content,
                appended_to_session=True,
            )
            if should_continue:
                rlog.verification_gate(tool_call.name, should_continue=True)
                continue

        if result.consecutive_errors >= 3:
            final_response = (
                "I ran into some issues. "
                "Let me know if you'd like me to try a different approach."
            )
            summary.final_response = final_response
            summary.failures.append("three consecutive tool errors")
            await emit(AgentEvent(type="response", content=final_response))
            result.halted = True
            result.final_response = final_response

        return result

    def _queue_duplicate_observation_nudge(
        self,
        tool_call: ToolCall,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Queue a concrete next-step nudge after duplicate observational actions."""

        if tool_call.name not in {"read", "glob", "grep", "bash"}:
            return

        current_task = getattr(self.context.session, "current_task", None)
        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
        )
        next_pending = next(
            (
                item
                for item in effective_pending_todo_items(
                    dod,
                    project_root=self.context.project_root,
                )
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        confirmed_facts = summarize_confirmed_facts(
            self.context.session.messages,
            max_items=2,
        )
        if _should_prioritize_missing_artifact(
            next_pending=next_pending,
            missing_artifact=missing_artifact,
        ):
            prefix = "Reuse the earlier observation instead of repeating it. "
            if confirmed_facts:
                prefix += f"Confirmed facts: {confirmed_facts}. "
            self.context.queue_steering_message(
                prefix
                + "An explicitly planned artifact is still missing."
                + _missing_artifact_resume_suffix(
                    missing_artifact,
                    project_root=self.context.project_root,
                )
                + " Do not switch into review or consistency-check mode until the missing artifact exists."
            )
            return
        if next_pending:
            mutation_suffix = ""
            if _todo_is_mutation_step(next_pending):
                mutation_suffix = _missing_artifact_resume_suffix(
                    missing_artifact,
                    project_root=self.context.project_root,
                )
                if not mutation_suffix:
                    mutation_suffix = (
                        " You already have enough evidence for that step, so stop gathering "
                        "more reference material and perform the change now."
                    )
            if confirmed_facts:
                self.context.queue_steering_message(
                    "Reuse the earlier observation instead of repeating it. "
                    f"Confirmed facts: {confirmed_facts}. "
                    f"Continue with the next pending item: `{next_pending}`. "
                    "Only gather more evidence if a specific fact required for that step is still unknown."
                    + mutation_suffix
                )
            else:
                self.context.queue_steering_message(
                    "Reuse the earlier observation instead of repeating it. "
                    f"Continue with the next pending item: `{next_pending}`. "
                    "Only gather more evidence if a specific fact required for that step is still unknown."
                    + mutation_suffix
                )
            return

        if missing_artifact is not None:
            self.context.queue_steering_message(
                "Reuse the earlier observation instead of repeating it. "
                + _missing_artifact_resume_suffix(
                    missing_artifact,
                    project_root=self.context.project_root,
                ).strip()
            )
            return

        if all_planned_artifacts_exist(dod, project_root=self.context.project_root):
            verification_commands = dod.verification_commands or derive_verification_commands(
                dod,
                project_root=self.context.project_root,
                task_statement=current_task,
                supplement_existing=True,
            )
            verification_suffix = (
                "Move to verification or final confirmation using the files already on disk."
                if verification_commands
                else "Finish the current review using the files already on disk."
            )
            self.context.queue_steering_message(
                "Reuse the earlier observation instead of repeating it. "
                "All explicitly planned artifacts already exist. "
                "Use the current task artifacts as the source of truth and do not reopen "
                "reference materials unless one specific gap is still unknown. "
                + verification_suffix
            )
            return

        preferred_next_step = infer_preferred_next_step(
            self.context.session.messages,
            current_task=current_task,
        )
        if preferred_next_step and confirmed_facts:
            self.context.queue_steering_message(
                "Reuse the earlier observation instead of repeating it. "
                f"Confirmed facts: {confirmed_facts}. "
                f"{preferred_next_step} "
                "Only gather more evidence if a specific filename, href, or title is still unknown."
            )
            return

        if preferred_next_step:
            self.context.queue_steering_message(
                "Reuse the earlier observation instead of repeating it. "
                f"{preferred_next_step} "
                "Only gather more evidence if a specific filename, href, or title is still unknown."
            )
            return

        target_path = str(
            tool_call.arguments.get("file_path")
            or tool_call.arguments.get("path")
            or ""
        ).strip()
        if target_path:
            self.context.queue_steering_message(
                "Reuse the earlier observation instead of repeating it. "
                f"Use the current contents of `{target_path}` and take a different next step. "
                "Only gather more evidence if a specific filename, href, or title is still unknown."
            )
            return

        self.context.queue_steering_message(
            "Reuse the earlier observation instead of repeating it. "
            "Choose a different next step that makes progress."
        )

    def _queue_blocked_shell_rewrite_nudge(self, tool_call: ToolCall) -> None:
        """Steer the model back to file tools after a blocked shell text rewrite."""

        if tool_call.name != "bash":
            return

        target = extract_shell_text_rewrite_target(
            str(tool_call.arguments.get("command", ""))
        )
        if target is None:
            return

        current_task = getattr(self.context.session, "current_task", None)
        confirmed_facts = summarize_confirmed_facts(
            self.context.session.messages,
            max_items=2,
        )
        preferred_next_step = infer_preferred_next_step(
            self.context.session.messages,
            current_task=current_task,
        )

        if preferred_next_step and confirmed_facts:
            self.context.queue_steering_message(
                "Use Loader's file tools for this text edit instead of a shell rewrite. "
                f"Confirmed facts: {confirmed_facts}. "
                f"{preferred_next_step} "
                f"Target `{target}` with edit/patch/write rather than `bash`."
            )
            return

        self.context.queue_steering_message(
            "Use Loader's file tools for this text edit instead of a shell rewrite. "
            f"Apply the change to `{target}` with edit/patch/write."
        )

    def _queue_blocked_active_repair_nudge(self, event_content: str) -> None:
        """Reinforce active repair focus after an out-of-scope blocked observation."""

        if "[Blocked - active repair scope:" not in event_content:
            return

        repair = extract_active_repair_context(self.context.session.messages)
        if repair is None:
            return

        if repair.allowed_paths:
            allowed_preview = ", ".join(f"`{path}`" for path in repair.allowed_paths[:3])
            if len(repair.allowed_paths) > 3:
                allowed_preview += ", ..."
            self.context.queue_steering_message(
                "Verification already identified the active repair target. "
                f"Stay on the concrete repair files {allowed_preview} "
                f"and repair `{repair.artifact_path}` directly. "
                "Do not reopen unrelated reference materials while this repair target is unresolved."
            )
            return

        roots_preview = ", ".join(f"`{root}`" for root in repair.allowed_roots[:2])
        if len(repair.allowed_roots) > 2:
            roots_preview += ", ..."
        self.context.queue_steering_message(
            "Verification already identified the active repair target. "
            f"Stay within the current artifact set under {roots_preview} "
            f"and repair `{repair.artifact_path}` directly. "
            "Do not reopen unrelated reference materials while this repair target is unresolved."
        )

    def _queue_blocked_active_repair_mutation_nudge(self, event_content: str) -> None:
        """Keep repair-phase mutations pinned to the named repair files."""

        if "[Blocked - active repair mutation scope:" not in event_content:
            return

        repair = extract_active_repair_context(self.context.session.messages)
        if repair is None or not repair.allowed_paths:
            return

        allowed_preview = ", ".join(f"`{path}`" for path in repair.allowed_paths[:3])
        if len(repair.allowed_paths) > 3:
            allowed_preview += ", ..."
        self.context.queue_steering_message(
            "Verification already identified the concrete repair files. "
            f"Keep mutations pinned to {allowed_preview} "
            f"and repair `{repair.artifact_path}` before widening the change set."
        )

    def _queue_blocked_late_reference_drift_nudge(
        self,
        event_content: str,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Reinforce missing-artifact progress after late-stage reference drift is blocked."""

        if "[Blocked - late reference drift:" not in event_content:
            return

        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
        )
        if missing_artifact is None:
            return

        planned_roots: list[str] = []
        seen_roots: set[str] = set()
        for target, expect_directory in collect_planned_artifact_targets(
            dod,
            project_root=self.context.project_root,
        ):
            root = str(target if expect_directory else target.parent)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            planned_roots.append(root)

        roots_preview = ", ".join(f"`{root}`" for root in planned_roots[:2])
        if len(planned_roots) > 2:
            roots_preview += ", ..."
        self.context.queue_steering_message(
            "Late-stage reference rereads are no longer helping. "
            "One explicitly planned artifact is still missing."
            + _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
            )
            + f" Stay within the current output roots under {roots_preview}"
            + " and finish that artifact before reopening older reference materials."
        )

    def _queue_blocked_completed_artifact_scope_nudge(
        self,
        event_content: str,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Keep post-build review anchored to the generated artifact set."""

        if "[Blocked - completed artifact set scope:" not in event_content:
            return

        planned_roots: list[str] = []
        seen_roots: set[str] = set()
        for target, expect_directory in collect_planned_artifact_targets(
            dod,
            project_root=self.context.project_root,
        ):
            root = str(target if expect_directory else target.parent)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            planned_roots.append(root)

        next_pending = next(
            (
                item
                for item in effective_pending_todo_items(
                    dod,
                    project_root=self.context.project_root,
                )
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        roots_preview = ", ".join(f"`{root}`" for root in planned_roots[:2])
        if len(planned_roots) > 2:
            roots_preview += ", ..."
        if next_pending and _todo_is_consistency_review_step(next_pending):
            self.context.queue_steering_message(
                "All explicitly planned artifacts already exist. "
                f"Stay within the current output roots under {roots_preview} and continue "
                f"with `{next_pending}` using the generated files as the source of truth. "
                "Do not reopen earlier reference materials."
            )
            return

        self.context.queue_steering_message(
            "All explicitly planned artifacts already exist. "
            f"Stay within the current output roots under {roots_preview} "
            "and move to verification or final confirmation using the generated files. "
            "Do not reopen earlier reference materials."
        )

    def _queue_blocked_html_edit_nudge(self, tool_call: ToolCall, event_content: str) -> None:
        """Keep blocked edit feedback generic; avoid task-class-specific steering."""

        if tool_call.name != "edit":
            return
        if "old_string and new_string are identical - no change would occur" not in event_content:
            return

        repair = extract_active_repair_context(self.context.session.messages)
        if repair is None:
            return

        target = (
            str(tool_call.arguments.get("file_path") or "").strip() or repair.artifact_path
        )
        if not target:
            return

        self.context.queue_steering_message(
            "That edit would make no on-disk change. "
            f"Stay on `{target}` and use the current file contents as the source of truth. "
            "Read the exact current text you need to change, then submit one `edit`, `patch`, "
            "or `write` call that actually changes the file. "
            "If a narrow single-line edit keeps bouncing, replace the surrounding block in one "
            "mutation instead of retrying the same no-op edit. "
            "Do not reopen unrelated reference materials while this concrete repair target is unresolved."
        )

    async def _record_successful_execution(
        self,
        *,
        tool_call: ToolCall,
        outcome,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
    ) -> str | None:
        """Update DoD bookkeeping after a successful tool execution."""

        is_mutating = is_state_mutating_tool_call(tool_call)
        previously_verified = dod.last_verification_result == "passed"
        record_successful_tool_call(dod, tool_call)
        if previously_verified and is_mutating:
            _mark_verification_stale(
                context=self.context,
                summary=summary,
                dod=dod,
                tool_call=tool_call,
            )
        elif is_mutating:
            _mark_verification_planned(
                context=self.context,
                summary=summary,
                dod=dod,
                tool_call=tool_call,
            )
        if tool_call.name == "TodoWrite" and outcome.registry_result is not None:
            new_todos = outcome.registry_result.metadata.get("new_todos", [])
            if isinstance(new_todos, list):
                sync_todos_to_definition_of_done(
                    dod,
                    new_todos,
                    project_root=self.context.project_root,
                )
            self._queue_todowrite_resume_nudge(dod=dod)
        else:
            pending_before = list(dod.pending_items)
            if advance_todos_from_tool_call(dod, tool_call):
                reconcile_aggregate_completion_steps(
                    dod,
                    project_root=self.context.project_root,
                )
                self._queue_next_pending_todo_nudge(
                    tool_call=tool_call,
                    pending_before=pending_before,
                    dod=dod,
                )
            self._queue_bookkeeping_resume_nudge(
                tool_call=tool_call,
                dod=dod,
            )
            self._queue_missing_artifact_progress_nudge(
                tool_call=tool_call,
                dod=dod,
            )
            self._queue_planned_artifact_handoff_nudge(
                tool_call=tool_call,
                dod=dod,
            )
        self.dod_store.save(dod)
        recovery_context = self.context.recovery_context
        if recovery_context is not None:
            recovery_context.note_success(tool_call.name, tool_call.arguments)
            if recovery_context.should_clear_after_success(
                tool_call.name,
                tool_call.arguments,
            ):
                self.context.recovery_context = None
        return None

    def _queue_next_pending_todo_nudge(
        self,
        *,
        tool_call: ToolCall,
        pending_before: list[str],
        dod: DefinitionOfDone,
    ) -> None:
        if is_state_mutating_tool_call(tool_call):
            return
        if tool_call.name not in {"read", "glob", "grep", "bash"}:
            return
        if tool_call.name == "bash":
            command = str(tool_call.arguments.get("command", "")).lower()
            if not any(
                token in command
                for token in (
                    "ls ",
                    " ls",
                    "find ",
                    "grep ",
                    "rg ",
                    "cat ",
                    "sed ",
                    "head ",
                    "tail ",
                )
            ):
                return

        completed_label = next(
            (
                item
                for item in pending_before
                if item not in dod.pending_items
                and item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        next_pending = next(
            (
                item
                for item in effective_pending_todo_items(
                    dod,
                    project_root=self.context.project_root,
                )
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        if not completed_label or not next_pending or next_pending == completed_label:
            return

        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
        )
        if _should_prioritize_missing_artifact(
            next_pending=next_pending,
            missing_artifact=missing_artifact,
        ):
            self.context.queue_steering_message(
                f"Confirmed progress: `{completed_label}` is now satisfied by the successful "
                f"`{tool_call.name}` result. One explicitly planned artifact is still missing."
                + _missing_artifact_resume_suffix(
                    missing_artifact,
                    project_root=self.context.project_root,
                )
                + " Do not switch into review or consistency-check mode until the missing artifact exists."
            )
            return

        mutation_suffix = ""
        if _todo_is_mutation_step(next_pending):
            mutation_suffix = _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
            )
            if not mutation_suffix:
                mutation_suffix = (
                    " You already have enough evidence for that step, so stop gathering "
                    "more reference material and perform the change now."
                )

        self.context.queue_steering_message(
            f"Confirmed progress: `{completed_label}` is now satisfied by the successful "
            f"`{tool_call.name}` result. Continue with the next pending item: "
            f"`{next_pending}` instead of rereading the same evidence.{mutation_suffix}"
        )

    def _queue_planned_artifact_handoff_nudge(
        self,
        *,
        tool_call: ToolCall,
        dod: DefinitionOfDone,
    ) -> None:
        if not is_state_mutating_tool_call(tool_call):
            return
        if not all_planned_artifacts_exist(dod, project_root=self.context.project_root):
            return

        next_pending = next(
            (
                item
                for item in effective_pending_todo_items(
                    dod,
                    project_root=self.context.project_root,
                )
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        verification_commands = dod.verification_commands or derive_verification_commands(
            dod,
            project_root=self.context.project_root,
            task_statement=getattr(self.context.session, "current_task", "") or "",
            supplement_existing=True,
        )

        if next_pending and _todo_is_consistency_review_step(next_pending):
            verification_suffix = (
                " Move to verification once no specific mismatch remains."
                if verification_commands
                else " Avoid another full reread unless one specific inconsistency is still unknown."
            )
            self.context.queue_steering_message(
                "All explicitly planned artifacts now exist. "
                f"Continue with the next pending item: `{next_pending}`. "
                "Use the files already on disk as the source of truth instead of restarting "
                "discovery or inventing alternate filenames."
                + verification_suffix
            )
            return

        if verification_commands:
            self.context.queue_steering_message(
                "All explicitly planned artifacts now exist. "
                "Do not expand the artifact set or restart discovery unless a specific gap is "
                "still known. Move to verification or final confirmation using the files that "
                "already exist."
            )

    def _queue_missing_artifact_progress_nudge(
        self,
        *,
        tool_call: ToolCall,
        dod: DefinitionOfDone,
    ) -> None:
        if not is_state_mutating_tool_call(tool_call):
            return
        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
        )
        if missing_artifact is None:
            return

        current_label = _current_mutation_label(tool_call)
        todo_refresh = _todo_refresh_guidance(
            dod,
            project_root=self.context.project_root,
        )
        self.context.queue_steering_message(
            f"Confirmed progress: {current_label} is now recorded."
            " One explicitly planned artifact is still missing."
            + _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
            )
            + todo_refresh
            + " Do not move to verification, final confirmation, or TodoWrite-only "
            "bookkeeping until that artifact exists."
            + " Do not spend another turn on working notes or rediscovery alone."
        )

    def _queue_todowrite_resume_nudge(
        self,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
        )
        next_pending = next(
            (
                item
                for item in effective_pending_todo_items(
                    dod,
                    project_root=self.context.project_root,
                )
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        if missing_artifact is None:
            if next_pending and _todo_is_mutation_step(next_pending):
                self.context.queue_steering_message(
                    "Todo tracking is updated. Continue with the next pending item: "
                    f"`{next_pending}`. Use the current output files as the source of "
                    "truth, and do not reopen reference materials unless one specific "
                    "fact required for that step is still unknown. Perform the mutation "
                    "now instead of spending another turn on planning, rereads, or "
                    "verification."
                )
                return

            if (
                next_pending
                and _todo_is_consistency_review_step(next_pending)
                and not all_planned_artifacts_exist(
                    dod,
                    project_root=self.context.project_root,
                )
            ):
                self.context.queue_steering_message(
                    "Todo tracking is updated. Continue with the next pending item: "
                    f"`{next_pending}`. Use the current output files as the source of "
                    "truth, and do not reopen reference materials unless one specific "
                    "mismatch is still unknown."
                )
                return

            if not all_planned_artifacts_exist(dod, project_root=self.context.project_root):
                return

            verification_commands = dod.verification_commands or derive_verification_commands(
                dod,
                project_root=self.context.project_root,
                task_statement=getattr(self.context.session, "current_task", "") or "",
                supplement_existing=True,
            )
            if next_pending and _todo_is_consistency_review_step(next_pending):
                verification_suffix = (
                    " Move to verification once no specific mismatch remains."
                    if verification_commands
                    else " Finish the targeted consistency pass without reopening reference materials."
                )
                self.context.queue_steering_message(
                    "Todo tracking is updated. All explicitly planned artifacts now exist. "
                    f"Continue with the next pending item: `{next_pending}`. "
                    "Use the current output files as the source of truth, and do not restart "
                    "early discovery or reopen reference materials."
                    + verification_suffix
                )
                return

            verification_suffix = (
                " Move to verification or final confirmation using the files already on disk."
                if verification_commands
                else " Finish the task using the files already on disk."
            )
            self.context.queue_steering_message(
                "Todo tracking is updated. All explicitly planned artifacts now exist. "
                "Do not restart discovery, reopen reference materials, or spend another turn "
                "on TodoWrite alone."
                + verification_suffix
            )
            return

        todo_refresh = _todo_refresh_guidance(
            dod,
            project_root=self.context.project_root,
        )
        next_pending_suffix = (
            f" Continue with the next pending item: `{next_pending}`."
            if next_pending
            else ""
        )
        self.context.queue_steering_message(
            "Todo tracking is updated. An explicitly planned artifact is still missing."
            + next_pending_suffix
            + _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
            )
            + todo_refresh
            + " Do not spend the next turn on TodoWrite alone, bookkeeping notes, "
            "verification, or final confirmation until that artifact exists."
        )

    def _queue_bookkeeping_resume_nudge(
        self,
        *,
        tool_call: ToolCall,
        dod: DefinitionOfDone,
    ) -> None:
        if tool_call.name not in _BOOKKEEPING_NOTE_TOOL_NAMES:
            return

        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
        )
        if missing_artifact is None:
            return

        next_pending = next(
            (
                item
                for item in effective_pending_todo_items(
                    dod,
                    project_root=self.context.project_root,
                )
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        todo_refresh = _todo_refresh_guidance(
            dod,
            project_root=self.context.project_root,
        )
        if (
            next_pending
            and not _todo_is_mutation_step(next_pending)
            and not _todo_is_consistency_review_step(next_pending)
        ):
            self.context.queue_steering_message(
                "Bookkeeping note is recorded. Continue with the next pending item: "
                f"`{next_pending}`. Make your next response one concrete evidence-gathering "
                "tool call that advances that step, not another bookkeeping-only turn."
                + todo_refresh
                + " Do not jump ahead to later artifact creation, verification, or final "
                "confirmation until that step is satisfied."
            )
            return

        self.context.queue_steering_message(
            "Bookkeeping note is recorded. An explicitly planned artifact is still missing."
            + _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
            )
            + todo_refresh
            + " Do not spend the next turn on additional notes, rediscovery, "
            "verification, or final confirmation until that artifact exists."
        )


def _todo_is_consistency_review_step(item: str) -> bool:
    text = item.lower()
    return any(hint in text for hint in _CONSISTENCY_REVIEW_HINTS)


def _should_prioritize_missing_artifact(
    *,
    next_pending: str | None,
    missing_artifact: tuple[Path, bool] | None,
) -> bool:
    if missing_artifact is None:
        return False
    if not next_pending:
        return True
    if _todo_is_consistency_review_step(next_pending):
        return True
    return not _todo_is_mutation_step(next_pending)


def _next_missing_planned_artifact(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> tuple[Path, bool] | None:
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if not planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=expect_directory,
            project_root=project_root,
        ):
            return target, expect_directory
    return None


def _missing_artifact_resume_suffix(
    missing_artifact: tuple[Path, bool] | None,
    *,
    project_root: Path,
) -> str:
    if missing_artifact is None:
        return ""

    target, expect_directory = missing_artifact
    label = target.name or str(target)
    if expect_directory and not label.endswith("/"):
        label += "/"
    if expect_directory:
        next_output_file = infer_next_declared_html_output_file(
            target=target,
            project_root=project_root,
        )
        if next_output_file is not None:
            guidance = (
                f" Resume by creating `{next_output_file.name}` now. It is the next missing "
                f"declared output under `{label}`. Prefer one `write` call for "
                f"`{next_output_file}` instead of more rereads."
            )
            if not next_output_file.parent.exists():
                guidance += (
                    " The `write` tool can create that file's parent directories automatically,"
                    " so do the write in one step instead of stopping for a separate mkdir."
                )
            guidance += (
                " Make your next response the concrete mutation tool call itself, not another"
                " bookkeeping-only turn."
            )
            return guidance
        if target.is_dir():
            return (
                f" Resume by creating the next output file under `{label}` now. Prefer one "
                f"concrete `write` call for a file inside `{target}` instead of more rereads."
                " Make your next response the concrete mutation tool call itself, not another"
                " bookkeeping-only turn."
            )
        return (
            f" Resume by creating `{label}` now. Prefer one concrete directory-creation "
            f"step for `{target}` instead of more rereads."
        )
    guidance = (
        f" Resume by creating `{label}` now. Prefer one `write` call for `{target}` "
        "instead of more rereads."
    )
    if not target.parent.exists():
        guidance += (
            " The `write` tool can create that file's parent directories automatically,"
            " so do the write in one step instead of stopping for a separate mkdir."
        )
    guidance += (
        " Make your next response the concrete mutation tool call itself, not another"
        " bookkeeping-only turn."
    )
    return guidance


def _todo_refresh_guidance(
    dod: DefinitionOfDone,
    *,
    project_root: Path | None = None,
) -> str:
    non_special_pending = [
        item
        for item in effective_pending_todo_items(dod, project_root=project_root)
        if item not in _TODO_NUDGE_EXCLUDED_ITEMS
    ]
    non_special_completed = [
        item for item in dod.completed_items if item not in _TODO_NUDGE_EXCLUDED_ITEMS
    ]
    if len(dod.touched_files) < 2 and (len(non_special_pending) + len(non_special_completed)) < 3:
        return ""
    return (
        " If the tracked steps no longer match the confirmed progress, refresh `TodoWrite` "
        "in the same response as the next concrete step instead of spending a full turn on "
        "bookkeeping alone."
    )


def _mark_verification_stale(
    *,
    context: RuntimeContext,
    summary: TurnSummary,
    dod: DefinitionOfDone,
    tool_call: ToolCall,
) -> None:
    detail = _stale_verification_detail(tool_call)
    stale_attempt = ensure_active_verification_attempt(dod)
    next_attempt = begin_new_verification_attempt(
        dod,
        supersedes_attempt_id=stale_attempt.attempt_id,
    )
    append_verification_timeline_entry(
        context,
        summary,
        reason_code="verification_stale",
        reason_summary="previous verification became stale after new mutating work",
        evidence_summary=[f"fresh verification required after {detail}"],
        evidence_provenance=_stale_verification_provenance(dod, detail=detail),
        verification_observations=_stale_verification_observations(
            dod,
            detail=detail,
            stale_attempt_id=stale_attempt.attempt_id,
            stale_attempt_number=stale_attempt.attempt_number,
            superseded_by_attempt_id=next_attempt.attempt_id,
        ),
    )
    dod.last_verification_result = VerificationObservationStatus.STALE.value
    dod.evidence = []
    while _VERIFY_ITEM in dod.completed_items:
        dod.completed_items.remove(_VERIFY_ITEM)
    if _VERIFY_ITEM not in dod.pending_items:
        dod.pending_items.append(_VERIFY_ITEM)


def _todo_is_mutation_step(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in _MUTATION_TODO_HINTS)


def _mark_verification_planned(
    *,
    context: RuntimeContext,
    summary: TurnSummary,
    dod: DefinitionOfDone,
    tool_call: ToolCall,
) -> None:
    if dod.last_verification_result in {
        VerificationObservationStatus.PLANNED.value,
        VerificationObservationStatus.PENDING.value,
        VerificationObservationStatus.STALE.value,
    }:
        return
    if not dod.verification_commands:
        dod.verification_commands = derive_verification_commands(
            dod,
            project_root=context.project_root,
            task_statement=dod.task_statement,
        )
    commands = [command for command in dod.verification_commands if command]
    if not commands:
        return

    attempt = begin_new_verification_attempt(dod)
    detail = _stale_verification_detail(tool_call)
    append_verification_timeline_entry(
        context,
        summary,
        reason_code="verification_planned",
        reason_summary="verification is planned after new mutating work",
        evidence_summary=[f"verification planned for `{command}`" for command in commands[:2]],
        evidence_provenance=[
            EvidenceProvenance(
                category="verification",
                source="dod.verification_commands",
                summary=f"verification planned for `{command}`",
                status=EvidenceProvenanceStatus.MISSING.value,
                subject=command,
                detail=detail,
            )
            for command in commands
        ],
        verification_observations=[
            VerificationObservation(
                status=VerificationObservationStatus.PLANNED.value,
                summary=f"verification planned for `{command}`",
                command=command,
                kind="runtime",
                detail=detail,
                attempt_id=attempt.attempt_id,
                attempt_number=attempt.attempt_number,
            )
            for command in commands
        ],
    )
    dod.last_verification_result = VerificationObservationStatus.PLANNED.value
    while _VERIFY_ITEM in dod.completed_items:
        dod.completed_items.remove(_VERIFY_ITEM)
    if _VERIFY_ITEM not in dod.pending_items:
        dod.pending_items.append(_VERIFY_ITEM)


def _stale_verification_observations(
    dod: DefinitionOfDone,
    *,
    detail: str,
    stale_attempt_id: str,
    stale_attempt_number: int,
    superseded_by_attempt_id: str,
) -> list[VerificationObservation]:
    return [
        VerificationObservation(
            status=VerificationObservationStatus.STALE.value,
            summary=f"verification became stale for `{command}` after new mutating work",
            command=command,
            kind="runtime",
            detail=detail,
            attempt_id=stale_attempt_id,
            attempt_number=stale_attempt_number,
            supersedes_attempt_id=superseded_by_attempt_id,
        )
        for command in _stale_verification_commands(dod)
    ]


def _stale_verification_provenance(
    dod: DefinitionOfDone,
    *,
    detail: str,
) -> list[EvidenceProvenance]:
    return [
        EvidenceProvenance(
            category="verification",
            source="tool_execution",
            summary=f"fresh verification required for `{command}` after new mutating work",
            status=EvidenceProvenanceStatus.MISSING.value,
            subject=command,
            detail=detail,
        )
        for command in _stale_verification_commands(dod)
    ]


def _stale_verification_commands(dod: DefinitionOfDone) -> list[str]:
    commands = [command for command in dod.verification_commands if command]
    if commands:
        return commands
    observed = [evidence.command for evidence in dod.evidence if evidence.command]
    if observed:
        return observed
    return ["verification"]


def _stale_verification_detail(tool_call: ToolCall) -> str:
    if tool_call.name in {"write", "edit", "patch"}:
        file_path = str(tool_call.arguments.get("file_path", "")).strip()
        if file_path:
            return f"{tool_call.name} changed {file_path}"
    if tool_call.name == "bash":
        command = str(tool_call.arguments.get("command", "")).strip()
        if command:
            return f"bash ran `{command}`"
    return f"{tool_call.name} changed the workspace"


def _current_mutation_label(tool_call: ToolCall) -> str:
    if tool_call.name in {"write", "edit", "patch"}:
        file_path = str(tool_call.arguments.get("file_path", "")).strip()
        if file_path:
            return f"`{Path(file_path).name or file_path}`"
    if tool_call.name == "bash":
        command = str(tool_call.arguments.get("command", "")).strip()
        if command:
            return f"`{command}`"
    return f"the successful `{tool_call.name}` result"


def _tool_call_label(tool_call: ToolCall) -> str:
    """Human-readable label for one tool call."""
    name = tool_call.name
    if name in ("write", "edit", "patch"):
        path = str(tool_call.arguments.get("file_path", "")).strip()
        if path:
            short = Path(path).name
            verb = "Write" if name == "write" else "Edit"
            return f"{verb} {short}"
    if name == "bash":
        cmd = str(tool_call.arguments.get("command", "")).strip()
        if cmd:
            return f"Run {cmd[:40]}"
    if name == "read":
        path = str(tool_call.arguments.get("file_path", "")).strip()
        if path:
            return f"Read {Path(path).name}"
    if name == "glob":
        pattern = str(tool_call.arguments.get("pattern", "")).strip()
        if pattern:
            return f"Search {pattern[:30]}"
    return ""


def _batch_planned_labels(tool_calls: list[ToolCall]) -> list[str]:
    """Build labels for all tool calls in a batch (for upfront planning display)."""
    labels = []
    for tc in tool_calls:
        label = _tool_call_label(tc)
        if label and label not in labels:
            labels.append(label)
    return labels
