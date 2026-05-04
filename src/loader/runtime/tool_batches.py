"""Tool-batch execution and recovery bookkeeping for the typed runtime."""

from __future__ import annotations

import shlex
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
    all_planned_artifact_outputs_exist,
    all_planned_artifacts_exist,
    begin_new_verification_attempt,
    collect_planned_artifact_targets,
    derive_verification_commands,
    ensure_active_verification_attempt,
    infer_next_output_file,
    is_state_mutating_tool_call,
    planned_artifact_target_satisfied,
    record_successful_tool_call,
    synthesize_todo_items,
)
from .events import AgentEvent, TurnSummary
from .evidence_provenance import EvidenceProvenance, EvidenceProvenanceStatus
from .executor import ToolExecutionState, ToolExecutor
from .logging import get_runtime_logger
from .path_display import display_runtime_path
from .policy_timeline import append_verification_timeline_entry
from .recovery import RecoveryContext, detect_missing_mutation_payload
from .repair_focus import extract_active_repair_context, path_within_allowed_roots
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
    infer_pending_todo_output_target,
    preferred_pending_todo_item,
    reconcile_aggregate_completion_steps,
    sync_todos_to_definition_of_done,
    todo_describes_aggregate_mutation,
    todo_describes_broad_setup_step,
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
    "develop",
    "developing",
    "populate",
    "populating",
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
    "ensure",
)
_BOOKKEEPING_NOTE_TOOL_NAMES = {
    "notepad_write_working",
    "notepad_append",
    "notepad_write_priority",
    "notepad_write_manual",
}
_SUMMARY_ARTIFACT_NAMES = {
    "index.html",
    "index.htm",
    "readme",
    "readme.md",
    "readme.rst",
    "readme.txt",
}
_OBSERVATION_TOOLS = frozenset({"read", "glob", "grep", "bash"})
_READ_ONLY_BASH_PREFIXES = frozenset(
    {"ls", "pwd", "find", "stat", "cat", "head", "tail", "rg", "grep"}
)
_MUTATING_BASH_FRAGMENTS = (
    " >",
    ">>",
    "| tee",
    "touch ",
    "mkdir ",
    "rm ",
    "mv ",
    "cp ",
    "sed -i",
    "perl -pi",
    "git add",
    "git commit",
    "git apply",
)


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
                self._queue_blocked_invalid_mutation_nudge(
                    tool_call,
                    outcome.event_content,
                    dod=dod,
                )
                self._queue_blocked_html_declared_file_creation_nudge(
                    tool_call,
                    outcome.event_content,
                    dod=dod,
                )
                self._queue_blocked_html_declared_target_nudge(
                    tool_call,
                    outcome.event_content,
                )
                self._queue_blocked_html_missing_target_nudge(
                    tool_call,
                    outcome.event_content,
                    dod=dod,
                )
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
                self._queue_blocked_html_edit_nudge(
                    tool_call,
                    outcome.event_content,
                    dod=dod,
                )
            else:
                self._queue_post_mutation_self_audit_nudge(tool_call, dod=dod)
                self._queue_completed_artifact_observation_handoff_nudge(
                    tool_call,
                    dod=dod,
                )

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
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
            missing_artifact=missing_artifact,
        )
        confirmed_facts = summarize_confirmed_facts(
            self.context.session.messages,
            max_items=2,
        )
        if _should_prioritize_missing_artifact(
            dod=dod,
            next_pending=next_pending,
            missing_artifact=missing_artifact,
            project_root=self.context.project_root,
        ):
            prefix = "Reuse the earlier observation instead of repeating it. "
            if confirmed_facts:
                prefix += f"Confirmed facts: {confirmed_facts}. "
            self.context.queue_steering_message(
                prefix
                + "A declared output artifact is still missing."
                + _missing_artifact_resume_suffix(
                    missing_artifact,
                    project_root=self.context.project_root,
                    messages=list(getattr(self.context.session, "messages", []) or []),
                )
                + " Do not switch into review or consistency-check mode until the missing artifact exists."
            )
            return
        if next_pending:
            mutation_suffix = ""
            if _todo_is_mutation_step(next_pending):
                mutation_suffix = _pending_item_resume_suffix(
                    dod,
                    next_pending=next_pending,
                    missing_artifact=missing_artifact,
                    project_root=self.context.project_root,
                    messages=list(getattr(self.context.session, "messages", []) or []),
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
                    messages=list(getattr(self.context.session, "messages", []) or []),
                ).strip()
            )
            return

        if all_planned_artifact_outputs_exist(dod, project_root=self.context.project_root):
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
                "All explicitly planned artifacts already exist on disk. "
                "Use the current task artifacts as the source of truth and do not reopen "
                "reference materials unless one specific gap is still unknown. "
                "If anything is still wrong, repair the current files instead of expanding the artifact set. "
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

    def _queue_post_mutation_self_audit_nudge(
        self,
        tool_call: ToolCall,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Steer out of rereading the file that was just written when the next output is known."""

        if tool_call.name != "read":
            return

        file_path = str(tool_call.arguments.get("file_path", "")).strip()
        if not file_path:
            return

        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        if missing_artifact is None:
            return

        read_target = Path(file_path).expanduser().resolve(strict=False)
        last_touched = _last_touched_file_path(dod)
        if last_touched is None or read_target != last_touched:
            return

        self.context.queue_steering_message(
            f"You already have the current contents of `{read_target.name}` from the successful write. "
            "A declared output artifact is still missing."
            + _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
                messages=list(getattr(self.context.session, "messages", []) or []),
            )
            + " Do not spend another turn rereading the file you just wrote or on TodoWrite alone."
        )

    def _queue_completed_artifact_observation_handoff_nudge(
        self,
        tool_call: ToolCall,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Turn successful post-build audit reads into verify/finalize handoffs."""

        if tool_call.name not in _OBSERVATION_TOOLS:
            return
        if dod.status in {"fixing", "done"}:
            return
        if extract_active_repair_context(self.context.session.messages) is not None:
            return
        if not all_planned_artifact_outputs_exist(dod, project_root=self.context.project_root):
            return

        observed_paths = _extract_observation_paths(tool_call)
        if not observed_paths:
            return

        planned_roots = _planned_output_roots(
            dod,
            project_root=self.context.project_root,
        )
        if not planned_roots:
            return
        if not all(path_within_allowed_roots(path, planned_roots) for path in observed_paths):
            return

        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
        )
        verification_commands = dod.verification_commands or derive_verification_commands(
            dod,
            project_root=self.context.project_root,
            task_statement=getattr(self.context.session, "current_task", "") or "",
            supplement_existing=True,
        )
        roots_preview = ", ".join(f"`{root}`" for root in planned_roots[:2])
        if len(planned_roots) > 2:
            roots_preview += ", ..."

        if next_pending and _todo_is_consistency_review_step(next_pending):
            verification_suffix = (
                " If no specific mismatch remains, move to verification now."
                if verification_commands
                else " If no specific mismatch remains, finish the task now."
            )
            self.context.queue_ephemeral_steering_message(
                "All explicitly planned artifacts already exist. "
                f"Continue with `{next_pending}` using the generated files under {roots_preview} "
                "as the source of truth, but do not keep broad-rereading the output set. "
                "If you already know a concrete mismatch, fix it directly."
                + verification_suffix
            )
            return

        if verification_commands:
            self.context.set_workflow_mode("verify")
            self.context.queue_steering_message(
                "All explicitly planned artifacts already exist. "
                f"Use the generated files under {roots_preview} as the source of truth and stop broad rereads. "
                "If you already know a concrete mismatch, fix it directly. "
                "Verification should run next. Do not reopen reference materials or keep auditing the same files."
            )
            return

        verification_suffix = (
            "Move to verification or final confirmation using the files already on disk."
            if verification_commands
            else "Finish the task using the files already on disk."
        )
        self.context.queue_ephemeral_steering_message(
            "All explicitly planned artifacts already exist. "
            f"Use the generated files under {roots_preview} as the source of truth and stop broad rereads. "
            "If you already know a concrete mismatch, fix it directly. "
            + verification_suffix
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
            messages=list(getattr(self.context.session, "messages", []) or []),
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
                messages=list(getattr(self.context.session, "messages", []) or []),
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

        blocked_completed_scope = (
            "[Blocked - completed artifact set scope:" in event_content
        )
        blocked_post_build_audit = "[Blocked - post-build audit loop:" in event_content
        if not blocked_completed_scope and not blocked_post_build_audit:
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

        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
        )
        verification_commands = dod.verification_commands or derive_verification_commands(
            dod,
            project_root=self.context.project_root,
            task_statement=getattr(self.context.session, "current_task", "") or "",
            supplement_existing=True,
        )
        if verification_commands:
            self.context.set_workflow_mode("verify")
        roots_preview = ", ".join(f"`{root}`" for root in planned_roots[:2])
        if len(planned_roots) > 2:
            roots_preview += ", ..."
        if next_pending and _todo_is_consistency_review_step(next_pending):
            self.context.queue_steering_message(
                "All explicitly planned artifacts already exist. "
                f"Stay within the current output roots under {roots_preview} and continue "
                f"with `{next_pending}` using the generated files as the source of truth. "
                "Do not reopen earlier reference materials."
                + (
                    " Verification should run next using those generated files."
                    if verification_commands
                    else ""
                )
            )
            return

        self.context.queue_steering_message(
            "All explicitly planned artifacts already exist. "
            f"Stay within the current output roots under {roots_preview} "
            "and move to verification or final confirmation using the generated files. "
            "Do not reopen earlier reference materials."
        )

    def _queue_blocked_html_edit_nudge(
        self,
        tool_call: ToolCall,
        event_content: str,
        *,
        dod: DefinitionOfDone,
    ) -> None:
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

        verification_commands = dod.verification_commands or derive_verification_commands(
            dod,
            project_root=self.context.project_root,
            task_statement=getattr(self.context.session, "current_task", "") or "",
            supplement_existing=True,
        )
        if all_planned_artifacts_exist(dod, project_root=self.context.project_root):
            verification_suffix = (
                " Move to verification or final confirmation using the files already on disk."
                if verification_commands
                else " If no concrete mismatch remains, stop editing and finish from the files already on disk."
            )
            self.context.queue_steering_message(
                "That edit would make no on-disk change. "
                f"`{target}` already matches the change you attempted. "
                "All explicitly planned artifacts already exist."
                + verification_suffix
            )
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

    def _queue_blocked_html_declared_target_nudge(
        self,
        tool_call: ToolCall,
        event_content: str,
    ) -> None:
        """Steer blocked HTML graph edits back to the root-declared local targets."""

        if tool_call.name not in {"write", "edit", "patch"}:
            return
        if "HTML page introduces new local targets outside the current declared artifact set" not in event_content:
            return

        target = str(
            tool_call.arguments.get("file_path")
            or tool_call.arguments.get("path")
            or ""
        ).strip()
        if not target:
            return

        closest_targets = _extract_blocked_html_target_list(
            event_content,
            "Closest declared local targets include:",
        )
        declared_targets = _extract_blocked_html_target_list(
            event_content,
            "Already-declared local targets include:",
        )

        guidance = (
            "That HTML mutation introduced sibling targets outside the current declared local-link set. "
            f"Stay on `{target}`."
        )
        if closest_targets:
            guidance += (
                " Remove the invented hrefs or replace them with the closest declared target(s): "
                + ", ".join(f"`{candidate}`" for candidate in closest_targets[:3])
                + "."
            )
        elif declared_targets:
            guidance += (
                " Remove the invented hrefs or keep local links within the declared target set, for example: "
                + ", ".join(f"`{candidate}`" for candidate in declared_targets[:3])
                + "."
            )
        guidance += (
            " Resend one concrete mutation for that same file now instead of rereading the reference guide."
        )
        self.context.queue_steering_message(guidance)

    def _queue_blocked_html_declared_file_creation_nudge(
        self,
        tool_call: ToolCall,
        event_content: str,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Steer blocked undeclared HTML file creation back through the root guide."""

        if tool_call.name not in {"write", "edit", "patch"}:
            return
        if "HTML file creation falls outside the current declared artifact set" not in event_content:
            return

        target = str(
            tool_call.arguments.get("file_path")
            or tool_call.arguments.get("path")
            or ""
        ).strip()
        if not target:
            return

        target_path = Path(target).expanduser().resolve(strict=False)
        root_index = (target_path.parent.parent / "index.html").resolve(strict=False)
        relative_target = None
        try:
            relative_target = target_path.relative_to(root_index.parent).as_posix()
        except ValueError:
            relative_target = target_path.name

        if all_planned_artifact_outputs_exist(dod, project_root=self.context.project_root):
            verification_commands = dod.verification_commands or derive_verification_commands(
                dod,
                project_root=self.context.project_root,
                task_statement=getattr(self.context.session, "current_task", "") or "",
                supplement_existing=True,
            )
            verification_suffix = (
                " Move to verification or final confirmation using the files already on disk."
                if verification_commands
                else " Finish the task using the files already on disk."
            )
            self.context.queue_steering_message(
                "All explicitly planned artifacts already exist on disk. "
                f"Do not expand the output set with `{relative_target}`. "
                "Use the current generated files as the source of truth and repair or verify them instead."
                + verification_suffix
            )
            return

        guidance = (
            "That new HTML file is outside the current root-declared artifact set. "
            f"Before creating `{relative_target}`, update `{root_index}` so the guide root "
            "explicitly links to that page, then retry the file creation. "
            "Stay on the active guide files; do not reopen the earlier reference guide first."
        )
        self.context.queue_steering_message(guidance)

    def _queue_blocked_html_missing_target_nudge(
        self,
        tool_call: ToolCall,
        event_content: str,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Turn post-build missing-link expansions into verify/repair handoffs."""

        if tool_call.name not in {"write", "edit", "patch"}:
            return
        if "Edited HTML links point to files that do not exist" not in event_content:
            return
        if not all_planned_artifact_outputs_exist(dod, project_root=self.context.project_root):
            return

        verification_commands = dod.verification_commands or derive_verification_commands(
            dod,
            project_root=self.context.project_root,
            task_statement=getattr(self.context.session, "current_task", "") or "",
            supplement_existing=True,
        )
        verification_suffix = (
            " Move to verification or final confirmation using the files already on disk."
            if verification_commands
            else " Finish the task using the files already on disk."
        )
        self.context.queue_steering_message(
            "All explicitly planned artifacts already exist on disk. "
            "Do not introduce new local-link targets beyond the current output set. "
            "Repair the existing generated files instead of expanding the guide."
            + verification_suffix
        )

    def _queue_blocked_invalid_mutation_nudge(
        self,
        tool_call: ToolCall,
        event_content: str,
        *,
        dod: DefinitionOfDone,
    ) -> None:
        """Recover blocked mutations that omitted a real target path or text payload."""

        fix = detect_missing_mutation_payload(
            tool_call.name,
            tool_call.arguments,
            event_content,
        )
        if fix is None:
            return

        self._record_blocked_invalid_mutation_attempt(tool_call, event_content)

        messages = list(getattr(self.context.session, "messages", []) or [])
        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
            messages=messages,
        )
        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
            missing_artifact=missing_artifact,
        )
        missing_artifact = _prefer_missing_artifact_for_pending_item(
            dod,
            missing_artifact=missing_artifact,
            next_pending=next_pending,
            project_root=self.context.project_root,
        )
        resume_target = _preferred_resume_target_path(
            dod,
            next_pending=next_pending,
            missing_artifact=missing_artifact,
            project_root=self.context.project_root,
            messages=messages,
        )
        resume_suffix = _pending_item_resume_suffix(
            dod,
            next_pending=next_pending,
            missing_artifact=missing_artifact,
            project_root=self.context.project_root,
            messages=messages,
        )
        target_label = f"`{resume_target.name or str(resume_target)}`" if resume_target else ""

        if fix.get("kind") == "missing_target":
            prefix = f"That `{tool_call.name}` call did not provide a valid `file_path`."
            if target_label:
                prefix += f" Stay on {target_label}."
            self.context.queue_steering_message(
                prefix
                + resume_suffix
                + " Resend one concrete "
                + _invalid_mutation_call_shape(tool_call.name)
                + " now instead of another working note, reread, or empty response."
            )
            return

        invalid_fields = ", ".join(f"`{field}`" for field in fix["invalid_fields"])
        prefix = f"That `{tool_call.name}` call omitted the real text payload."
        if invalid_fields:
            prefix += f" {invalid_fields} are summary fields, not valid mutation inputs."
        if target_label:
            prefix += f" Stay on {target_label}."
        self.context.queue_steering_message(
            prefix
            + resume_suffix
            + " Resend one concrete "
            + _invalid_mutation_call_shape(tool_call.name)
            + " now instead of rereading more files."
        )

    def _record_blocked_invalid_mutation_attempt(
        self,
        tool_call: ToolCall,
        error: str,
    ) -> None:
        """Seed recovery state from blocked malformed mutations for later retry guidance."""

        recovery_context = self.context.recovery_context
        if recovery_context is None or not recovery_context.is_related_failure(
            tool_call.name,
            tool_call.arguments,
            error,
        ):
            recovery_context = RecoveryContext(
                original_tool=tool_call.name,
                original_args=tool_call.arguments,
                max_retries=self.context.config.max_recovery_attempts,
            )
            self.context.recovery_context = recovery_context

        if not recovery_context.is_similar_attempt(
            tool_call.name,
            tool_call.arguments,
        ):
            recovery_context.add_attempt(
                tool_call.name,
                tool_call.arguments,
                error,
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
        if previously_verified and is_mutating:
            _mark_verification_stale(
                context=self.context,
                summary=summary,
                dod=dod,
                tool_call=tool_call,
            )
        elif is_mutating and _should_plan_verification_for_tool_call(
            dod,
            tool_call=tool_call,
            project_root=self.context.project_root,
        ):
            _mark_verification_planned(
                context=self.context,
                summary=summary,
                dod=dod,
                tool_call=tool_call,
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
        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
            missing_artifact=missing_artifact,
        )
        missing_artifact = _prefer_missing_artifact_for_pending_item(
            dod,
            missing_artifact=missing_artifact,
            next_pending=next_pending,
            project_root=self.context.project_root,
        )
        has_substantive_file_artifact_progress = (
            _has_confirmed_substantive_file_artifact_progress(
                dod,
                project_root=self.context.project_root,
            )
        )
        if not completed_label or not next_pending or next_pending == completed_label:
            return
        if _should_prioritize_missing_artifact(
            dod=dod,
            next_pending=next_pending,
            missing_artifact=missing_artifact,
            project_root=self.context.project_root,
        ):
            if not has_substantive_file_artifact_progress:
                compact_handoff = _compact_missing_artifact_handoff(
                    missing_artifact,
                    project_root=self.context.project_root,
                    messages=list(getattr(self.context.session, "messages", []) or []),
                    encourage_initial_version=True,
                )
                if compact_handoff:
                    self.context.queue_steering_message(
                        f"Confirmed progress: `{completed_label}` is now satisfied by the successful "
                        f"`{tool_call.name}` result. {compact_handoff}"
                        " Do not reread reference material or spend the next turn on bookkeeping."
                    )
                    return
            self.context.queue_steering_message(
                f"Confirmed progress: `{completed_label}` is now satisfied by the successful "
                f"`{tool_call.name}` result. One declared output artifact is still missing."
                + _missing_artifact_resume_suffix(
                    missing_artifact,
                    project_root=self.context.project_root,
                    messages=list(getattr(self.context.session, "messages", []) or []),
                )
                + " Do not switch into review or consistency-check mode until the missing artifact exists."
            )
            return

        mutation_suffix = ""
        if _todo_is_mutation_step(next_pending):
            mutation_suffix = _pending_item_resume_suffix(
                dod,
                next_pending=next_pending,
                missing_artifact=missing_artifact,
                project_root=self.context.project_root,
                messages=list(getattr(self.context.session, "messages", []) or []),
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
        if not all_planned_artifact_outputs_exist(dod, project_root=self.context.project_root):
            return

        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
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
                "All explicitly planned artifacts now exist on disk. "
                f"Continue with the next pending item: `{next_pending}`. "
                "Use the files already on disk as the source of truth instead of restarting "
                "discovery or inventing alternate filenames."
                + verification_suffix
            )
            return

        if verification_commands:
            self.context.queue_steering_message(
                "All explicitly planned artifacts now exist on disk. "
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
        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
            missing_artifact=missing_artifact,
        )
        missing_artifact = _prefer_missing_artifact_for_pending_item(
            dod,
            missing_artifact=missing_artifact,
            next_pending=next_pending,
            project_root=self.context.project_root,
        )

        current_label = _current_mutation_label(tool_call)
        has_substantive_file_artifact_progress = (
            _has_confirmed_substantive_file_artifact_progress(
                dod,
                project_root=self.context.project_root,
            )
        )
        resume_target = _preferred_resume_target_path(
            dod,
            next_pending=next_pending,
            missing_artifact=missing_artifact,
            project_root=self.context.project_root,
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        resume_suffix = _pending_item_resume_suffix(
            dod,
            next_pending=next_pending,
            missing_artifact=missing_artifact,
            project_root=self.context.project_root,
            messages=list(getattr(self.context.session, "messages", []) or []),
        )
        if (
            not has_substantive_file_artifact_progress
            and _is_pure_directory_creation_tool_call(tool_call)
        ):
            if (
                next_pending
                and _todo_is_mutation_step(next_pending)
                and resume_target is not None
                and resume_target.suffix
            ):
                compact_resume = _compact_missing_artifact_handoff(
                    (resume_target, False),
                    project_root=self.context.project_root,
                    messages=list(getattr(self.context.session, "messages", []) or []),
                    encourage_initial_version=True,
                )
                if compact_resume:
                    self.context.queue_steering_message(
                        "Directory setup is complete. "
                        + compact_resume
                        + " Do not reread older reference files before that mutation."
                    )
                    return
                self.context.queue_steering_message(
                    f"Directory setup is complete. Continue with the next pending item: `{next_pending}`."
                    + resume_suffix
                    + " Do not reread older reference files before that mutation."
                )
            return
        use_persistent_handoff = _should_use_persistent_missing_artifact_handoff(
            dod,
            project_root=self.context.project_root,
        )
        session_messages = list(getattr(self.context.session, "messages", []) or [])
        if use_persistent_handoff and _recent_recovery_prompt(session_messages):
            use_persistent_handoff = False
        queue_message = (
            self.context.queue_steering_message
            if use_persistent_handoff
            else self.context.queue_ephemeral_steering_message
        )
        if resume_target is not None and resume_target.suffix:
            compact_resume = _compact_missing_artifact_handoff(
                (resume_target, False),
                project_root=self.context.project_root,
                messages=session_messages,
                encourage_initial_version=False,
            )
            if compact_resume:
                queue_message(
                    f"Confirmed progress: {current_label} is now recorded. "
                    + compact_resume
                    + " Do not reread reference material or spend the next turn on bookkeeping."
                )
                return
        todo_refresh = _todo_refresh_guidance(
            dod,
            project_root=self.context.project_root,
        )
        if not has_substantive_file_artifact_progress:
            compact_handoff = _compact_missing_artifact_handoff(
                missing_artifact,
                project_root=self.context.project_root,
                messages=session_messages,
                encourage_initial_version=False,
            )
            if compact_handoff:
                queue_message(
                    f"Confirmed progress: {current_label} is now recorded. "
                    + compact_handoff
                    + " Do not reread reference material or spend the next turn on bookkeeping."
                )
                return
        if _late_stage_missing_artifact_build(
            dod,
            project_root=self.context.project_root,
        ):
            queue_message(
                f"Confirmed progress: {current_label} is now recorded."
                + resume_suffix
                + " No TodoWrite, no verification, no rereads until that artifact exists."
            )
            return
        queue_message(
            f"Confirmed progress: {current_label} is now recorded."
            " One declared output artifact is still missing."
            + resume_suffix
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
        session_messages = list(getattr(self.context.session, "messages", []) or [])
        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
            messages=session_messages,
        )
        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
            missing_artifact=missing_artifact,
        )
        missing_artifact = _prefer_missing_artifact_for_pending_item(
            dod,
            missing_artifact=missing_artifact,
            next_pending=next_pending,
            project_root=self.context.project_root,
        )
        outputs_exist = all_planned_artifact_outputs_exist(
            dod,
            project_root=self.context.project_root,
        )
        if missing_artifact is None:
            if next_pending and _todo_is_mutation_step(next_pending) and not outputs_exist:
                pending_target = infer_pending_todo_output_target(
                    dod,
                    next_pending,
                    project_root=self.context.project_root,
                )
                if pending_target is not None:
                    concrete_message = (
                        "Todo tracking is updated. Continue with the next pending item: "
                        f"`{next_pending}`. Resume by creating `{pending_target.name}` now. "
                        f"Prefer one `write` call for `{pending_target}` instead of more rereads. "
                    )
                    if not pending_target.parent.exists():
                        concrete_message += (
                            "The `write` tool can create that file's parent directories "
                            "automatically, so do the write in one step instead of stopping "
                            "for a separate mkdir. "
                        )
                    concrete_message += (
                        "Use the current output files as the source of truth, and do not "
                        "reopen reference materials unless one specific fact required for "
                        "that step is still unknown. Make your next response the concrete "
                        "mutation tool call itself, not another bookkeeping-only turn. "
                        "Perform the mutation now instead of spending another turn on "
                        "planning, rereads, or verification."
                    )
                    self.context.queue_steering_message(concrete_message)
                    return
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
                and not outputs_exist
            ):
                self.context.queue_ephemeral_steering_message(
                    "Todo tracking is updated. Continue with the next pending item: "
                    f"`{next_pending}`. Use the current output files as the source of "
                    "truth, and do not reopen reference materials unless one specific "
                    "mismatch is still unknown."
                )
                return

            if not outputs_exist:
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
                self.context.queue_ephemeral_steering_message(
                    "Todo tracking is updated. All explicitly planned artifacts now exist on disk. "
                    f"Continue with the next pending item: `{next_pending}`. "
                    "Use the current output files as the source of truth, and do not restart "
                    "early discovery or reopen reference materials."
                    + verification_suffix
                )
                return

            if verification_commands:
                self.context.set_workflow_mode("verify")
                self.context.queue_steering_message(
                    "Todo tracking is updated. All explicitly planned artifacts now exist on disk. "
                    "Verification should run next. Use the current output files as the source of truth, "
                    "and do not restart discovery, reopen reference materials, or spend another turn "
                    "on TodoWrite alone."
                )
                return

            verification_suffix = (
                " Move to verification or final confirmation using the files already on disk."
                if verification_commands
                else " Finish the task using the files already on disk."
            )
            self.context.queue_ephemeral_steering_message(
                "Todo tracking is updated. All explicitly planned artifacts now exist on disk. "
                "Do not restart discovery, reopen reference materials, or spend another turn "
                "on TodoWrite alone. Repair or verify the current files instead of expanding the artifact set."
                + verification_suffix
            )
            return

        todo_refresh = _todo_refresh_guidance(
            dod,
            project_root=self.context.project_root,
        )
        resume_target = _preferred_resume_target_path(
            dod,
            next_pending=next_pending,
            missing_artifact=missing_artifact,
            project_root=self.context.project_root,
            messages=session_messages,
        )
        pending_target = _preferred_pending_target_path(
            dod,
            next_pending=next_pending,
            project_root=self.context.project_root,
        )
        next_pending_suffix = _pending_item_handoff_prefix(
            next_pending,
            pending_target=pending_target,
            resume_target=resume_target,
        )
        if resume_target is not None and resume_target.suffix:
            compact_resume = _compact_missing_artifact_handoff(
                (resume_target, False),
                project_root=self.context.project_root,
                messages=session_messages,
                encourage_initial_version=False,
            )
            if compact_resume:
                self.context.queue_steering_message(
                    "Todo tracking is updated. "
                    + compact_resume
                    + " Do not spend the next turn on TodoWrite alone, bookkeeping notes, "
                    "verification, or final confirmation until that artifact exists."
                )
                return
        self.context.queue_steering_message(
            "Todo tracking is updated. A declared output artifact is still missing."
            + next_pending_suffix
            + _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
                messages=session_messages,
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

        session_messages = list(getattr(self.context.session, "messages", []) or [])
        missing_artifact = _next_missing_planned_artifact(
            dod,
            project_root=self.context.project_root,
            messages=session_messages,
        )
        if missing_artifact is None:
            return

        next_pending = preferred_pending_todo_item(
            dod,
            project_root=self.context.project_root,
            missing_artifact=missing_artifact,
        )
        missing_artifact = _prefer_missing_artifact_for_pending_item(
            dod,
            missing_artifact=missing_artifact,
            next_pending=next_pending,
            project_root=self.context.project_root,
        )
        todo_refresh = _todo_refresh_guidance(
            dod,
            project_root=self.context.project_root,
        )
        if (
            next_pending
            and not _todo_is_mutation_step(next_pending)
            and not _todo_is_consistency_review_step(next_pending)
            and not _should_prioritize_missing_artifact(
                dod=dod,
                next_pending=next_pending,
                missing_artifact=(
                    missing_artifact
                    if _has_confirmed_artifact_progress(
                        dod,
                        project_root=self.context.project_root,
                    )
                    else None
                ),
                project_root=self.context.project_root,
            )
        ):
            self.context.queue_ephemeral_steering_message(
                "Bookkeeping note is recorded. Continue with the next pending item: "
                f"`{next_pending}`. Make your next response one concrete evidence-gathering "
                "tool call that advances that step, not another bookkeeping-only turn."
                + todo_refresh
                + " Do not jump ahead to later artifact creation, verification, or final "
                "confirmation until that step is satisfied."
            )
            return

        self.context.queue_ephemeral_steering_message(
            "Bookkeeping note is recorded. A declared output artifact is still missing."
            + _missing_artifact_resume_suffix(
                missing_artifact,
                project_root=self.context.project_root,
                messages=session_messages,
            )
            + todo_refresh
            + " Do not spend the next turn on additional notes, rediscovery, "
            "verification, or final confirmation until that artifact exists."
        )


def _todo_is_consistency_review_step(item: str) -> bool:
    text = item.lower()
    return any(hint in text for hint in _CONSISTENCY_REVIEW_HINTS)


def _planned_output_roots(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> tuple[str, ...]:
    planned_roots: list[str] = []
    seen_roots: set[str] = set()
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
    ):
        root = str(target if expect_directory else target.parent)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        planned_roots.append(root)
    return tuple(planned_roots)


def _extract_observation_paths(tool_call: ToolCall) -> list[str]:
    arguments = tool_call.arguments
    if tool_call.name == "read":
        file_path = str(arguments.get("file_path", "")).strip()
        return [file_path] if file_path else []

    if tool_call.name in {"glob", "grep"}:
        candidates: list[str] = []
        search_path = str(arguments.get("path", "")).strip()
        if search_path:
            anchored_path = _derive_search_anchor(
                search_path,
                str(arguments.get("pattern", "")).strip(),
            )
            candidates.append(anchored_path or search_path)
        pattern = str(arguments.get("pattern", "")).strip()
        if not search_path and pattern.startswith(("/", "~")):
            candidates.append(str(Path(pattern).expanduser().parent))
        return candidates

    command = str(arguments.get("command", "")).strip()
    if not _is_read_only_bash(command):
        return []
    return _extract_bash_paths(command)


def _derive_search_anchor(search_path: str, pattern: str) -> str:
    base = str(Path(search_path).expanduser())
    normalized_pattern = pattern.strip()
    if not normalized_pattern:
        return base
    if normalized_pattern.startswith(("~", "/")):
        pattern_path = Path(normalized_pattern).expanduser()
        try:
            return str(pattern_path.parent.resolve(strict=False))
        except Exception:
            return str(pattern_path.parent)
    if "/" in normalized_pattern:
        prefix = normalized_pattern.rsplit("/", 1)[0].strip()
        if prefix and prefix not in {".", ".."}:
            joined = Path(base).joinpath(prefix).expanduser()
            try:
                return str(joined.resolve(strict=False))
            except Exception:
                return str(joined)
    return base


def _is_read_only_bash(command: str) -> bool:
    normalized = " ".join(command.split())
    if not normalized:
        return False
    if extract_shell_text_rewrite_target(normalized) is not None:
        return False
    if any(fragment in normalized for fragment in _MUTATING_BASH_FRAGMENTS):
        return False
    try:
        argv = shlex.split(normalized)
    except ValueError:
        return False
    if not argv:
        return False
    return argv[0] in _READ_ONLY_BASH_PREFIXES


def _extract_bash_paths(command: str) -> list[str]:
    try:
        argv = shlex.split(command)
    except ValueError:
        return []
    if not argv:
        return []

    command_name = argv[0]
    if command_name == "pwd":
        return [str(Path.cwd())]

    paths: list[str] = []
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        if command_name in {"ls", "stat", "cat", "head", "tail"}:
            paths.append(arg)
            continue
        if command_name in {"find", "rg", "grep"}:
            paths.append(str(Path.cwd()) if arg in {".", "./"} else arg)
            break
    return paths


def _should_prioritize_missing_artifact(
    *,
    dod: DefinitionOfDone,
    next_pending: str | None,
    missing_artifact: tuple[Path, bool] | None,
    project_root: Path,
) -> bool:
    if missing_artifact is None:
        return False
    if not next_pending:
        return True
    if _pending_todo_conflicts_with_missing_artifact(
        dod,
        item=next_pending,
        missing_artifact=missing_artifact,
        project_root=project_root,
    ):
        return True
    if _todo_is_consistency_review_step(next_pending):
        return True
    return not _todo_is_mutation_step(next_pending)


def _pending_todo_conflicts_with_missing_artifact(
    dod: DefinitionOfDone,
    *,
    item: str,
    missing_artifact: tuple[Path, bool],
    project_root: Path,
) -> bool:
    text = item.strip().lower()
    if not text or item in _TODO_NUDGE_EXCLUDED_ITEMS:
        return False

    target, expect_directory = missing_artifact
    inferred_target = infer_pending_todo_output_target(
        dod,
        item,
        project_root=project_root,
    )
    if inferred_target is None:
        return not expect_directory and _todo_is_mutation_step(item)

    inferred_target = inferred_target.resolve(strict=False)
    target = target.resolve(strict=False)
    if expect_directory:
        return target != inferred_target and target not in inferred_target.parents
    return inferred_target != target


def _next_missing_planned_artifact(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
    messages: list[Any] | None = None,
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
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if not expect_directory or not target.is_dir():
            continue
        next_output_file, _ = infer_next_output_file(
            target=target,
            project_root=project_root,
            messages=list(messages or []),
        )
        if next_output_file is not None and not next_output_file.exists():
            return next_output_file, False
    return None


def _prefer_missing_artifact_for_pending_item(
    dod: DefinitionOfDone,
    *,
    missing_artifact: tuple[Path, bool] | None,
    next_pending: str | None,
    project_root: Path,
) -> tuple[Path, bool] | None:
    if missing_artifact is None or not next_pending:
        return missing_artifact

    inferred_target = infer_pending_todo_output_target(
        dod,
        next_pending,
        project_root=project_root,
    )
    if inferred_target is None or inferred_target.exists():
        return missing_artifact

    normalized_target = inferred_target.expanduser().resolve(strict=False)
    for planned_target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        normalized_planned = planned_target.expanduser().resolve(strict=False)
        if expect_directory and normalized_planned == normalized_target:
            return normalized_target, True
        if expect_directory:
            try:
                normalized_target.relative_to(normalized_planned)
            except ValueError:
                continue
            return normalized_target, False
        if normalized_planned == normalized_target:
            return normalized_target, False
    return missing_artifact


def _late_stage_missing_artifact_build(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> bool:
    completed = 0
    missing = 0
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=expect_directory,
            project_root=project_root,
        ):
            completed += 1
        else:
            missing += 1
    return completed >= 7 and missing > 0


def _has_confirmed_artifact_progress(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> bool:
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=expect_directory,
            project_root=project_root,
        ):
            return True
    return bool(dod.touched_files)


def _has_confirmed_file_artifact_progress(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> bool:
    return _confirmed_file_artifact_count(dod, project_root=project_root) > 0


def _has_confirmed_substantive_file_artifact_progress(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> bool:
    return _confirmed_substantive_file_artifact_count(
        dod,
        project_root=project_root,
    ) > 0


def _last_touched_file_path(dod: DefinitionOfDone) -> Path | None:
    for raw_path in reversed(dod.touched_files):
        path_text = str(raw_path or "").strip()
        if not path_text:
            continue
        candidate = Path(path_text).expanduser().resolve(strict=False)
        if candidate.suffix:
            return candidate
    return None


def _confirmed_file_artifact_count(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> int:
    count = 0
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if expect_directory:
            continue
        if planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=False,
            project_root=project_root,
        ):
            count += 1
    if count:
        return count
    return sum(
        1
        for path in dod.touched_files
        if str(path).strip()
        and Path(path).expanduser().resolve(strict=False).suffix
    )


def _confirmed_substantive_file_artifact_count(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> int:
    count = 0
    for target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if expect_directory or _is_summary_artifact_path(target):
            continue
        if planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=False,
            project_root=project_root,
        ):
            count += 1
    if count:
        return count
    return sum(
        1
        for path in dod.touched_files
        if str(path).strip()
        and Path(path).expanduser().resolve(strict=False).suffix
        and not _is_summary_artifact_path(path)
    )


def _should_use_persistent_missing_artifact_handoff(
    dod: DefinitionOfDone,
    *,
    project_root: Path,
) -> bool:
    return _confirmed_substantive_file_artifact_count(
        dod,
        project_root=project_root,
    ) == 0


def _is_summary_artifact_path(path: str | Path) -> bool:
    return Path(path).name.lower() in _SUMMARY_ARTIFACT_NAMES


def _next_missing_planned_file_within_directory(
    dod: DefinitionOfDone,
    *,
    target: Path,
    project_root: Path,
) -> Path | None:
    normalized_target = target.expanduser().resolve(strict=False)
    if normalized_target.suffix:
        return None

    for planned_target, expect_directory in collect_planned_artifact_targets(
        dod,
        project_root=project_root,
        max_paths=12,
    ):
        if expect_directory:
            continue
        normalized_planned = planned_target.expanduser().resolve(strict=False)
        try:
            normalized_planned.relative_to(normalized_target)
        except ValueError:
            continue
        if planned_artifact_target_satisfied(
            dod,
            target=normalized_planned,
            expect_directory=False,
            project_root=project_root,
        ):
            continue
        return normalized_planned
    return None


def _missing_artifact_resume_suffix(
    missing_artifact: tuple[Path, bool] | None,
    *,
    project_root: Path,
    messages: list[Any] | None = None,
) -> str:
    if missing_artifact is None:
        return ""

    target, expect_directory = missing_artifact
    return _resume_suffix_for_target(
        target,
        expect_directory=expect_directory,
        project_root=project_root,
        messages=messages,
    )


def _pending_item_resume_suffix(
    dod: DefinitionOfDone,
    *,
    next_pending: str | None,
    missing_artifact: tuple[Path, bool] | None,
    project_root: Path,
    messages: list[Any] | None = None,
) -> str:
    if next_pending:
        pending_target = infer_pending_todo_output_target(
            dod,
            next_pending,
            project_root=project_root,
        )
        if pending_target is not None and not pending_target.exists():
            normalized_target = pending_target.expanduser().resolve(strict=False)
            return _resume_suffix_for_target(
                normalized_target,
                expect_directory=not bool(normalized_target.suffix),
                project_root=project_root,
                messages=messages,
                allow_inferred_child=False,
            )
    if missing_artifact is not None and missing_artifact[1]:
        next_planned_file = _next_missing_planned_file_within_directory(
            dod,
            target=missing_artifact[0],
            project_root=project_root,
        )
        if next_planned_file is not None:
            parent_label = missing_artifact[0].name or str(missing_artifact[0])
            return (
                f" Resume by creating `{next_planned_file.name}` now."
                f" It is the next missing declared output under `{parent_label}/`."
                f" Prefer one `write` call for `{next_planned_file}` instead of more rereads."
                " Make your next response the concrete mutation tool call itself, not another"
                " bookkeeping-only turn."
            )
    return _missing_artifact_resume_suffix(
        missing_artifact,
        project_root=project_root,
        messages=messages,
    )


def _pending_item_handoff_prefix(
    next_pending: str | None,
    *,
    pending_target: Path | None,
    resume_target: Path | None,
) -> str:
    if not next_pending:
        return ""
    if (
        pending_target is None
        and resume_target is not None
        and resume_target.suffix
        and todo_describes_aggregate_mutation(next_pending)
        and not todo_describes_broad_setup_step(next_pending)
    ):
        return f" Continue with the next concrete output: `{resume_target.name}`."
    return f" Continue with the next pending item: `{next_pending}`."


def _preferred_pending_target_path(
    dod: DefinitionOfDone,
    *,
    next_pending: str | None,
    project_root: Path,
) -> Path | None:
    if not next_pending:
        return None
    pending_target = infer_pending_todo_output_target(
        dod,
        next_pending,
        project_root=project_root,
    )
    if pending_target is None:
        return None
    return pending_target.expanduser().resolve(strict=False)


def _preferred_resume_target_path(
    dod: DefinitionOfDone,
    *,
    next_pending: str | None,
    missing_artifact: tuple[Path, bool] | None,
    project_root: Path,
    messages: list[Any] | None = None,
) -> Path | None:
    if next_pending:
        pending_target = infer_pending_todo_output_target(
            dod,
            next_pending,
            project_root=project_root,
        )
        if pending_target is not None and not pending_target.exists():
            return pending_target.expanduser().resolve(strict=False)

    if missing_artifact is None:
        return None

    target, expect_directory = missing_artifact
    normalized_target = target.expanduser().resolve(strict=False)
    if not expect_directory:
        return normalized_target

    next_planned_file = _next_missing_planned_file_within_directory(
        dod,
        target=normalized_target,
        project_root=project_root,
    )
    if next_planned_file is not None:
        return next_planned_file.expanduser().resolve(strict=False)

    next_output_file, _ = infer_next_output_file(
        target=normalized_target,
        project_root=project_root,
        messages=list(messages or []),
    )
    if next_output_file is not None:
        return next_output_file.expanduser().resolve(strict=False)
    return normalized_target


def _invalid_mutation_call_shape(tool_name: str) -> str:
    if tool_name == "write":
        return "`write(file_path=..., content=...)`"
    if tool_name == "edit":
        return "`edit(file_path=..., old_string=..., new_string=...)`"
    if tool_name == "patch":
        return "`patch(file_path=..., patch='...')` or `patch(..., hunks=[...])`"
    return f"`{tool_name}(...)`"


def _extract_blocked_html_target_list(event_content: str, marker: str) -> list[str]:
    if marker not in event_content:
        return []
    tail = event_content.split(marker, 1)[1].strip()
    target_text = tail.split(". ", 1)[0].strip()
    if not target_text:
        return []
    return [item.strip() for item in target_text.split(",") if item.strip()]


def _resume_suffix_for_target(
    target: Path,
    *,
    expect_directory: bool,
    project_root: Path,
    messages: list[Any] | None = None,
    allow_inferred_child: bool = True,
) -> str:
    label = target.name or str(target)
    display_target = display_runtime_path(target)
    if expect_directory and not label.endswith("/"):
        label += "/"
    if expect_directory:
        if allow_inferred_child:
            next_output_file, next_output_source = infer_next_output_file(
                target=target,
                project_root=project_root,
                messages=list(messages or []),
            )
            if next_output_file is not None:
                guidance_origin = (
                    f"It is the next missing declared output under `{label}`."
                    if next_output_source == "declared"
                    else (
                        "It mirrors the observed filename pattern from another "
                        f"`{label}` directory you already inspected."
                    )
                )
                guidance = (
                    f" Resume by creating `{next_output_file.name}` now. {guidance_origin} "
                    f"Prefer one `write` call for "
                    f"`{display_runtime_path(next_output_file)}` instead of more rereads."
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
                f"concrete `write` call for a file inside `{display_target}` instead of more rereads."
                " Make your next response the concrete mutation tool call itself, not another"
                " bookkeeping-only turn."
            )
        return (
            f" Resume by creating `{label}` now. Prefer one concrete directory-creation "
            f"step for `{display_target}` instead of more rereads."
        )
    guidance = (
        f" Resume by creating `{label}` now. Prefer one `write` call for `{display_target}` "
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


def _compact_missing_artifact_handoff(
    missing_artifact: tuple[Path, bool] | None,
    *,
    project_root: Path,
    messages: list[Any] | None = None,
    encourage_initial_version: bool = False,
) -> str:
    """Build a shorter first-mutation handoff once the next output target is known."""

    if missing_artifact is None:
        return ""

    target, expect_directory = missing_artifact
    label = target.name or str(target)
    display_target = display_runtime_path(target)
    if expect_directory and not label.endswith("/"):
        label += "/"
    if expect_directory:
        next_output_file, _ = infer_next_output_file(
            target=target,
            project_root=project_root,
            messages=list(messages or []),
        )
        if next_output_file is None:
            if target.is_dir():
                return (
                    f"Next step: create the next output file under `{label}`. Prefer one "
                    f"concrete `write` call inside `{display_target}` now."
                )
            return (
                f"Next step: create `{label}`. Prefer one concrete directory-creation step "
                f"for `{display_target}` now."
            )
        guidance = (
            f"Next step: create `{next_output_file.name}`. Prefer one "
            f"`write(file_path=..., content=...)` call for `{display_runtime_path(next_output_file)}` now."
        )
        if not next_output_file.parent.exists():
            guidance += (
                " The `write` tool can create that file's parent directories automatically."
            )
        if encourage_initial_version:
            guidance += (
                " Write a compact but real initial version of that file now; you can expand "
                "or refine it in later edits."
            )
        guidance += " Make your next response the concrete mutation tool call itself."
        return guidance

    guidance = (
        f"Next step: create `{label}`. Prefer one "
        f"`write(file_path=..., content=...)` call for `{display_target}` now."
    )
    if not target.parent.exists():
        guidance += (
            " The `write` tool can create that file's parent directories automatically."
        )
    if encourage_initial_version:
        guidance += (
            " Write a compact but real initial version of that file now; you can expand "
            "or refine it in later edits."
        )
    guidance += " Make your next response the concrete mutation tool call itself."
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


def _should_plan_verification_for_tool_call(
    dod: DefinitionOfDone,
    *,
    tool_call: ToolCall,
    project_root: Path,
) -> bool:
    actionable_pending = [
        item
        for item in effective_pending_todo_items(
            dod,
            project_root=project_root,
        )
        if item not in _TODO_NUDGE_EXCLUDED_ITEMS
    ]
    if any(
        _todo_is_mutation_step(item) or _todo_is_consistency_review_step(item)
        for item in actionable_pending
    ):
        return False
    if tool_call.name in {"write", "edit", "patch"}:
        return True
    if tool_call.name != "bash":
        return False
    if any(
        Path(path).expanduser().resolve(strict=False).suffix
        for path in dod.touched_files
        if str(path).strip()
    ):
        return True
    return any(
        not expect_directory
        and planned_artifact_target_satisfied(
            dod,
            target=target,
            expect_directory=False,
            project_root=project_root,
        )
        for target, expect_directory in collect_planned_artifact_targets(
            dod,
            project_root=project_root,
            max_paths=12,
        )
    )


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


def _is_pure_directory_creation_tool_call(tool_call: ToolCall) -> bool:
    if tool_call.name != "bash":
        return False
    command = str(tool_call.arguments.get("command", "")).strip()
    if not command or any(
        operator in command for operator in ("&&", "||", ";", "|", "$(", ">", "<")
    ):
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return bool(parts) and parts[0] == "mkdir"


def _recent_recovery_prompt(messages: list[Any]) -> bool:
    for message in reversed(messages[-4:]):
        role = getattr(message, "role", None)
        if getattr(role, "value", role) != "user":
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        if content.startswith("[EMPTY ASSISTANT RESPONSE]"):
            return True
        if content.startswith("[CONTINUE CURRENT STEP]"):
            return True
    return False


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
