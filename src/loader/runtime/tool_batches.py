"""Tool-batch execution and recovery bookkeeping for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.base import Role, ToolCall
from .compaction import infer_preferred_next_step, summarize_confirmed_facts
from .context import RuntimeContext
from .dod import (
    DefinitionOfDone,
    DefinitionOfDoneStore,
    begin_new_verification_attempt,
    derive_verification_commands,
    ensure_active_verification_attempt,
    is_state_mutating_tool_call,
    record_successful_tool_call,
    synthesize_todo_items,
)
from .events import AgentEvent, TurnSummary
from .evidence_provenance import EvidenceProvenance, EvidenceProvenanceStatus
from .executor import ToolExecutionState, ToolExecutor
from .logging import get_runtime_logger
from .policy_timeline import append_verification_timeline_entry
from .safeguard_services import extract_shell_text_rewrite_target
from .semantic_rules import html_toc as html_toc_rule
from .tool_batch_checks import ToolBatchConfidenceGate, ToolBatchVerificationGate
from .tool_batch_recovery import ToolBatchRecoveryController
from .verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
)
from .workflow import advance_todos_from_tool_call, sync_todos_to_definition_of_done

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
        self._inventory_hint_targets: set[str] = set()

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
                self._annotate_verified_html_inventory(executed_tool_call, outcome)
                self._queue_verified_html_inventory_nudge(executed_tool_call)
                self._annotate_validated_html_toc_completion(executed_tool_call, outcome)
                self._queue_validated_html_toc_completion_nudge(executed_tool_call)
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
        next_pending = next(
            (
                item
                for item in dod.pending_items
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        confirmed_facts = summarize_confirmed_facts(
            self.context.session.messages,
            max_items=2,
        )
        if next_pending and not html_toc_rule.task_targets_html_toc(current_task):
            if confirmed_facts:
                self.context.queue_steering_message(
                    "Reuse the earlier observation instead of repeating it. "
                    f"Confirmed facts: {confirmed_facts}. "
                    f"Continue with the next pending item: `{next_pending}`. "
                    "Only gather more evidence if a specific fact required for that step is still unknown."
                )
            else:
                self.context.queue_steering_message(
                    "Reuse the earlier observation instead of repeating it. "
                    f"Continue with the next pending item: `{next_pending}`. "
                    "Only gather more evidence if a specific fact required for that step is still unknown."
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

    def _queue_blocked_html_edit_nudge(self, tool_call: ToolCall, event_content: str) -> None:
        """Steer blocked TOC edits back to the confirmed chapter inventory."""

        if tool_call.name not in {"edit", "patch"}:
            return
        if not self._targets_html_toc_task():
            return

        target_path = str(tool_call.arguments.get("file_path", "")).strip()
        if not html_toc_rule.is_html_toc_index_path(target_path):
            return

        validation = html_toc_rule.validate_html_toc(target_path)
        if (
            "old_string and new_string are identical" in event_content
            and validation is not None
            and validation.valid
        ):
            action_tracker = getattr(self.context.safeguards, "action_tracker", None)
            note_validated = getattr(action_tracker, "note_validated_html_toc", None)
            if callable(note_validated):
                note_validated(target_path)
            target_label = html_toc_rule.describe_html_toc_target(target_path)
            self.context.queue_steering_message(
                f"The HTML table-of-contents target {target_label} already matches the "
                "validated replacement block. "
                f"Semantic verification preview: validated {validation.link_count} linked "
                "entries. "
                "Do not call `edit`, `patch`, or reread the same TOC again. Briefly state "
                f"that {target_label} is already updated so Loader can continue the "
                "verification gate or finish the task."
            )
            return

        current_task = getattr(self.context.session, "current_task", None)
        confirmed_facts = summarize_confirmed_facts(
            self.context.session.messages,
            max_items=2,
            focus_path=target_path,
        )
        preferred_next_step = infer_preferred_next_step(
            self.context.session.messages,
            current_task=current_task,
            focus_path=target_path,
        )
        verified_inventory = html_toc_rule.summarize_html_inventory(target_path, limit=12)
        current_excerpt = html_toc_rule.extract_html_toc_excerpt(target_path)
        suggested_replacement = html_toc_rule.build_html_toc_replacement_block(target_path)
        suggested_call = html_toc_rule.build_html_toc_edit_call_template(target_path)
        target_label = html_toc_rule.describe_html_toc_target(target_path)
        excerpt_suffix = (
            f"\nCurrent TOC block:\n{current_excerpt}"
            if current_excerpt
            else ""
        )
        replacement_suffix = (
            f"\nSuggested replacement block:\n{suggested_replacement}"
            if suggested_replacement
            else ""
        )
        call_suffix = (
            f"\nSuggested edit call:\n{suggested_call}"
            if suggested_call
            else ""
        )

        if preferred_next_step and confirmed_facts and verified_inventory:
            self.context.queue_steering_message(
                f"Use the current TOC target contents plus the verified sibling inventory for "
                f"{target_label} instead of guessing. "
                f"Confirmed facts: {confirmed_facts}. "
                f"Known chapter inventory: {verified_inventory}. "
                f"{preferred_next_step} "
                f"Apply those exact href/title pairs in {target_label}. "
                "Do not rewrite the whole document. For `edit`, set `old_string` to the "
                "current TOC block above exactly and set `new_string` to the suggested "
                "replacement block below exactly."
                f"{excerpt_suffix}"
                f"{replacement_suffix}"
                f"{call_suffix}"
            )
            return

        if verified_inventory:
            self.context.queue_steering_message(
                f"Use the current TOC target contents plus the verified sibling inventory for "
                f"{target_label} instead of guessing. "
                f"Known chapter inventory: {verified_inventory}. "
                f"Apply those exact href/title pairs in {target_label}. "
                "Do not rewrite the whole document. For `edit`, set `old_string` to the "
                "current TOC block above exactly and set `new_string` to the suggested "
                "replacement block below exactly."
                f"{excerpt_suffix}"
                f"{replacement_suffix}"
                f"{call_suffix}"
            )
            return

        self.context.queue_steering_message(
            f"Use the current TOC target contents when retrying the edit for {target_label} "
            "instead of guessing. "
            f"{excerpt_suffix}".strip()
        )

    def _queue_verified_html_inventory_nudge(self, tool_call: ToolCall) -> None:
        """Proactively hand off verified chapter inventory after sibling discovery."""

        if tool_call.name != "glob":
            return

        chapters_path = str(tool_call.arguments.get("path", "")).strip()
        if not chapters_path.endswith("chapters"):
            return

        index_path = str(Path(chapters_path).expanduser().parent / "index.html")
        if index_path in self._inventory_hint_targets:
            return

        if not self._targets_html_toc_task():
            return

        verified_inventory = html_toc_rule.summarize_html_inventory(index_path, limit=12)
        if not verified_inventory:
            return

        self._inventory_hint_targets.add(index_path)
        target_label = html_toc_rule.describe_html_toc_target(index_path)
        chapters_label = html_toc_rule.describe_html_toc_chapters_dir(index_path)
        self.context.queue_steering_message(
            f"You already have the verified sibling inventory needed for {target_label}. "
            f"Known chapter inventory: {verified_inventory}. "
            f"Update {target_label} using those exact href/title pairs instead of rereading "
            f"files in {chapters_label} unless one specific title is still unknown."
        )

    def _annotate_verified_html_inventory(self, tool_call: ToolCall, outcome) -> None:
        """Attach verified chapter inventory directly to a successful discovery result."""

        if tool_call.name != "glob":
            return

        chapters_path = str(tool_call.arguments.get("path", "")).strip()
        if not chapters_path.endswith("chapters"):
            return

        if not self._targets_html_toc_task():
            return

        index_path = str(Path(chapters_path).expanduser().parent / "index.html")
        verified_inventory = html_toc_rule.summarize_html_inventory(index_path, limit=12)
        if not verified_inventory:
            return

        action_tracker = getattr(self.context.safeguards, "action_tracker", None)
        note_inventory = getattr(action_tracker, "note_verified_html_inventory", None)
        if callable(note_inventory):
            note_inventory(index_path)

        note = f"Verified chapter inventory: {verified_inventory}"
        merged_event = outcome.event_content
        if note not in merged_event:
            merged_event = f"{note}\n{merged_event}".strip()
            outcome.event_content = merged_event
            outcome.result_output = merged_event
            outcome.message.content = f"{note}\n{outcome.message.content}".strip()
            if outcome.message.tool_results:
                outcome.message.tool_results[0].content = merged_event

    def _annotate_validated_html_toc_completion(self, tool_call: ToolCall, outcome) -> None:
        """Attach semantic TOC validation evidence to a successful mutating result."""

        if not self._targets_html_toc_task():
            return
        target_path = self._validated_html_toc_target(tool_call)
        if target_path is None:
            return

        validation = html_toc_rule.validate_html_toc(target_path)
        if validation is None or not validation.valid:
            return

        action_tracker = getattr(self.context.safeguards, "action_tracker", None)
        note_validated = getattr(action_tracker, "note_validated_html_toc", None)
        if callable(note_validated):
            note_validated(target_path)

        note = (
            "Semantic verification preview: "
            f"validated {validation.link_count} toc links in {Path(target_path).name}"
        )
        merged_event = outcome.event_content
        if note not in merged_event:
            merged_event = f"{merged_event}\n{note}".strip()
            outcome.event_content = merged_event
            outcome.result_output = merged_event
            outcome.message.content = f"{outcome.message.content}\n{note}".strip()
            if outcome.message.tool_results:
                outcome.message.tool_results[0].content = merged_event

    def _queue_validated_html_toc_completion_nudge(self, tool_call: ToolCall) -> None:
        """Push the next model turn toward finishing once the TOC already validates."""

        if not self._targets_html_toc_task():
            return
        target_path = self._validated_html_toc_target(tool_call)
        if target_path is None:
            return

        validation = html_toc_rule.validate_html_toc(target_path)
        if validation is None or not validation.valid:
            return

        if tool_call.name == "read":
            target_label = html_toc_rule.describe_html_toc_target(target_path)
            chapters_label = html_toc_rule.describe_html_toc_chapters_dir(target_path)
            self.context.queue_steering_message(
                f"The HTML table-of-contents target {target_label} already satisfies the "
                "verified link/title constraints. "
                f"Semantic verification preview: validated {validation.link_count} linked "
                "entries. "
                "No TOC edit is required unless you can point to one specific incorrect href or "
                f"title. Do not reread {target_label} or files in {chapters_label} again. "
                "Briefly state that the table of contents is already correct so Loader can "
                "finish the task."
            )
            return

        target_label = html_toc_rule.describe_html_toc_target(target_path)
        chapters_label = html_toc_rule.describe_html_toc_chapters_dir(target_path)
        self.context.queue_steering_message(
            f"The HTML table-of-contents target {target_label} already satisfies the "
            "verified link/title constraints. "
            f"Semantic verification preview: validated {validation.link_count} linked "
            "entries. "
            f"Do not reread {target_label} or files in {chapters_label} unless a specific "
            "href or title is still unresolved. Briefly state that the table of contents has "
            "been updated so Loader can run the verification gate."
        )

    @staticmethod
    def _validated_html_toc_target(tool_call: ToolCall) -> str | None:
        """Return the index target for a validated HTML TOC action."""

        target_path = ""
        if tool_call.name in {"write", "edit", "patch", "read"}:
            target_path = str(tool_call.arguments.get("file_path", "")).strip()
        elif tool_call.name == "bash":
            target_path = (
                extract_shell_text_rewrite_target(
                    str(tool_call.arguments.get("command", ""))
                )
                or ""
            ).strip()

        if not target_path:
            return None
        if not html_toc_rule.is_html_toc_index_path(target_path):
            return None
        return str(Path(target_path).expanduser())

    def _targets_html_toc_task(self) -> bool:
        current_task = str(getattr(self.context.session, "current_task", "") or "").lower()
        if not current_task:
            for message in reversed(getattr(self.context.session, "messages", [])):
                if getattr(message, "role", None) != Role.USER:
                    continue
                content = str(getattr(message, "content", "") or "").strip().lower()
                if content:
                    current_task = content
                    break
        return html_toc_rule.task_targets_html_toc(current_task)

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
                sync_todos_to_definition_of_done(dod, new_todos)
        else:
            pending_before = list(dod.pending_items)
            if advance_todos_from_tool_call(dod, tool_call):
                self._queue_next_pending_todo_nudge(
                    tool_call=tool_call,
                    pending_before=pending_before,
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
                for item in dod.pending_items
                if item not in _TODO_NUDGE_EXCLUDED_ITEMS
            ),
            None,
        )
        if not completed_label or not next_pending or next_pending == completed_label:
            return

        self.context.queue_steering_message(
            f"Confirmed progress: `{completed_label}` is now satisfied by the successful "
            f"`{tool_call.name}` result. Continue with the next pending item: "
            f"`{next_pending}` instead of rereading the same evidence."
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
