"""Tool-batch execution and recovery bookkeeping for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.base import ToolCall
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
from .tool_batch_checks import ToolBatchConfidenceGate, ToolBatchVerificationGate
from .tool_batch_recovery import ToolBatchRecoveryController
from .verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
)
from .workflow import sync_todos_to_definition_of_done

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = (
    Callable[[str, str, str, dict[str, Any] | None], Awaitable[bool]] | None
)
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None

_VERIFY_ITEM = "Collect verification evidence"


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
                    tool_call=tool_call,
                    outcome=outcome,
                    emit=emit,
                )
                if recovery_result is not None:
                    summary.tool_result_messages.append(recovery_result)
                    self.context.session.append(recovery_result)
                    continue

            if outcome.state == ToolExecutionState.EXECUTED and not outcome.is_error:
                loop_response = await self._record_successful_execution(
                    tool_call=tool_call,
                    outcome=outcome,
                    dod=dod,
                    emit=emit,
                    summary=summary,
                )
                # Mark this tool's label as completed and emit live progress
                label = _tool_call_label(tool_call)
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
                    tool_name=tool_call.name,
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
        self.dod_store.save(dod)
        self.context.recovery_context = None
        return None


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
