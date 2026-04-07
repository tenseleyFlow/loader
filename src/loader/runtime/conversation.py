"""Typed turn engine for Loader runtime execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..llm.base import Message, Role
from .artifact_invalidation import ArtifactInvalidationAssessor
from .assistant_turns import AssistantTurnRequester
from .completion_policy import CompletionPolicy
from .dod import DefinitionOfDone, DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import TurnFinalizer, merge_usage
from .phases import TurnPhase, TurnPhaseTracker, TurnTransitionKind
from .repair import ResponseRepairer
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .turn_completion import TurnCompletionAction, TurnCompletionController
from .turn_preparation import TurnPreparationController
from .workflow import (
    ModeDecision,
    WorkflowArtifactStore,
    WorkflowDecisionKind,
    WorkflowPolicy,
    WorkflowSignalExtractor,
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
    build_execute_bridge,
)
from .workflow_lanes import WorkflowLaneRunner
from .workflow_recovery import WorkflowRecoveryController

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


class ConversationRuntime:
    """Runs one explicit conversation turn against the current session."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.tracer = RuntimeTracer()
        self.executor: ToolExecutor | None = None
        self.dod_store = DefinitionOfDoneStore(agent.project_root)
        self.workflow_signals = WorkflowSignalExtractor()
        self.workflow_policy = WorkflowPolicy(self.workflow_signals)
        self.artifact_invalidation = ArtifactInvalidationAssessor()
        self.artifact_store = WorkflowArtifactStore(agent.project_root)
        self.turn_requester = AssistantTurnRequester(agent, self.tracer)
        self.tool_batches = ToolBatchRunner(agent, self.dod_store)
        self.workflow_lanes = WorkflowLaneRunner(
            agent,
            artifact_store=self.artifact_store,
            dod_store=self.dod_store,
            workflow_policy=self.workflow_policy,
        )
        self.workflow_recovery = WorkflowRecoveryController(
            agent,
            artifact_invalidation=self.artifact_invalidation,
            workflow_policy=self.workflow_policy,
            workflow_signals=self.workflow_signals,
            workflow_lanes=self.workflow_lanes,
            set_workflow_mode=self._set_workflow_mode,
            append_timeline=self._append_workflow_timeline_from_decision,
            append_execute_bridge=self._maybe_append_execute_bridge,
        )
        self.repairer = ResponseRepairer(agent)
        self.completion_policy = CompletionPolicy(agent)
        self.phase_tracker = TurnPhaseTracker(agent, self.tracer)
        self.finalizer = TurnFinalizer(
            agent,
            self.tracer,
            self.dod_store,
            self._set_workflow_mode,
        )
        self.turn_completion = TurnCompletionController(
            agent,
            repairer=self.repairer,
            completion_policy=self.completion_policy,
            finalizer=self.finalizer,
            phase_tracker=self.phase_tracker,
        )
        self.turn_preparation = TurnPreparationController(
            agent,
            tracer=self.tracer,
            phase_tracker=self.phase_tracker,
            dod_store=self.dod_store,
            workflow_policy=self.workflow_policy,
            workflow_signals=self.workflow_signals,
            workflow_lanes=self.workflow_lanes,
            finalizer=self.finalizer,
            set_workflow_mode=self._set_workflow_mode,
            append_timeline=self._append_workflow_timeline_from_decision,
            append_execute_bridge=self._maybe_append_execute_bridge,
        )

    async def run_turn(
        self,
        task: str,
        emit: EventSink,
        on_confirmation: ConfirmationHandler = None,
        on_user_question: UserQuestionHandler = None,
        requested_mode: str | None = None,
        original_task: str | None = None,
    ) -> TurnSummary:
        """Run one task turn and return a structured summary."""

        iterations = 0
        final_response = ""
        actions_taken: list[str] = []
        continuation_count = 0
        empty_retry_count = 0
        max_empty_retries = 5
        extracted_iterations = 0
        max_extracted_iterations = 3
        consecutive_errors = 0

        prepared_turn = await self.turn_preparation.prepare(
            task=task,
            emit=emit,
            requested_mode=requested_mode,
            original_task=original_task,
            on_user_question=on_user_question,
        )
        self.executor = prepared_turn.executor
        summary = prepared_turn.summary
        dod = prepared_turn.definition_of_done
        task = prepared_turn.task
        effective_task = prepared_turn.effective_task
        effective_max_tokens = prepared_turn.effective_max_tokens
        rollback_plan = prepared_turn.rollback_plan

        while iterations < self.agent.config.max_iterations:
            iterations += 1
            summary.iterations = iterations
            self.tracer.record("turn.iteration_started", iteration=iterations)

            if iterations == 1 and len(self.agent.messages) == 1:
                task_lower = task.lower()
                action_keywords = [
                    "create",
                    "write",
                    "make",
                    "run",
                    "execute",
                    "build",
                    "install",
                    "delete",
                    "remove",
                    "add",
                    "edit",
                    "modify",
                    "update",
                    "fix",
                ]
                if any(keyword in task_lower for keyword in action_keywords):
                    self.agent.session.append(Message(role=Role.ASSISTANT, content="["))

            steering_messages = self.agent._drain_steering_queue()
            for steering_message in steering_messages:
                await emit(AgentEvent(type="steering", content=steering_message))
                self.agent.session.append(
                    Message(
                        role=Role.USER,
                        content=f"[USER INTERRUPTION]: {steering_message}",
                    )
                )

            if await self.workflow_recovery.maybe_refresh_plan_for_drift(
                task=original_task or task,
                dod=dod,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                executor=self.executor,
            ):
                continue

            await self.phase_tracker.enter(
                TurnPhase.ASSISTANT,
                emit,
                detail="Requesting assistant response",
                reason_code="request_assistant_response",
            )
            await emit(AgentEvent(type="thinking"))
            assistant_turn = await self.turn_requester.request_turn(
                emit=emit,
                max_tokens=effective_max_tokens,
            )
            merge_usage(summary.usage, assistant_turn.usage)

            content = assistant_turn.content
            response_content = assistant_turn.response_content
            tool_calls = list(assistant_turn.tool_calls)
            pending_tool_calls_seen = set(assistant_turn.pending_tool_calls_seen)

            if not content.strip():
                await self.phase_tracker.enter(
                    TurnPhase.REPAIR,
                    emit,
                    detail="Repairing empty assistant response",
                    reason_code="repair_empty_response",
                    kind=TurnTransitionKind.RETRY,
                )
                empty_retry_count += 1
                empty_decision = self.repairer.handle_empty_response(
                    task=task,
                    original_task=original_task,
                    empty_retry_count=empty_retry_count,
                    max_empty_retries=max_empty_retries,
                )
                if empty_decision.should_continue and empty_decision.retry_prompt:
                    self.agent.session.append(
                        Message(
                            role=Role.ASSISTANT,
                            content=empty_decision.retry_prompt,
                        )
                    )
                    continue

                final_response = empty_decision.final_response or ""
                summary.final_response = final_response
                if empty_decision.failure:
                    summary.failures.append(empty_decision.failure)
                await emit(AgentEvent(type="response", content=final_response))
                break

            analysis = self.repairer.analyze_response(
                content=content,
                response_content=response_content,
                tool_calls=tool_calls,
                extracted_iterations=extracted_iterations,
                max_extracted_iterations=max_extracted_iterations,
            )
            content = analysis.content
            tool_calls = list(analysis.tool_calls)
            tool_source = analysis.tool_source
            extracted_iterations = analysis.extracted_iterations
            if analysis.clear_stream:
                await self.phase_tracker.enter(
                    TurnPhase.REPAIR,
                    emit,
                    detail="Repairing raw-text tool fallback",
                    reason_code="repair_raw_text_tool_fallback",
                    kind=TurnTransitionKind.REROUTE,
                )
                await emit(AgentEvent(type="clear_stream"))

            if analysis.is_final_answer:
                assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                self.agent.session.append(assistant_message)
                summary.assistant_messages.append(assistant_message)
                final_response = analysis.final_response or content
                summary.final_response = final_response
                self.tracer.record("turn.completed", reason="final_answer")
                await emit(AgentEvent(type="response", content=final_response))
                break

            if tool_calls:
                if analysis.should_stop:
                    assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                    self.agent.session.append(assistant_message)
                    summary.assistant_messages.append(assistant_message)
                    final_response = analysis.final_response or content
                    summary.final_response = final_response
                    if analysis.failure:
                        summary.failures.append(analysis.failure)
                    await emit(AgentEvent(type="response", content=final_response))
                    break

                await self.phase_tracker.enter(
                    TurnPhase.TOOLS,
                    emit,
                    detail="Executing tool batch",
                    reason_code="execute_tool_batch",
                )
                assistant_message = Message(
                    role=Role.ASSISTANT,
                    content=response_content,
                    tool_calls=tool_calls,
                )
                self.agent.session.append(assistant_message)
                summary.assistant_messages.append(assistant_message)
                self.tracer.record(
                    "assistant.tool_batch",
                    tool_count=len(tool_calls),
                    source=tool_source,
                )

                batch_result = await self.tool_batches.execute_batch(
                    tool_calls=tool_calls,
                    tool_source=tool_source,
                    pending_tool_calls_seen=pending_tool_calls_seen,
                    emit=emit,
                    summary=summary,
                    dod=dod,
                    executor=self.executor,
                    on_confirmation=on_confirmation,
                    on_user_question=on_user_question,
                    emit_confirmation=self._emit_confirmation(emit),
                    consecutive_errors=consecutive_errors,
                )
                actions_taken.extend(batch_result.actions_taken)
                consecutive_errors = batch_result.consecutive_errors
                if batch_result.halted:
                    return await self._finalize_turn(
                        summary,
                        emit,
                        reason_code="tool_batch_halted",
                        reason_summary="Finalizing after halted tool batch",
                    )

                continue

            assert self.executor is not None
            completion_decision = await self.turn_completion.handle_text_response(
                content=content,
                response_content=response_content,
                task=task,
                effective_task=effective_task,
                iterations=iterations,
                max_iterations=self.agent.config.max_iterations,
                actions_taken=actions_taken,
                continuation_count=continuation_count,
                dod=dod,
                emit=emit,
                summary=summary,
                executor=self.executor,
                rollback_plan=rollback_plan,
            )
            continuation_count = completion_decision.continuation_count
            if completion_decision.action == TurnCompletionAction.CONTINUE:
                continue
            if completion_decision.action == TurnCompletionAction.FINALIZE:
                return await self._finalize_turn(
                    summary,
                    emit,
                    reason_code=completion_decision.finalize_reason_code
                    or "turn_complete",
                    reason_summary=completion_decision.finalize_reason_summary
                    or "Finalizing completed turn",
                )
            break

        return await self._finalize_turn(
            summary,
            emit,
            reason_code="turn_complete",
            reason_summary="Finalizing completed turn",
        )

    async def _finalize_turn(
        self,
        summary: TurnSummary,
        emit: EventSink,
        *,
        reason_code: str,
        reason_summary: str,
    ) -> TurnSummary:
        await self.phase_tracker.enter(
            TurnPhase.FINALIZE,
            emit,
            detail=reason_summary,
            reason_code=reason_code,
            kind=TurnTransitionKind.TERMINAL,
        )
        final_summary = self.finalizer.finalize_summary(summary)
        self.phase_tracker.clear()
        return final_summary

    async def _set_workflow_mode(
        self,
        decision: ModeDecision,
        *,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
    ) -> None:
        mode = decision.mode
        self.agent.set_workflow_mode(mode.value)
        self.agent.session.update_runtime_state(
            workflow_mode=mode.value,
            workflow_reason_code=decision.reason_code,
            workflow_reason_summary=decision.reason_summary,
            workflow_decision_kind=decision.decision_kind.value,
            workflow_ambiguity_score=decision.ambiguity_score,
            workflow_complexity_score=decision.complexity_score,
            workflow_scheduled_next_mode=(
                decision.scheduled_next_mode.value
                if decision.scheduled_next_mode is not None
                else None
            ),
        )
        dod.current_mode = mode.value
        if not dod.mode_history or dod.mode_history[-1] != mode.value:
            dod.mode_history.append(mode.value)
        summary.workflow_mode = mode.value
        summary.workflow_reason_code = decision.reason_code
        summary.workflow_reason_summary = decision.reason_summary
        summary.workflow_decision_kind = decision.decision_kind.value
        self._append_workflow_timeline_from_decision(
            decision,
            kind={
                WorkflowDecisionKind.HANDOFF: WorkflowTimelineEntryKind.HANDOFF,
                WorkflowDecisionKind.REENTRY: WorkflowTimelineEntryKind.REENTRY,
            }.get(decision.decision_kind, WorkflowTimelineEntryKind.ROUTE),
            summary=summary,
            artifact_paths=[
                path
                for path in (
                    dod.clarify_brief,
                    dod.implementation_plan,
                    dod.verification_plan,
                )
                if path
            ],
        )
        summary.definition_of_done = dod
        self.dod_store.save(dod)
        await emit(
            AgentEvent(
                type="workflow_mode",
                content=f"Workflow: {mode.value} ({decision.reason_summary})",
                workflow_mode=mode.value,
                definition_of_done=dod,
            )
        )

    def _append_workflow_timeline_from_decision(
        self,
        decision: ModeDecision,
        *,
        kind: WorkflowTimelineEntryKind,
        summary: TurnSummary | None = None,
        artifact_paths: list[str] | None = None,
    ) -> None:
        entry = WorkflowTimelineEntry.from_decision(
            decision,
            kind=kind,
            prompt_format=self.agent.prompt_format,
            prompt_sections=self.agent.prompt_sections,
            artifact_paths=artifact_paths,
        )
        self.agent.session.append_workflow_timeline_entry(entry)
        if summary is not None:
            summary.workflow_timeline = list(self.agent.session.workflow_timeline)

    def _maybe_append_execute_bridge(self, dod: DefinitionOfDone) -> None:
        bridge = build_execute_bridge(
            Path(dod.clarify_brief) if dod.clarify_brief else None,
            Path(dod.implementation_plan) if dod.implementation_plan else None,
            Path(dod.verification_plan) if dod.verification_plan else None,
        )
        if bridge and not any(
            message.role == Role.USER and "[WORKFLOW BRIDGE]" in message.content
            for message in self.agent.messages[-4:]
        ):
            self.agent.session.append(
                Message(
                    role=Role.USER,
                    content=(
                        "[WORKFLOW BRIDGE]\n"
                        f"{bridge}\n\n"
                        "Honor these artifacts while you execute the task. "
                        "Keep TodoWrite current when the work spans multiple steps."
                    ),
                )
            )

    @staticmethod
    def _emit_confirmation(emit: EventSink):
        async def _emit(tool_name: str, message: str, details: str) -> None:
            await emit(
                AgentEvent(
                    type="confirmation",
                    tool_name=tool_name,
                    confirm_message=message,
                    confirm_details=details,
                )
            )

        return _emit
