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
from .finalization import TurnFinalizer
from .phases import TurnPhase, TurnPhaseTracker, TurnTransitionKind
from .repair import ResponseRepairer
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .turn_completion import TurnCompletionController
from .turn_iteration import TurnIterationAction, TurnIterationController
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
        self.turn_iteration = TurnIterationController(
            agent,
            tracer=self.tracer,
            phase_tracker=self.phase_tracker,
            turn_requester=AssistantTurnRequester(agent, self.tracer),
            repairer=self.repairer,
            tool_batches=ToolBatchRunner(agent, self.dod_store),
            turn_completion=self.turn_completion,
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

            assert self.executor is not None
            iteration_decision = await self.turn_iteration.run_iteration(
                task=task,
                effective_task=effective_task,
                original_task=original_task,
                effective_max_tokens=effective_max_tokens,
                iterations=iterations,
                max_iterations=self.agent.config.max_iterations,
                actions_taken=actions_taken,
                continuation_count=continuation_count,
                empty_retry_count=empty_retry_count,
                max_empty_retries=max_empty_retries,
                extracted_iterations=extracted_iterations,
                max_extracted_iterations=max_extracted_iterations,
                consecutive_errors=consecutive_errors,
                dod=dod,
                emit=emit,
                summary=summary,
                executor=self.executor,
                rollback_plan=rollback_plan,
                on_confirmation=on_confirmation,
                on_user_question=on_user_question,
                emit_confirmation=self._emit_confirmation(emit),
            )
            continuation_count = iteration_decision.continuation_count
            empty_retry_count = iteration_decision.empty_retry_count
            extracted_iterations = iteration_decision.extracted_iterations
            consecutive_errors = iteration_decision.consecutive_errors
            actions_taken.extend(iteration_decision.new_actions_taken)
            if iteration_decision.action == TurnIterationAction.CONTINUE:
                continue
            if iteration_decision.action == TurnIterationAction.FINALIZE:
                return await self._finalize_turn(
                    summary,
                    emit,
                    reason_code=iteration_decision.finalize_reason_code
                    or "turn_complete",
                    reason_summary=iteration_decision.finalize_reason_summary
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
