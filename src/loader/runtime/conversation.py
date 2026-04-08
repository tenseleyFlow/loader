"""Typed turn engine for Loader runtime execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .artifact_invalidation import ArtifactInvalidationAssessor
from .assistant_turns import AssistantTurnRequester
from .completion_policy import CompletionPolicy
from .dod import DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import TurnFinalizer
from .phases import TurnPhase, TurnPhaseTracker, TurnTransitionKind
from .repair import ResponseRepairer
from .response_routing import AssistantResponseRouter
from .tool_batches import ToolBatchRunner
from .tracing import RuntimeTracer
from .turn_completion import TurnCompletionController
from .turn_iteration import TurnIterationController
from .turn_loop import TurnLoopController
from .turn_preamble import TurnPreludeController
from .turn_preparation import TurnPreparationController
from .workflow import (
    WorkflowArtifactStore,
    WorkflowPolicy,
    WorkflowSignalExtractor,
)
from .workflow_lanes import WorkflowLaneRunner
from .workflow_recovery import WorkflowRecoveryController
from .workflow_state import WorkflowStateController

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None


class ConversationRuntime:
    """Runs one explicit conversation turn against the current session."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.context = agent._build_runtime_context()
        self.tracer = RuntimeTracer()
        self.executor: ToolExecutor | None = None
        self.dod_store = DefinitionOfDoneStore(agent.project_root)
        self.workflow_signals = WorkflowSignalExtractor()
        self.workflow_policy = WorkflowPolicy(self.workflow_signals)
        self.artifact_invalidation = ArtifactInvalidationAssessor()
        self.artifact_store = WorkflowArtifactStore(agent.project_root)
        self.workflow_state = WorkflowStateController(
            self.context,
            dod_store=self.dod_store,
        )
        self.workflow_lanes = WorkflowLaneRunner(
            self.context,
            artifact_store=self.artifact_store,
            dod_store=self.dod_store,
            workflow_policy=self.workflow_policy,
        )
        self.workflow_recovery = WorkflowRecoveryController(
            self.context,
            artifact_invalidation=self.artifact_invalidation,
            workflow_policy=self.workflow_policy,
            workflow_signals=self.workflow_signals,
            workflow_lanes=self.workflow_lanes,
            set_workflow_mode=self.workflow_state.set_workflow_mode,
            append_timeline=self.workflow_state.append_timeline_from_decision,
            append_execute_bridge=self.workflow_state.maybe_append_execute_bridge,
        )
        self.repairer = ResponseRepairer(self.context)
        self.completion_policy = CompletionPolicy(self.context)
        self.phase_tracker = TurnPhaseTracker(self.context, self.tracer)
        self.finalizer = TurnFinalizer(
            agent,
            self.tracer,
            self.dod_store,
            self.workflow_state.set_workflow_mode,
        )
        self.turn_completion = TurnCompletionController(
            self.context,
            repairer=self.repairer,
            completion_policy=self.completion_policy,
            finalizer=self.finalizer,
            phase_tracker=self.phase_tracker,
        )
        self.response_router = AssistantResponseRouter(
            self.context,
            tracer=self.tracer,
            phase_tracker=self.phase_tracker,
            tool_batches=ToolBatchRunner(self.context, self.dod_store),
            turn_completion=self.turn_completion,
        )
        self.turn_iteration = TurnIterationController(
            self.context,
            phase_tracker=self.phase_tracker,
            turn_requester=AssistantTurnRequester(self.context, self.tracer),
            repairer=self.repairer,
            response_router=self.response_router,
        )
        self.turn_preparation = TurnPreparationController(
            self.context,
            tracer=self.tracer,
            phase_tracker=self.phase_tracker,
            dod_store=self.dod_store,
            workflow_policy=self.workflow_policy,
            workflow_signals=self.workflow_signals,
            workflow_lanes=self.workflow_lanes,
            finalizer=self.finalizer,
            set_workflow_mode=self.workflow_state.set_workflow_mode,
            append_timeline=self.workflow_state.append_timeline_from_decision,
            append_execute_bridge=self.workflow_state.maybe_append_execute_bridge,
        )
        self.turn_preamble = TurnPreludeController(
            self.context,
            tracer=self.tracer,
            workflow_recovery=self.workflow_recovery,
        )
        self.turn_loop = TurnLoopController(
            self.context,
            turn_preamble=self.turn_preamble,
            turn_iteration=self.turn_iteration,
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

        prepared_turn = await self.turn_preparation.prepare(
            task=task,
            emit=emit,
            requested_mode=requested_mode,
            original_task=original_task,
            on_user_question=on_user_question,
        )
        self.context.capability_profile = self.agent.capability_profile
        self.context.workflow_mode = self.agent.workflow_mode
        self.context.prompt_format = self.agent.prompt_format
        self.context.prompt_sections = list(self.agent.prompt_sections)
        self.executor = prepared_turn.executor
        summary = prepared_turn.summary
        dod = prepared_turn.definition_of_done
        task = prepared_turn.task
        effective_task = prepared_turn.effective_task
        effective_max_tokens = prepared_turn.effective_max_tokens
        rollback_plan = prepared_turn.rollback_plan

        assert self.executor is not None
        loop_exit = await self.turn_loop.run_loop(
            task=task,
            effective_task=effective_task,
            original_task=original_task,
            effective_max_tokens=effective_max_tokens,
            dod=dod,
            emit=emit,
            summary=summary,
            executor=self.executor,
            rollback_plan=rollback_plan,
            on_confirmation=on_confirmation,
            on_user_question=on_user_question,
            emit_confirmation=self._emit_confirmation(emit),
        )
        return await self._finalize_turn(
            summary,
            emit,
            reason_code=loop_exit.reason_code,
            reason_summary=loop_exit.reason_summary,
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
