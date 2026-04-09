"""Turn bootstrap and workflow preparation for the conversation runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from .context import RuntimeContext
from .dod import DefinitionOfDone, DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutor
from .finalization import TurnFinalizer
from .hooks import build_default_tool_hooks
from .phases import TurnPhase, TurnPhaseTracker
from .rollback import RollbackPlan
from .task_classification import estimate_complexity, get_token_budget
from .tracing import RuntimeTracer
from .workflow import (
    ClarifyReview,
    ModeDecision,
    WorkflowDecisionKind,
    WorkflowMode,
    WorkflowPolicy,
    WorkflowSignalExtractor,
)
from .workflow_lanes import WorkflowLaneRunner

EventSink = Callable[[AgentEvent], Awaitable[None]]
UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]] | None
WorkflowModeSetter = Callable[..., Awaitable[None]]
TimelineAppender = Callable[..., None]
BridgeAppender = Callable[[DefinitionOfDone], None]


@dataclass(slots=True)
class PreparedTurn:
    """Runtime state produced before the main assistant/tool loop begins."""

    task: str
    effective_task: str
    summary: TurnSummary
    definition_of_done: DefinitionOfDone
    executor: ToolExecutor
    rollback_plan: RollbackPlan | None
    effective_max_tokens: int


class TurnPreparationController:
    """Owns capability refresh, turn bootstrap, and initial workflow routing."""

    def __init__(
        self,
        context: RuntimeContext,
        *,
        tracer: RuntimeTracer,
        phase_tracker: TurnPhaseTracker,
        dod_store: DefinitionOfDoneStore,
        workflow_policy: WorkflowPolicy,
        workflow_signals: WorkflowSignalExtractor,
        workflow_lanes: WorkflowLaneRunner,
        finalizer: TurnFinalizer,
        set_workflow_mode: WorkflowModeSetter,
        append_timeline: TimelineAppender,
        append_execute_bridge: BridgeAppender,
    ) -> None:
        self.context = context
        self.tracer = tracer
        self.phase_tracker = phase_tracker
        self.dod_store = dod_store
        self.workflow_policy = workflow_policy
        self.workflow_signals = workflow_signals
        self.workflow_lanes = workflow_lanes
        self.finalizer = finalizer
        self.set_workflow_mode = set_workflow_mode
        self.append_timeline = append_timeline
        self.append_execute_bridge = append_execute_bridge

    async def prepare(
        self,
        *,
        task: str,
        emit: EventSink,
        requested_mode: str | None,
        original_task: str | None,
        on_user_question: UserQuestionHandler,
    ) -> PreparedTurn:
        """Prepare runtime state before the iterative turn loop runs."""

        await self.phase_tracker.enter(
            TurnPhase.PREPARE,
            emit,
            detail="Preparing runtime state",
            reason_code="prepare_runtime",
        )
        await self._prepare_runtime_capabilities()

        effective_max_tokens = self._effective_max_tokens(task)
        executor, rollback_plan = self._build_executor()

        summary = TurnSummary(final_response="")
        summary.session_id = self.context.session.session_id
        effective_task = original_task or task
        dod = self.dod_store.create_or_resume(
            effective_task,
            retry_budget=self.context.config.verification_retry_budget,
        )
        summary.definition_of_done = dod

        self.context.session.clear_completion_trace(persist=False)
        self.context.session.update_runtime_state(
            active_dod_path=dod.storage_path,
            current_task=effective_task,
            workflow_mode=self.context.workflow_mode,
            permission_mode=self.context.active_permission_mode,
            permission_prompting_enabled=self.context.permission_policy.prompting_enabled,
            permission_rule_counts=self.context.active_permission_rule_counts,
            permission_rules_source=str(self.context.permission_config_status.source_path),
            last_completion_decision_code=None,
            last_completion_decision_summary=None,
        )
        await self.finalizer.emit_dod_status(emit, dod)

        prepared_task = await self._prepare_workflow(
            task=task,
            dod=dod,
            emit=emit,
            summary=summary,
            on_user_question=on_user_question,
            requested_mode=requested_mode,
            executor=executor,
        )
        return PreparedTurn(
            task=prepared_task,
            effective_task=effective_task,
            summary=summary,
            definition_of_done=dod,
            executor=executor,
            rollback_plan=rollback_plan,
            effective_max_tokens=effective_max_tokens,
        )

    def _effective_max_tokens(self, task: str) -> int:
        complexity = estimate_complexity(task)
        max_tokens, _ = get_token_budget(complexity)
        return min(self.context.config.max_tokens, max(max_tokens, 512))

    def _build_executor(self) -> tuple[ToolExecutor, RollbackPlan | None]:
        rollback_plan = (
            RollbackPlan() if self.context.config.reasoning.rollback else None
        )
        executor = ToolExecutor(
            self.context.registry,
            self.tracer,
            self.context.permission_policy,
            hooks=build_default_tool_hooks(
                action_tracker=self.context.safeguards.action_tracker,
                validator=self.context.safeguards.validator,
                registry=self.context.registry,
                rollback_plan=rollback_plan,
            ),
        )
        return executor, rollback_plan

    async def _prepare_workflow(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        on_user_question: UserQuestionHandler,
        requested_mode: str | None,
        executor: ToolExecutor,
    ) -> str:
        requested = WorkflowMode.from_str(requested_mode)
        decision = self.workflow_policy.route_from_signals(
            self.workflow_signals.extract_route_signals(
                task,
                requested_mode=requested.value if requested is not None else None,
                has_brief=self._artifact_exists(dod.clarify_brief),
                has_plan=self._artifact_exists(dod.implementation_plan)
                and self._artifact_exists(dod.verification_plan),
                timeline=self.context.session.workflow_timeline,
            )
        )
        await self.set_workflow_mode(
            decision,
            dod=dod,
            emit=emit,
            summary=summary,
        )

        clarify_review = ClarifyReview(
            should_continue=False,
            reason_code="clarify_not_needed",
            reason_summary="clarify was not needed for this route",
        )

        if decision.mode == WorkflowMode.CLARIFY:
            clarify_review = await self.workflow_lanes.run_clarify_mode(
                task=task,
                dod=dod,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                append_timeline=self.append_timeline,
            )
            decision = self.workflow_policy.route_from_signals(
                self.workflow_signals.extract_route_signals(
                    task,
                    has_brief=self._artifact_exists(dod.clarify_brief),
                    has_plan=self._artifact_exists(dod.implementation_plan)
                    and self._artifact_exists(dod.verification_plan),
                    allow_clarify=False,
                    unresolved_questions=clarify_review.unresolved_questions,
                    timeline=self.context.session.workflow_timeline,
                )
            )
            await self.set_workflow_mode(
                decision.with_context(
                    reason_code=f"post_clarify_{decision.reason_code}",
                    reason_summary=f"clarify handoff: {decision.reason_summary}",
                    decision_kind=WorkflowDecisionKind.HANDOFF,
                    unresolved_questions=clarify_review.unresolved_questions,
                ),
                dod=dod,
                emit=emit,
                summary=summary,
            )

        if decision.mode == WorkflowMode.PLAN:
            await self.workflow_lanes.run_plan_mode(
                task=task,
                dod=dod,
                emit=emit,
                refresh_reasons=clarify_review.unresolved_questions or None,
                executor=executor,
            )
            await self.set_workflow_mode(
                ModeDecision.transition(
                    WorkflowMode.EXECUTE,
                    reason_code="plan_artifacts_created",
                    reason_summary="plan artifacts created; switching to execute",
                    decision_kind=WorkflowDecisionKind.HANDOFF,
                ),
                dod=dod,
                emit=emit,
                summary=summary,
            )

        self.append_execute_bridge(dod)
        return task

    async def _prepare_runtime_capabilities(self) -> None:
        describe_model = getattr(self.context.backend, "describe_model", None)
        if callable(describe_model):
            await describe_model()

        previous_profile = self.context.capability_profile
        self.context.refresh_capability_profile()
        if self.context.capability_profile != previous_profile:
            self.tracer.record(
                "runtime.capabilities_refreshed",
                model_name=self.context.capability_profile.model_name,
                supports_native_tools=self.context.capability_profile.supports_native_tools,
                preferred_tool_call_format=(
                    self.context.capability_profile.preferred_tool_call_format
                ),
            )

    @staticmethod
    def _artifact_exists(path_str: str | None) -> bool:
        return bool(path_str and Path(path_str).exists())
