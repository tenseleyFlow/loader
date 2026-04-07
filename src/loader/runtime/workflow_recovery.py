"""Workflow recovery and reentry control for persisted artifacts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .artifact_invalidation import (
    ArtifactInvalidationAssessor,
    WorkflowRecoveryStrategy,
)
from .dod import DefinitionOfDone
from .events import AgentEvent, TurnSummary
from .workflow import (
    ArtifactFreshness,
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


class WorkflowRecoveryController:
    """Owns drift detection and workflow reentry decisions."""

    def __init__(
        self,
        agent: Any,
        *,
        artifact_invalidation: ArtifactInvalidationAssessor,
        workflow_policy: WorkflowPolicy,
        workflow_signals: WorkflowSignalExtractor,
        workflow_lanes: WorkflowLaneRunner,
        set_workflow_mode: WorkflowModeSetter,
        append_timeline: TimelineAppender,
        append_execute_bridge: BridgeAppender,
    ) -> None:
        self.agent = agent
        self.artifact_invalidation = artifact_invalidation
        self.workflow_policy = workflow_policy
        self.workflow_signals = workflow_signals
        self.workflow_lanes = workflow_lanes
        self.set_workflow_mode = set_workflow_mode
        self.append_timeline = append_timeline
        self.append_execute_bridge = append_execute_bridge

    async def maybe_refresh_plan_for_drift(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
        on_user_question: UserQuestionHandler,
        executor: Any,
    ) -> bool:
        """Refresh or reenter workflow when persisted artifacts drift."""

        if self.agent.workflow_mode != WorkflowMode.EXECUTE.value:
            return False
        if not (
            self._artifact_exists(dod.implementation_plan)
            and self._artifact_exists(dod.verification_plan)
        ):
            return False

        freshness = self.plan_freshness(dod)
        if not freshness.requires_refresh:
            return False

        strategy = WorkflowRecoveryStrategy(freshness.recovery_strategy)
        if strategy == WorkflowRecoveryStrategy.PLAN_REFRESH:
            return await self._run_plan_refresh_reentry(
                task=task,
                dod=dod,
                freshness=freshness,
                emit=emit,
                summary=summary,
                executor=executor,
            )
        if strategy == WorkflowRecoveryStrategy.CLARIFY_REENTRY:
            return await self._run_clarify_reentry_for_drift(
                task=task,
                dod=dod,
                freshness=freshness,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                executor=executor,
                force_plan_after_clarify=False,
            )
        if strategy == WorkflowRecoveryStrategy.FULL_REPLAN:
            return await self._run_clarify_reentry_for_drift(
                task=task,
                dod=dod,
                freshness=freshness,
                emit=emit,
                summary=summary,
                on_user_question=on_user_question,
                executor=executor,
                force_plan_after_clarify=True,
            )
        return False

    def plan_freshness(self, dod: DefinitionOfDone) -> ArtifactFreshness:
        """Assess whether the persisted workflow artifacts are stale."""

        return self.artifact_invalidation.assess(
            task_statement=dod.task_statement,
            clarify_text=self._artifact_text(dod.clarify_brief),
            implementation_text=self._artifact_text(dod.implementation_plan),
            verification_text=self._artifact_text(dod.verification_plan),
            acceptance_criteria=list(dod.acceptance_criteria),
            touched_files=list(dod.touched_files),
            last_verification_result=dod.last_verification_result,
        )

    async def _run_plan_refresh_reentry(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        freshness: ArtifactFreshness,
        emit: EventSink,
        summary: TurnSummary,
        executor: Any,
    ) -> bool:
        decision = self.workflow_policy.route_from_signals(
            self.workflow_signals.extract_route_signals(
                task,
                has_brief=self._artifact_exists(dod.clarify_brief),
                has_plan=True,
                allow_clarify=False,
                stale_plan=True,
                verification_pressure=bool(
                    dod.retry_count or dod.last_verification_result == "failed"
                ),
                unresolved_questions=freshness.reasons,
                timeline=self.agent.session.workflow_timeline,
            )
        )
        await self.set_workflow_mode(
            decision,
            dod=dod,
            emit=emit,
            summary=summary,
        )
        await self.workflow_lanes.run_plan_mode(
            task=task,
            dod=dod,
            emit=emit,
            refresh_reasons=freshness.reasons,
            executor=executor,
        )
        await self.set_workflow_mode(
            ModeDecision.transition(
                WorkflowMode.EXECUTE,
                reason_code="plan_refresh_completed",
                reason_summary="plan artifacts refreshed; returning to execute",
                decision_kind=WorkflowDecisionKind.HANDOFF,
                unresolved_questions=freshness.reasons,
            ),
            dod=dod,
            emit=emit,
            summary=summary,
        )
        self.append_execute_bridge(dod)
        return True

    async def _run_clarify_reentry_for_drift(
        self,
        *,
        task: str,
        dod: DefinitionOfDone,
        freshness: ArtifactFreshness,
        emit: EventSink,
        summary: TurnSummary,
        on_user_question: UserQuestionHandler,
        executor: Any,
        force_plan_after_clarify: bool,
    ) -> bool:
        clarify_reason_code = (
            "full_replan_requires_clarify"
            if force_plan_after_clarify
            else "clarify_reentry_required"
        )
        clarify_reason_summary = (
            "clarify and plan artifacts drifted; revisit requirements before replanning"
            if force_plan_after_clarify
            else "clarify artifacts drifted; revisit requirements before continuing"
        )
        await self.set_workflow_mode(
            ModeDecision.transition(
                WorkflowMode.CLARIFY,
                reason_code=clarify_reason_code,
                reason_summary=clarify_reason_summary,
                decision_kind=WorkflowDecisionKind.REENTRY,
                unresolved_questions=freshness.reasons,
            ),
            dod=dod,
            emit=emit,
            summary=summary,
        )
        clarify_review = await self.workflow_lanes.run_clarify_mode(
            task=task,
            dod=dod,
            emit=emit,
            summary=summary,
            on_user_question=on_user_question,
            append_timeline=self.append_timeline,
        )
        recovery_reasons = freshness.reasons + clarify_review.unresolved_questions

        if force_plan_after_clarify:
            await self.set_workflow_mode(
                ModeDecision.transition(
                    WorkflowMode.PLAN,
                    reason_code="full_replan_required",
                    reason_summary="clarify and plan artifacts drifted; rebuilding the plan",
                    decision_kind=WorkflowDecisionKind.REENTRY,
                    unresolved_questions=recovery_reasons,
                ),
                dod=dod,
                emit=emit,
                summary=summary,
            )
            await self.workflow_lanes.run_plan_mode(
                task=task,
                dod=dod,
                emit=emit,
                refresh_reasons=recovery_reasons,
                executor=executor,
            )
            await self.set_workflow_mode(
                ModeDecision.transition(
                    WorkflowMode.EXECUTE,
                    reason_code="full_replan_completed",
                    reason_summary="clarify and plan artifacts refreshed; returning to execute",
                    decision_kind=WorkflowDecisionKind.HANDOFF,
                    unresolved_questions=recovery_reasons,
                ),
                dod=dod,
                emit=emit,
                summary=summary,
            )
            self.append_execute_bridge(dod)
            return True

        decision = self.workflow_policy.route_from_signals(
            self.workflow_signals.extract_route_signals(
                task,
                has_brief=self._artifact_exists(dod.clarify_brief),
                has_plan=self._artifact_exists(dod.implementation_plan)
                and self._artifact_exists(dod.verification_plan),
                allow_clarify=False,
                unresolved_questions=recovery_reasons,
                timeline=self.agent.session.workflow_timeline,
            )
        )
        await self.set_workflow_mode(
            decision.with_context(
                reason_code=f"post_drift_{decision.reason_code}",
                reason_summary=f"clarify reentry handoff: {decision.reason_summary}",
                decision_kind=WorkflowDecisionKind.HANDOFF,
                unresolved_questions=recovery_reasons,
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
                refresh_reasons=recovery_reasons,
                executor=executor,
            )
            await self.set_workflow_mode(
                ModeDecision.transition(
                    WorkflowMode.EXECUTE,
                    reason_code="clarify_reentry_plan_created",
                    reason_summary="plan refreshed after clarify reentry; returning to execute",
                    decision_kind=WorkflowDecisionKind.HANDOFF,
                    unresolved_questions=recovery_reasons,
                ),
                dod=dod,
                emit=emit,
                summary=summary,
            )
        self.append_execute_bridge(dod)
        return True

    @staticmethod
    def _artifact_exists(path_str: str | None) -> bool:
        return bool(path_str and Path(path_str).exists())

    def _artifact_text(self, path_str: str | None) -> str | None:
        if not self._artifact_exists(path_str):
            return None
        assert path_str is not None
        return Path(path_str).read_text().strip()
