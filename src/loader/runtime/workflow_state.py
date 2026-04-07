"""Workflow-state coordination for conversation runtime turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from ..llm.base import Message, Role
from .dod import DefinitionOfDone, DefinitionOfDoneStore
from .events import AgentEvent, TurnSummary
from .workflow import (
    ModeDecision,
    WorkflowDecisionKind,
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
    build_execute_bridge,
)

EventSink = Callable[[AgentEvent], Awaitable[None]]


class WorkflowStateController:
    """Own workflow-mode state, timeline persistence, and execute-bridge prompts."""

    def __init__(
        self,
        agent,
        *,
        dod_store: DefinitionOfDoneStore,
    ) -> None:
        self.agent = agent
        self.dod_store = dod_store

    async def set_workflow_mode(
        self,
        decision: ModeDecision,
        *,
        dod: DefinitionOfDone,
        emit: EventSink,
        summary: TurnSummary,
    ) -> None:
        """Apply one workflow-mode decision across session, DoD, and summary state."""

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
        self.append_timeline_from_decision(
            decision,
            kind={
                WorkflowDecisionKind.HANDOFF: WorkflowTimelineEntryKind.HANDOFF,
                WorkflowDecisionKind.REENTRY: WorkflowTimelineEntryKind.REENTRY,
            }.get(decision.decision_kind, WorkflowTimelineEntryKind.ROUTE),
            summary=summary,
            artifact_paths=self._artifact_paths_for(dod),
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

    def append_timeline_from_decision(
        self,
        decision: ModeDecision,
        *,
        kind: WorkflowTimelineEntryKind,
        summary: TurnSummary | None = None,
        artifact_paths: list[str] | None = None,
    ) -> None:
        """Persist one workflow timeline entry derived from a routing decision."""

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

    def maybe_append_execute_bridge(self, dod: DefinitionOfDone) -> None:
        """Append one workflow bridge prompt before execute mode, if needed."""

        bridge = build_execute_bridge(
            Path(dod.clarify_brief) if dod.clarify_brief else None,
            Path(dod.implementation_plan) if dod.implementation_plan else None,
            Path(dod.verification_plan) if dod.verification_plan else None,
        )
        if bridge and not any(
            message.role == Role.USER and "[WORKFLOW BRIDGE]" in message.content
            for message in self.agent.session.messages[-4:]
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
    def _artifact_paths_for(dod: DefinitionOfDone) -> list[str]:
        return [
            path
            for path in (
                dod.clarify_brief,
                dod.implementation_plan,
                dod.verification_plan,
            )
            if path
        ]
