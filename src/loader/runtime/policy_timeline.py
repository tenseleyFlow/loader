"""Helpers for unified policy-accountability timeline entries."""

from __future__ import annotations

from .context import RuntimeContext
from .events import TurnSummary
from .workflow_policy import (
    WorkflowDecisionKind,
    WorkflowTimelineEntry,
    WorkflowTimelineEntryKind,
)


def append_policy_timeline_entry(
    context: RuntimeContext,
    summary: TurnSummary,
    *,
    kind: WorkflowTimelineEntryKind,
    reason_code: str,
    reason_summary: str,
    policy_stage: str | None = None,
    policy_outcome: str | None = None,
    decision_kind: WorkflowDecisionKind | str | None = WorkflowDecisionKind.FORCED,
) -> WorkflowTimelineEntry:
    """Append one typed completion/repair accountability event."""

    entry = WorkflowTimelineEntry.accountability(
        kind=kind,
        mode=context.workflow_mode,
        reason_code=reason_code,
        summary=f"{_policy_prefix(kind)}: {reason_summary}",
        policy_stage=policy_stage,
        policy_outcome=policy_outcome,
        decision_kind=decision_kind,
        prompt_format=context.prompt_format,
        prompt_sections=context.prompt_sections,
    )
    context.session.append_workflow_timeline_entry(entry)
    summary.workflow_timeline = list(context.session.workflow_timeline)
    return entry


def completion_timeline_kind(
    *,
    stage: str,
    outcome: str,
) -> WorkflowTimelineEntryKind:
    """Map one completion trace decision onto a unified timeline entry kind."""

    if outcome == "continue":
        return WorkflowTimelineEntryKind.COMPLETION_CONTINUE
    if outcome == "finalize":
        return WorkflowTimelineEntryKind.COMPLETION_FINALIZE
    if stage == "continuation_check":
        return WorkflowTimelineEntryKind.COMPLETION_CHECK
    return WorkflowTimelineEntryKind.COMPLETION_COMPLETE


def _policy_prefix(kind: WorkflowTimelineEntryKind) -> str:
    if kind.value.startswith("completion_"):
        return "completion"
    if kind.value.startswith("repair_"):
        return "repair"
    return "policy"
