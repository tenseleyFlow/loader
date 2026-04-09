"""Shared read models projected from the canonical workflow timeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .evidence_provenance import EvidenceProvenanceRollup, rollup_evidence_provenance
from .verification_observations import (
    VerificationObservation,
    describe_verification_attempt,
)
from .workflow_ledger import WorkflowLedger, workflow_ledger_highlights
from .workflow_policy import WorkflowTimelineEntry


@dataclass(slots=True)
class WorkflowTimelineProjection:
    """Projected views over persisted workflow timeline entries."""

    total_entries: int
    entries: list[WorkflowTimelineEntry] = field(default_factory=list)
    policy_entries: list[WorkflowTimelineEntry] = field(default_factory=list)
    latest_policy_entry: WorkflowTimelineEntry | None = None
    latest_policy_summary: str | None = None
    latest_policy_evidence: EvidenceProvenanceRollup | None = None
    latest_policy_observed_verification: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)


def project_workflow_timeline(
    entries: list[WorkflowTimelineEntry],
    *,
    workflow_ledger: WorkflowLedger | None = None,
    mode: str | None = None,
    kind: str | None = None,
    accountability_only: bool = False,
    limit: int | None = None,
) -> WorkflowTimelineProjection:
    """Project reusable filtered views from canonical workflow timeline entries."""

    source_entries = list(entries)
    policy_entries = filter_policy_accountability_entries(source_entries)
    filtered_entries = list(policy_entries if accountability_only else source_entries)
    if mode:
        filtered_entries = [entry for entry in filtered_entries if entry.mode == mode]
    if kind:
        filtered_entries = [entry for entry in filtered_entries if entry.kind == kind]

    highlights = workflow_timeline_highlights(filtered_entries)
    if workflow_ledger is not None:
        highlights.extend(workflow_ledger_highlights(workflow_ledger))
    if limit is not None:
        filtered_entries = filtered_entries[-limit:]

    latest_policy_entry = _latest_matching_entry(
        source_entries,
        is_policy_accountability_entry,
    )

    return WorkflowTimelineProjection(
        total_entries=len(source_entries),
        entries=filtered_entries,
        policy_entries=policy_entries,
        latest_policy_entry=latest_policy_entry,
        latest_policy_summary=latest_policy_accountability_summary(source_entries),
        latest_policy_evidence=(
            workflow_entry_evidence_rollup(latest_policy_entry)
            if latest_policy_entry is not None
            else None
        ),
        latest_policy_observed_verification=(
            summarize_observed_verification(
                latest_policy_entry.verification_observations,
            )
            if latest_policy_entry is not None
            else []
        ),
        highlights=list(dict.fromkeys(highlights)),
    )


def filter_policy_accountability_entries(
    entries: list[WorkflowTimelineEntry],
) -> list[WorkflowTimelineEntry]:
    """Return only unified policy-accountability entries from the workflow timeline."""

    return [entry for entry in entries if is_policy_accountability_entry(entry)]


def latest_policy_accountability_summary(
    entries: list[WorkflowTimelineEntry],
) -> str | None:
    """Return one compact explanation for the latest canonical policy event."""

    entry = _latest_matching_entry(entries, is_policy_accountability_entry)
    if entry is None:
        return None
    return workflow_entry_explanation(entry)


def workflow_timeline_highlights(
    entries: list[WorkflowTimelineEntry],
) -> list[str]:
    """Build the operator-facing highlight list for timeline answers."""

    highlights: list[str] = []

    clarify_entry = _latest_matching_entry(
        entries,
        lambda entry: entry.kind in {"clarify_continue", "clarify_exit"},
    )
    if clarify_entry is not None:
        prefix = (
            "Asked again:"
            if clarify_entry.kind == "clarify_continue"
            else "Clarify stopped:"
        )
        highlights.append(prefix + " " + workflow_entry_explanation(clarify_entry))

    recovery_entry = _latest_matching_entry(
        entries,
        lambda entry: entry.kind in {"reentry", "plan_refresh"}
        or "replan" in entry.reason_code
        or "refresh" in entry.reason_code,
    )
    if recovery_entry is not None:
        highlights.append(
            "Recovered workflow: " + workflow_entry_explanation(recovery_entry)
        )

    repair_entry = _latest_matching_entry(
        entries,
        lambda entry: entry.kind.startswith("repair_"),
    )
    if repair_entry is not None:
        prefix = "Repair failed:" if repair_entry.kind == "repair_fail" else "Repair path:"
        highlights.append(prefix + " " + workflow_entry_explanation(repair_entry))

    completion_entry = _latest_matching_entry(
        entries,
        lambda entry: entry.kind
        in {
            "completion_check",
            "completion_continue",
            "completion_complete",
            "completion_finalize",
        },
    )
    if completion_entry is not None:
        highlights.append(
            "Completion decision: " + workflow_entry_explanation(completion_entry)
        )

    verify_entry = _latest_matching_entry(
        entries,
        lambda entry: entry.kind in {"verify_skip", "verify_observation"}
        or "verify_skip" in entry.reason_code,
    )
    if verify_entry is not None:
        if verify_entry.kind == "verify_skip":
            prefix = "Skipped verify:"
        elif verify_entry.policy_outcome == "planned":
            prefix = "Verify planned:"
        elif verify_entry.policy_outcome == "pending":
            prefix = "Verify pending:"
        elif verify_entry.policy_outcome == "stale":
            prefix = "Verify stale:"
        else:
            prefix = "Verify observed:"
        highlights.append(prefix + " " + workflow_entry_explanation(verify_entry))

    return list(dict.fromkeys(highlights))


def is_policy_accountability_entry(entry: WorkflowTimelineEntry) -> bool:
    """Return whether one workflow timeline entry is a policy-accountability event."""

    kind = entry.kind
    return kind.startswith(("completion_", "repair_")) or kind in {
        "verify_skip",
        "verify_observation",
    }


def workflow_entry_explanation(entry: WorkflowTimelineEntry) -> str:
    """Render one compact explanation for policy and workflow timeline surfaces."""

    parts = [entry.summary]
    evidence_rollup = workflow_entry_evidence_rollup(entry)
    if entry.reason_code:
        parts.append(f"code={entry.reason_code}")
    if entry.clarify_stage:
        parts.append(f"stage={entry.clarify_stage}")
    if entry.clarify_pressure_kind:
        parts.append(f"pressure={entry.clarify_pressure_kind}")
    if entry.policy_stage:
        parts.append(f"policy-stage={entry.policy_stage}")
    if entry.policy_outcome:
        parts.append(f"policy-outcome={entry.policy_outcome}")
    if entry.missing_readiness_gates:
        parts.append("gates=" + ",".join(entry.missing_readiness_gates))
    if entry.unresolved_questions:
        parts.append(entry.unresolved_questions[0])
    if evidence_rollup.blocking:
        parts.append("needs=" + "; ".join(evidence_rollup.blocking[:2]))
    if evidence_rollup.supporting:
        parts.append("satisfied=" + "; ".join(evidence_rollup.supporting[:2]))
    if entry.evidence_summary and not evidence_rollup.blocking and not evidence_rollup.supporting:
        parts.append("evidence=" + "; ".join(entry.evidence_summary[:2]))
    if entry.evidence_provenance:
        parts.append(
            "provenance=" + format_evidence_provenance_brief(entry.evidence_provenance)
        )
    observed = summarize_observed_verification(entry.verification_observations)
    if observed:
        parts.append("observed=" + "; ".join(observed))
    if entry.signal_summary:
        parts.append("; ".join(entry.signal_summary[:2]))
    return " | ".join(part for part in parts if part)


def _latest_matching_entry(
    entries: list[WorkflowTimelineEntry],
    predicate,
) -> WorkflowTimelineEntry | None:
    for entry in reversed(entries):
        if predicate(entry):
            return entry
    return None


def format_evidence_provenance_brief(entries, *, max_entries: int = 2) -> str:
    """Render a compact operator-facing provenance summary."""

    parts: list[str] = []
    for entry in list(entries)[:max_entries]:
        source = f"@{entry.source}" if entry.source else ""
        subject = f"({entry.subject})" if entry.subject else ""
        parts.append(f"{entry.status}:{entry.category}{source}{subject}")
    return "; ".join(parts)


def summarize_observed_verification(
    entries: list[VerificationObservation],
    *,
    max_items: int = 2,
) -> list[str]:
    """Render observed verification facts for operator-facing inspection surfaces."""

    summaries: list[str] = []
    for entry in entries[:max_items]:
        summary = entry.summary.strip()
        attempt = describe_verification_attempt(entry)
        if entry.detail:
            detail = entry.detail.strip()
            if detail and detail not in summary:
                summary = f"{summary} [{detail}]"
        if attempt:
            if "[" in summary and summary.endswith("]"):
                summary = summary[:-1] + f"; {attempt}]"
            elif attempt not in summary:
                summary = f"{summary} [{attempt}]"
        if summary and summary not in summaries:
            summaries.append(summary)
    return summaries


def workflow_entry_evidence_rollup(
    entry: WorkflowTimelineEntry,
) -> EvidenceProvenanceRollup:
    """Return the grouped evidence view for one workflow timeline entry."""

    rollup = rollup_evidence_provenance(entry.evidence_provenance, max_items_per_status=2)
    if entry.evidence_summary and not (
        rollup.supporting or rollup.missing or rollup.contradicted or rollup.context
    ):
        rollup.context.extend(
            item
            for item in entry.evidence_summary[:2]
            if item not in rollup.context
        )
    return rollup
