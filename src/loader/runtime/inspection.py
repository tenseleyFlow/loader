"""Runtime inspection helpers for doctor, status, and session surfaces."""

from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..config import get_default_model
from ..context.project import ProjectContext, detect_project
from ..runtime.capabilities import CapabilityProfile, resolve_capability_profile
from ..tools.base import ToolRegistry, create_default_registry
from .dod import DefinitionOfDone, DefinitionOfDoneStore, VerificationEvidence
from .explore_state import ExploreStateStore
from .permissions import (
    PermissionConfigStatus,
    PermissionDecision,
    PermissionMode,
    PermissionRule,
    build_permission_policy,
    load_permission_rules,
    permission_path_hint,
    summarize_permission_input,
)
from .prompt_history import PromptSnapshot
from .prompting import build_system_prompt_result
from .session import SessionSnapshot, SessionStore
from .verification_observations import (
    VerificationObservation,
    VerificationObservationStatus,
    describe_verification_attempt,
)
from .workflow_ledger import WorkflowLedger
from .workflow_policy import WorkflowTimelineEntry
from .workflow_timeline_read_model import (
    project_workflow_timeline,
)


class CheckStatus(StrEnum):
    """Health status for one doctor check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(slots=True)
class DoctorCheck:
    """One diagnostic check in the doctor report."""

    name: str
    status: CheckStatus
    message: str
    remediation: str


@dataclass(slots=True)
class ToolPermissionSummary:
    """Resolved permission requirement for one registered tool."""

    tool_name: str
    required_mode: str
    resolution: str

    @property
    def allowed_in_active_mode(self) -> bool:
        return self.resolution == PermissionDecision.ALLOW.value


@dataclass(slots=True)
class PermissionRuleSummary:
    """Normalized view of one permission rule."""

    disposition: str
    tool_name: str | None
    contains: str | None
    path_contains: str | None
    summary: str


@dataclass(slots=True)
class PermissionSnapshot:
    """Operator-facing snapshot of the active permission policy."""

    project_root: Path
    active_mode: str
    prompting_enabled: bool
    rule_counts: dict[str, int]
    rules_valid: bool
    rules_source: str
    rules_error: str | None
    normalized_rules: dict[str, list[PermissionRuleSummary]] = field(
        default_factory=dict
    )
    tool_permissions: list[ToolPermissionSummary] = field(default_factory=list)


@dataclass(slots=True)
class PermissionCheckResult:
    """Dry-run evaluation of one hypothetical tool request."""

    project_root: Path
    tool_name: str
    arguments: dict[str, Any]
    input_summary: str
    path_hint: str | None
    required_mode: str
    active_mode: str
    prompting_enabled: bool
    rules_source: str
    decision: str
    reason: str | None
    matched_rule: str | None
    matched_disposition: str | None


@dataclass(slots=True)
class DoctorReport:
    """Full diagnostic report for one workspace."""

    project_root: Path
    model: str
    backend: str
    permission_mode: str
    permission_prompting_enabled: bool
    permission_rule_counts: dict[str, int]
    permission_rules_valid: bool
    permission_rules_source: str
    permission_rules_error: str | None
    project_context: ProjectContext
    capability_profile: CapabilityProfile
    checks: list[DoctorCheck] = field(default_factory=list)
    tool_permissions: list[ToolPermissionSummary] = field(default_factory=list)

    @property
    def overall_status(self) -> CheckStatus:
        if any(check.status == CheckStatus.FAIL for check in self.checks):
            return CheckStatus.FAIL
        if any(check.status == CheckStatus.WARN for check in self.checks):
            return CheckStatus.WARN
        return CheckStatus.PASS


@dataclass(slots=True)
class VerificationSummary:
    """Compact view of one verification evidence item."""

    command: str
    status: str
    kind: str
    detail: str
    attempt: str = ""


@dataclass(slots=True)
class StatusSnapshot:
    """Current runtime status read from persisted state."""

    project_root: Path
    model: str
    capability_profile: CapabilityProfile
    active_session_id: str | None
    workflow_mode: str
    workflow_reason_code: str | None
    workflow_reason_summary: str | None
    workflow_decision_kind: str | None
    workflow_ambiguity_score: float | None
    workflow_complexity_score: float | None
    workflow_scheduled_next_mode: str | None
    active_turn_phase: str | None
    completion_decision_code: str | None
    completion_decision_summary: str | None
    latest_policy_summary: str | None
    last_turn_transition_summary: str | None
    last_turn_transition_kind: str | None
    last_turn_transition_reason_code: str | None
    permission_mode: str
    permission_prompting_enabled: bool
    permission_rule_counts: dict[str, int]
    permission_rules_valid: bool
    permission_rules_source: str
    prompt_format: str | None
    prompt_sections: list[str]
    current_task: str | None
    message_count: int
    active_dod_path: str | None
    dod_status: str | None
    dod_pending_items_count: int
    last_verification_result: str | None
    recent_verification: list[VerificationSummary]
    latest_policy_supporting_evidence: list[str] = field(default_factory=list)
    latest_policy_blocking_evidence: list[str] = field(default_factory=list)
    latest_policy_observed_verification: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    compaction_count: int = 0
    project_type: str = "unknown"
    explore_updated_at: str | None = None
    explore_turn_count: int = 0
    explore_message_count: int = 0
    explore_history_mode: str | None = None
    explore_last_query: str | None = None
    explore_last_response: str | None = None
    runtime_owner_type: str | None = None
    runtime_owner_path: str | None = None


@dataclass(slots=True)
class ExploreContinuitySnapshot:
    """Operator-facing view of persisted explore continuity."""

    project_root: Path
    exists: bool
    updated_at: str | None = None
    turn_count: int = 0
    message_count: int = 0
    model_name: str | None = None
    last_history_mode: str | None = None
    last_query: str | None = None
    last_response: str | None = None


@dataclass(slots=True)
class SessionSummary:
    """List view for one persisted session."""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    workflow_mode: str
    workflow_reason_code: str | None
    workflow_reason_summary: str | None
    workflow_decision_kind: str | None
    permission_mode: str
    permission_prompting_enabled: bool
    permission_rule_counts: dict[str, int]
    permission_rules_source: str | None
    prompt_format: str | None
    active_turn_phase: str | None
    completion_decision_code: str | None
    completion_decision_summary: str | None
    last_turn_transition_summary: str | None
    current_task: str | None
    active_dod_path: str | None
    dod_status: str | None
    is_current: bool = False
    runtime_owner_type: str | None = None
    runtime_owner_path: str | None = None


@dataclass(slots=True)
class SessionDetail:
    """Expanded detail view for one session."""

    snapshot: SessionSnapshot
    is_current: bool
    definition_of_done: DefinitionOfDone | None
    recent_verification: list[VerificationSummary] = field(default_factory=list)


@dataclass(slots=True)
class PromptPreview:
    """Operator-facing preview of the current prompt contract."""

    project_root: Path
    model: str
    capability_profile: CapabilityProfile
    active_session_id: str | None
    workflow_mode: str
    workflow_reason_code: str | None
    workflow_reason_summary: str | None
    workflow_decision_kind: str | None
    permission_mode: str
    current_task: str | None
    prompt_format: str
    section_names: list[str] = field(default_factory=list)
    prompt_sections: list[str] = field(default_factory=list)
    content: str = ""


@dataclass(slots=True)
class WorkflowTimelineSnapshot:
    """Operator-facing view of persisted workflow history."""

    project_root: Path
    session_id: str | None
    is_current: bool
    workflow_mode: str
    current_task: str | None
    total_entries: int = 0
    latest_policy_summary: str | None = None
    latest_policy_supporting_evidence: list[str] = field(default_factory=list)
    latest_policy_blocking_evidence: list[str] = field(default_factory=list)
    latest_policy_observed_verification: list[str] = field(default_factory=list)
    selected_mode: str | None = None
    selected_kind: str | None = None
    selected_accountability_only: bool = False
    entry_limit: int | None = None
    highlights: list[str] = field(default_factory=list)
    entries: list[WorkflowTimelineEntry] = field(default_factory=list)
    workflow_ledger: WorkflowLedger = field(default_factory=WorkflowLedger)
    runtime_owner_type: str | None = None
    runtime_owner_path: str | None = None


@dataclass(slots=True)
class PromptDiffSnapshot:
    """Operator-facing diff between persisted prompt contracts."""

    project_root: Path
    session_id: str | None
    current_task: str | None
    current: PromptSnapshot | None
    previous: PromptSnapshot | None
    highlights: list[str] = field(default_factory=list)
    unified_diff: str = ""


@dataclass(slots=True)
class ArtifactDiffEntry:
    """One persisted artifact diff entry."""

    kind: str
    current_path: Path
    previous_path: Path | None
    highlights: list[str] = field(default_factory=list)
    unified_diff: str = ""


@dataclass(slots=True)
class WorkflowArtifactDiffSnapshot:
    """Operator-facing diff across persisted workflow artifacts."""

    project_root: Path
    session_id: str | None
    current_task: str | None
    entries: list[ArtifactDiffEntry] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)


def capability_summary(profile: CapabilityProfile) -> str:
    """Render a short human-readable capability summary."""

    tool_mode = "native tools" if profile.supports_native_tools else "ReAct tools"
    return (
        f"{profile.model_name}: {tool_mode}, "
        f"context {profile.context_window}, "
        f"verify {profile.verification_strictness}"
    )


async def collect_doctor_report(
    project_root: Path | str | None = None,
    *,
    model: str | None = None,
    backend: str = "ollama",
    permission_mode: PermissionMode | str = PermissionMode.WORKSPACE_WRITE,
    registry: ToolRegistry | None = None,
    backend_factory: Any | None = None,
) -> DoctorReport:
    """Collect a diagnostic report without entering the main runtime loop."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    resolved_model = model or get_default_model()
    resolved_permission_mode = _coerce_permission_mode(permission_mode)
    project_context = detect_project(resolved_root)
    registry = registry or create_default_registry(resolved_root)
    registry.configure_workspace_root(resolved_root)
    rule_status = load_permission_rules(resolved_root)

    capability_profile = resolve_capability_profile(resolved_model)
    checks: list[DoctorCheck] = []

    (
        backend_status,
        backend_message,
        backend_remediation,
        model_details,
    ) = await _backend_health_check(
        backend=backend,
        model=resolved_model,
        backend_factory=backend_factory,
    )
    checks.append(
        DoctorCheck(
            name="backend",
            status=backend_status,
            message=backend_message,
            remediation=backend_remediation,
        )
    )

    if model_details is not None:
        capability_profile = resolve_capability_profile(
            resolved_model,
            model_details=model_details,
        )
    checks.append(_capability_check(capability_profile))
    checks.append(_workspace_detection_check(project_context))
    checks.append(_write_access_check(resolved_root))
    checks.append(_test_build_check(project_context))
    checks.append(_state_health_check(resolved_root))
    checks.append(
        _permission_mode_check(
            registry,
            resolved_permission_mode,
            rule_status,
        )
    )

    tool_permissions = _tool_permission_summaries(
        registry,
        resolved_permission_mode,
        rule_status,
    )
    return DoctorReport(
        project_root=resolved_root,
        model=resolved_model,
        backend=backend,
        permission_mode=resolved_permission_mode.as_str(),
        permission_prompting_enabled=(
            resolved_permission_mode == PermissionMode.PROMPT or bool(rule_status.rules.ask)
        ),
        permission_rule_counts=rule_status.rules.counts,
        permission_rules_valid=rule_status.valid,
        permission_rules_source=str(rule_status.source_path),
        permission_rules_error=rule_status.error,
        project_context=project_context,
        capability_profile=capability_profile,
        checks=checks,
        tool_permissions=tool_permissions,
    )


def collect_status_snapshot(
    project_root: Path | str | None = None,
    *,
    model: str | None = None,
    permission_mode: PermissionMode | str = PermissionMode.WORKSPACE_WRITE,
) -> StatusSnapshot:
    """Read the current Loader runtime state from persisted artifacts."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    resolved_model = model or get_default_model()
    project_context = detect_project(resolved_root)
    session_store = SessionStore(resolved_root)
    snapshot = session_store.load_latest()
    capability_profile = resolve_capability_profile(resolved_model)
    default_permission_mode = _coerce_permission_mode(permission_mode).as_str()
    rule_status = load_permission_rules(resolved_root)
    explore_continuity = collect_explore_continuity_snapshot(resolved_root)

    if snapshot is None:
        return StatusSnapshot(
            project_root=resolved_root,
            model=resolved_model,
            capability_profile=capability_profile,
            active_session_id=None,
            runtime_owner_type=None,
            runtime_owner_path=None,
            workflow_mode="execute",
            workflow_reason_code=None,
            workflow_reason_summary=None,
            workflow_decision_kind=None,
            workflow_ambiguity_score=None,
            workflow_complexity_score=None,
            workflow_scheduled_next_mode=None,
            active_turn_phase=None,
            completion_decision_code=None,
            completion_decision_summary=None,
            latest_policy_summary=None,
            latest_policy_supporting_evidence=[],
            latest_policy_blocking_evidence=[],
            latest_policy_observed_verification=[],
            last_turn_transition_summary=None,
            last_turn_transition_kind=None,
            last_turn_transition_reason_code=None,
            permission_mode=default_permission_mode,
            permission_prompting_enabled=(
                _coerce_permission_mode(permission_mode) == PermissionMode.PROMPT
                or bool(rule_status.rules.ask)
            ),
            permission_rule_counts=rule_status.rules.counts,
            permission_rules_valid=rule_status.valid,
            permission_rules_source=str(rule_status.source_path),
            prompt_format=(
                "native" if capability_profile.supports_native_tools else "react"
            ),
            prompt_sections=[],
            current_task=None,
            message_count=0,
            active_dod_path=None,
            dod_status=None,
            dod_pending_items_count=0,
            last_verification_result=None,
            recent_verification=[],
            usage={},
            compaction_count=0,
            project_type=project_context.project_type,
            explore_updated_at=explore_continuity.updated_at,
            explore_turn_count=explore_continuity.turn_count,
            explore_message_count=explore_continuity.message_count,
            explore_history_mode=explore_continuity.last_history_mode,
            explore_last_query=explore_continuity.last_query,
            explore_last_response=explore_continuity.last_response,
        )

    dod = _load_dod(snapshot.active_dod_path, project_root=resolved_root)
    has_persisted_policy = snapshot.permission_rules_source is not None
    permission_rule_counts = (
        dict(snapshot.permission_rule_counts)
        if has_persisted_policy
        else rule_status.rules.counts
    )
    permission_prompting_enabled = (
        snapshot.permission_prompting_enabled
        if has_persisted_policy
        else (
            (snapshot.permission_mode or default_permission_mode) == "prompt"
            or bool(rule_status.rules.ask)
        )
    )
    projection = project_workflow_timeline(snapshot.workflow_timeline)
    recent_verification = _recent_verification_summaries(
        timeline=snapshot.workflow_timeline,
        evidence=dod.evidence if dod else [],
    )
    return StatusSnapshot(
        project_root=resolved_root,
        model=resolved_model,
        capability_profile=capability_profile,
        active_session_id=snapshot.session_id,
        runtime_owner_type=snapshot.runtime_owner_type,
        runtime_owner_path=snapshot.runtime_owner_path,
        workflow_mode=snapshot.workflow_mode,
        workflow_reason_code=snapshot.workflow_reason_code,
        workflow_reason_summary=snapshot.workflow_reason_summary,
        workflow_decision_kind=snapshot.workflow_decision_kind,
        workflow_ambiguity_score=snapshot.workflow_ambiguity_score,
        workflow_complexity_score=snapshot.workflow_complexity_score,
        workflow_scheduled_next_mode=snapshot.workflow_scheduled_next_mode,
        active_turn_phase=snapshot.active_turn_phase,
        completion_decision_code=snapshot.last_completion_decision_code,
        completion_decision_summary=snapshot.last_completion_decision_summary,
        latest_policy_summary=projection.latest_policy_summary,
        latest_policy_supporting_evidence=(
            list(projection.latest_policy_evidence.supporting)
            if projection.latest_policy_evidence is not None
            else []
        ),
        latest_policy_blocking_evidence=(
            list(projection.latest_policy_evidence.blocking)
            if projection.latest_policy_evidence is not None
            else []
        ),
        latest_policy_observed_verification=list(
            projection.latest_policy_observed_verification
        ),
        last_turn_transition_summary=snapshot.last_turn_transition_summary,
        last_turn_transition_kind=snapshot.last_turn_transition_kind,
        last_turn_transition_reason_code=snapshot.last_turn_transition_reason_code,
        permission_mode=snapshot.permission_mode or default_permission_mode,
        permission_prompting_enabled=permission_prompting_enabled,
        permission_rule_counts=permission_rule_counts,
        permission_rules_valid=rule_status.valid,
        permission_rules_source=(
            snapshot.permission_rules_source or str(rule_status.source_path)
        ),
        prompt_format=snapshot.prompt_format,
        prompt_sections=list(snapshot.prompt_sections),
        current_task=snapshot.current_task,
        message_count=len(snapshot.messages),
        active_dod_path=snapshot.active_dod_path,
        dod_status=dod.status if dod else None,
        dod_pending_items_count=len(dod.pending_items) if dod else 0,
        last_verification_result=_last_verification_result(
            dod=dod,
            recent_verification=recent_verification,
        ),
        recent_verification=recent_verification,
        usage=dict(snapshot.usage),
        compaction_count=(snapshot.compaction.count if snapshot.compaction else 0),
        project_type=project_context.project_type,
        explore_updated_at=explore_continuity.updated_at,
        explore_turn_count=explore_continuity.turn_count,
        explore_message_count=explore_continuity.message_count,
        explore_history_mode=explore_continuity.last_history_mode,
        explore_last_query=explore_continuity.last_query,
        explore_last_response=explore_continuity.last_response,
    )


def collect_explore_continuity_snapshot(
    project_root: Path | str | None = None,
) -> ExploreContinuitySnapshot:
    """Return the current persisted explore continuity snapshot, if any."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    snapshot = ExploreStateStore(resolved_root).load()
    if snapshot is None:
        return ExploreContinuitySnapshot(
            project_root=resolved_root,
            exists=False,
        )
    return ExploreContinuitySnapshot(
        project_root=resolved_root,
        exists=True,
        updated_at=snapshot.updated_at,
        turn_count=snapshot.turn_count,
        message_count=len(snapshot.messages),
        model_name=snapshot.model_name,
        last_history_mode=snapshot.last_history_mode,
        last_query=snapshot.last_query,
        last_response=snapshot.last_response,
    )


def reset_explore_continuity(project_root: Path | str | None = None) -> bool:
    """Clear persisted explore continuity and report whether state existed."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    store = ExploreStateStore(resolved_root)
    had_state = store.load() is not None
    store.clear()
    return had_state


def list_session_summaries(project_root: Path | str | None = None) -> list[SessionSummary]:
    """List all persisted sessions in newest-first order."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    store = SessionStore(resolved_root)
    current_session_id = _current_session_id(store)
    if not store.sessions_root.exists():
        return []

    entries: list[SessionSummary] = []
    for path in sorted(_session_paths(store.sessions_root), reverse=True):
        snapshot = SessionSnapshot.from_dict(json.loads(path.read_text()))
        dod = _load_dod(snapshot.active_dod_path, project_root=resolved_root)
        entries.append(
            SessionSummary(
                session_id=snapshot.session_id,
                created_at=snapshot.created_at,
                updated_at=snapshot.updated_at,
                runtime_owner_type=snapshot.runtime_owner_type,
                runtime_owner_path=snapshot.runtime_owner_path,
                message_count=len(snapshot.messages),
                workflow_mode=snapshot.workflow_mode,
                workflow_reason_code=snapshot.workflow_reason_code,
                workflow_reason_summary=snapshot.workflow_reason_summary,
                workflow_decision_kind=snapshot.workflow_decision_kind,
                permission_mode=snapshot.permission_mode,
                permission_prompting_enabled=(
                    snapshot.permission_prompting_enabled
                    or snapshot.permission_mode == "prompt"
                ),
                permission_rule_counts=dict(snapshot.permission_rule_counts),
                permission_rules_source=snapshot.permission_rules_source,
                prompt_format=snapshot.prompt_format,
                active_turn_phase=snapshot.active_turn_phase,
                completion_decision_code=snapshot.last_completion_decision_code,
                completion_decision_summary=snapshot.last_completion_decision_summary,
                last_turn_transition_summary=snapshot.last_turn_transition_summary,
                current_task=snapshot.current_task,
                active_dod_path=snapshot.active_dod_path,
                dod_status=dod.status if dod else None,
                is_current=snapshot.session_id == current_session_id,
            )
        )
    entries.sort(key=lambda entry: entry.updated_at, reverse=True)
    return entries


def load_session_detail(
    session_id: str,
    *,
    project_root: Path | str | None = None,
) -> SessionDetail:
    """Load one persisted session and its active DoD, if any."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    store = SessionStore(resolved_root)
    snapshot = store.load(session_id)
    current_session_id = _current_session_id(store)
    dod = _load_dod(snapshot.active_dod_path, project_root=resolved_root)
    return SessionDetail(
        snapshot=snapshot,
        is_current=snapshot.session_id == current_session_id,
        definition_of_done=dod,
        recent_verification=_recent_verification_summaries(
            timeline=snapshot.workflow_timeline,
            evidence=dod.evidence if dod else [],
        ),
    )


def collect_prompt_preview(
    project_root: Path | str | None = None,
    *,
    model: str | None = None,
    workflow_mode: str | None = None,
    permission_mode: PermissionMode | str | None = None,
    current_task: str | None = None,
    force_react: bool = False,
    registry: ToolRegistry | None = None,
) -> PromptPreview:
    """Render the current prompt contract without invoking the backend."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    resolved_model = model or get_default_model()
    capability_profile = resolve_capability_profile(resolved_model)
    project_context = detect_project(resolved_root)
    snapshot = SessionStore(resolved_root).load_latest()

    effective_workflow_mode = workflow_mode or (
        snapshot.workflow_mode if snapshot is not None else "execute"
    )
    effective_permission_mode = (
        _coerce_permission_mode(permission_mode).as_str()
        if permission_mode is not None
        else (
            snapshot.permission_mode
            if snapshot is not None
            else PermissionMode.WORKSPACE_WRITE.as_str()
        )
    )
    effective_task = current_task or (snapshot.current_task if snapshot is not None else None)

    registry = registry or create_default_registry(resolved_root)
    registry.configure_workspace_root(resolved_root)
    prompt_result = build_system_prompt_result(
        tools=registry.get_schemas(),
        use_react=force_react or not capability_profile.supports_native_tools,
        project_context=project_context,
        workflow_mode=effective_workflow_mode,
        permission_mode=effective_permission_mode,
        cwd=resolved_root,
        current_task=effective_task,
    )

    return PromptPreview(
        project_root=resolved_root,
        model=resolved_model,
        capability_profile=capability_profile,
        active_session_id=snapshot.session_id if snapshot is not None else None,
        workflow_mode=effective_workflow_mode,
        workflow_reason_code=(
            snapshot.workflow_reason_code
            if snapshot is not None and effective_workflow_mode == snapshot.workflow_mode
            else None
        ),
        workflow_reason_summary=(
            snapshot.workflow_reason_summary
            if snapshot is not None and effective_workflow_mode == snapshot.workflow_mode
            else None
        ),
        workflow_decision_kind=(
            snapshot.workflow_decision_kind
            if snapshot is not None and effective_workflow_mode == snapshot.workflow_mode
            else None
        ),
        permission_mode=effective_permission_mode,
        current_task=effective_task,
        prompt_format=prompt_result.prompt_format,
        section_names=list(prompt_result.section_names),
        prompt_sections=list(prompt_result.dynamic_section_names),
        content=prompt_result.content,
    )


def collect_prompt_diff(
    session_id: str | None = None,
    *,
    project_root: Path | str | None = None,
) -> PromptDiffSnapshot:
    """Load the latest persisted prompt-contract diff for one session."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    store = SessionStore(resolved_root)
    snapshot = store.load(session_id) if session_id else store.load_latest()
    if snapshot is None:
        return PromptDiffSnapshot(
            project_root=resolved_root,
            session_id=None,
            current_task=None,
            current=None,
            previous=None,
            highlights=[],
            unified_diff="",
        )

    current = snapshot.prompt_history[-1] if snapshot.prompt_history else None
    previous = _previous_prompt_snapshot(snapshot.prompt_history)
    highlights = _prompt_diff_highlights(previous, current)
    unified_diff = _unified_diff(
        previous.content if previous is not None else "",
        current.content if current is not None else "",
        from_label=_prompt_snapshot_label(previous, fallback="previous prompt"),
        to_label=_prompt_snapshot_label(current, fallback="current prompt"),
    )
    return PromptDiffSnapshot(
        project_root=resolved_root,
        session_id=snapshot.session_id,
        current_task=snapshot.current_task,
        current=current,
        previous=previous,
        highlights=highlights,
        unified_diff=unified_diff,
    )


def collect_workflow_artifact_diffs(
    session_id: str | None = None,
    *,
    project_root: Path | str | None = None,
) -> WorkflowArtifactDiffSnapshot:
    """Load persisted workflow-artifact diffs for the latest or named session."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    store = SessionStore(resolved_root)
    snapshot = store.load(session_id) if session_id else store.load_latest()
    if snapshot is None:
        return WorkflowArtifactDiffSnapshot(
            project_root=resolved_root,
            session_id=None,
            current_task=None,
            entries=[],
            highlights=[],
        )

    dod = _load_dod(snapshot.active_dod_path, project_root=resolved_root)
    if dod is None:
        return WorkflowArtifactDiffSnapshot(
            project_root=resolved_root,
            session_id=snapshot.session_id,
            current_task=snapshot.current_task,
            entries=[],
            highlights=[],
        )

    entries: list[ArtifactDiffEntry] = []
    for kind, path_str in (
        ("clarify_brief", dod.clarify_brief),
        ("implementation_plan", dod.implementation_plan),
        ("verification_plan", dod.verification_plan),
    ):
        if not path_str:
            continue
        current_path = Path(path_str)
        if not current_path.exists():
            continue
        previous_path = _previous_artifact_path(current_path)
        previous_text = previous_path.read_text() if previous_path and previous_path.exists() else ""
        current_text = current_path.read_text()
        entries.append(
            ArtifactDiffEntry(
                kind=kind,
                current_path=current_path,
                previous_path=previous_path,
                highlights=_artifact_diff_highlights(
                    kind=kind,
                    current_path=current_path,
                    previous_path=previous_path,
                    previous_text=previous_text,
                    current_text=current_text,
                ),
                unified_diff=_unified_diff(
                    previous_text,
                    current_text,
                    from_label=str(previous_path) if previous_path else f"previous {kind}",
                    to_label=str(current_path),
                ),
            )
        )

    highlights = [entry.highlights[0] for entry in entries if entry.highlights]
    return WorkflowArtifactDiffSnapshot(
        project_root=resolved_root,
        session_id=snapshot.session_id,
        current_task=snapshot.current_task,
        entries=entries,
        highlights=highlights,
    )


def collect_permission_snapshot(
    project_root: Path | str | None = None,
    *,
    permission_mode: PermissionMode | str = PermissionMode.WORKSPACE_WRITE,
    registry: ToolRegistry | None = None,
) -> PermissionSnapshot:
    """Collect an operator-facing snapshot of the active permission policy."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    resolved_permission_mode = _coerce_permission_mode(permission_mode)
    registry = registry or create_default_registry(resolved_root)
    registry.configure_workspace_root(resolved_root)
    rule_status = load_permission_rules(resolved_root)

    return PermissionSnapshot(
        project_root=resolved_root,
        active_mode=resolved_permission_mode.as_str(),
        prompting_enabled=(
            resolved_permission_mode == PermissionMode.PROMPT
            or bool(rule_status.rules.ask)
        ),
        rule_counts=rule_status.rules.counts,
        rules_valid=rule_status.valid,
        rules_source=str(rule_status.source_path),
        rules_error=rule_status.error,
        normalized_rules=_normalize_permission_rules(rule_status),
        tool_permissions=_tool_permission_summaries(
            registry,
            resolved_permission_mode,
            rule_status,
        ),
    )


def collect_workflow_timeline(
    session_id: str | None = None,
    *,
    project_root: Path | str | None = None,
    mode: str | None = None,
    kind: str | None = None,
    accountability_only: bool = False,
    limit: int | None = None,
) -> WorkflowTimelineSnapshot:
    """Load persisted workflow history for the latest or named session."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    store = SessionStore(resolved_root)
    snapshot = store.load(session_id) if session_id else store.load_latest()
    current_session_id = _current_session_id(store)
    if snapshot is None:
        return WorkflowTimelineSnapshot(
            project_root=resolved_root,
            session_id=None,
            is_current=False,
            runtime_owner_type=None,
            runtime_owner_path=None,
            workflow_mode="execute",
            current_task=None,
            total_entries=0,
            latest_policy_summary=None,
            latest_policy_supporting_evidence=[],
            latest_policy_blocking_evidence=[],
            latest_policy_observed_verification=[],
            selected_mode=mode,
            selected_kind=kind,
            selected_accountability_only=accountability_only,
            entry_limit=limit,
            highlights=[],
            entries=[],
            workflow_ledger=WorkflowLedger(),
        )

    projection = project_workflow_timeline(
        snapshot.workflow_timeline,
        workflow_ledger=snapshot.workflow_ledger,
        mode=mode,
        kind=kind,
        accountability_only=accountability_only,
        limit=limit,
    )

    return WorkflowTimelineSnapshot(
        project_root=resolved_root,
        session_id=snapshot.session_id,
        is_current=snapshot.session_id == current_session_id,
        runtime_owner_type=snapshot.runtime_owner_type,
        runtime_owner_path=snapshot.runtime_owner_path,
        workflow_mode=snapshot.workflow_mode,
        current_task=snapshot.current_task,
        total_entries=projection.total_entries,
        latest_policy_summary=projection.latest_policy_summary,
        latest_policy_supporting_evidence=(
            list(projection.latest_policy_evidence.supporting)
            if projection.latest_policy_evidence is not None
            else []
        ),
        latest_policy_blocking_evidence=(
            list(projection.latest_policy_evidence.blocking)
            if projection.latest_policy_evidence is not None
            else []
        ),
        latest_policy_observed_verification=list(
            projection.latest_policy_observed_verification
        ),
        selected_mode=mode,
        selected_kind=kind,
        selected_accountability_only=accountability_only,
        entry_limit=limit,
        highlights=list(projection.highlights),
        entries=list(projection.entries),
        workflow_ledger=snapshot.workflow_ledger.copy(),
    )


def dry_run_permission_check(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    project_root: Path | str | None = None,
    permission_mode: PermissionMode | str = PermissionMode.WORKSPACE_WRITE,
    registry: ToolRegistry | None = None,
) -> PermissionCheckResult:
    """Dry-run one hypothetical permission request against the active policy."""

    resolved_root = Path(project_root or Path.cwd()).expanduser().resolve()
    resolved_permission_mode = _coerce_permission_mode(permission_mode)
    registry = registry or create_default_registry(resolved_root)
    registry.configure_workspace_root(resolved_root)
    rule_status = load_permission_rules(resolved_root)
    if not rule_status.valid:
        raise ValueError(
            "Invalid permission policy configuration at "
            f"{rule_status.source_path}: {rule_status.error}"
        )

    tool = registry.get(tool_name)
    if tool is None:
        raise KeyError(tool_name)

    required_mode = tool.get_required_permission(**arguments)
    policy = _build_inspection_policy(
        registry,
        resolved_permission_mode,
        rule_status,
    )
    outcome = policy.authorize(
        tool_name,
        required_mode=required_mode,
        arguments=arguments,
    )
    request = outcome.request
    input_summary = (
        request.input_summary
        if request is not None
        else summarize_permission_input(tool_name, arguments)
    )
    path_hint = request.path_hint if request is not None else permission_path_hint(arguments)
    return PermissionCheckResult(
        project_root=resolved_root,
        tool_name=tool_name,
        arguments=dict(arguments),
        input_summary=input_summary,
        path_hint=path_hint,
        required_mode=required_mode.as_str(),
        active_mode=resolved_permission_mode.as_str(),
        prompting_enabled=policy.prompting_enabled,
        rules_source=str(rule_status.source_path),
        decision=outcome.decision.value,
        reason=outcome.reason,
        matched_rule=_rule_summary(outcome.matched_rule),
        matched_disposition=(
            outcome.matched_disposition.value
            if outcome.matched_disposition is not None
            else None
        ),
    )


def _previous_prompt_snapshot(
    history: list[PromptSnapshot],
) -> PromptSnapshot | None:
    if len(history) < 2:
        return None
    current = history[-1]
    for snapshot in reversed(history[:-1]):
        if not snapshot.matches_contract(current):
            return snapshot
    return history[-2]


def _prompt_snapshot_label(
    snapshot: PromptSnapshot | None,
    *,
    fallback: str,
) -> str:
    if snapshot is None:
        return fallback
    return (
        f"{snapshot.timestamp} "
        f"{snapshot.workflow_mode}/{snapshot.permission_mode}/{snapshot.prompt_format}"
    )


def _prompt_diff_highlights(
    previous: PromptSnapshot | None,
    current: PromptSnapshot | None,
) -> list[str]:
    if current is None:
        return []
    if previous is None:
        return ["No earlier prompt snapshot is available for comparison."]

    highlights: list[str] = []
    if previous.workflow_mode != current.workflow_mode:
        highlights.append(
            f"Workflow mode changed: {previous.workflow_mode} -> {current.workflow_mode}"
        )
    if previous.permission_mode != current.permission_mode:
        highlights.append(
            "Permission mode changed: "
            f"{previous.permission_mode} -> {current.permission_mode}"
        )
    if previous.prompt_format != current.prompt_format:
        highlights.append(
            f"Prompt format changed: {previous.prompt_format} -> {current.prompt_format}"
        )
    if previous.current_task != current.current_task:
        highlights.append("Task framing changed across prompt snapshots.")
    added_sections = [item for item in current.prompt_sections if item not in previous.prompt_sections]
    removed_sections = [item for item in previous.prompt_sections if item not in current.prompt_sections]
    if added_sections:
        highlights.append("Added sections: " + ", ".join(added_sections))
    if removed_sections:
        highlights.append("Removed sections: " + ", ".join(removed_sections))
    additions, removals = _line_change_counts(previous.content, current.content)
    highlights.append(f"Prompt body lines changed: +{additions} / -{removals}")
    return highlights


def _previous_artifact_path(current_path: Path) -> Path | None:
    name = current_path.name
    if name in {"implementation.md", "verification.md"}:
        parent = current_path.parent
        if "-" not in parent.name:
            return None
        slug = parent.name.split("-", maxsplit=1)[1]
        candidates = sorted(
            item
            for item in parent.parent.glob(f"*-{slug}")
            if item.is_dir() and (item / name).exists()
        )
        previous_dir = _previous_sorted_item(candidates, parent)
        if previous_dir is None:
            return None
        return previous_dir / name

    stem = current_path.stem
    if "-" not in stem:
        return None
    slug = stem.split("-", maxsplit=1)[1]
    candidates = sorted(
        item
        for item in current_path.parent.glob(f"*-{slug}{current_path.suffix}")
        if item.is_file()
    )
    return _previous_sorted_item(candidates, current_path)


def _previous_sorted_item(items: list[Any], current: Any) -> Any | None:
    previous: Any | None = None
    for item in items:
        if item == current:
            return previous
        previous = item
    return previous


def _artifact_diff_highlights(
    *,
    kind: str,
    current_path: Path,
    previous_path: Path | None,
    previous_text: str,
    current_text: str,
) -> list[str]:
    label = kind.replace("_", " ")
    if previous_path is None:
        return [f"{label}: no previous artifact version is available."]
    additions, removals = _line_change_counts(previous_text, current_text)
    if not additions and not removals:
        return [f"{label}: no content changes between persisted versions."]
    return [
        f"{label}: +{additions} / -{removals} lines vs {previous_path.name}",
        f"current={current_path.name}",
    ]


def _line_change_counts(previous_text: str, current_text: str) -> tuple[int, int]:
    additions = 0
    removals = 0
    diff_lines = difflib.unified_diff(
        previous_text.splitlines(),
        current_text.splitlines(),
        lineterm="",
    )
    for line in diff_lines:
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            removals += 1
    return additions, removals


def _unified_diff(
    previous_text: str,
    current_text: str,
    *,
    from_label: str,
    to_label: str,
) -> str:
    return "\n".join(
        difflib.unified_diff(
            previous_text.splitlines(),
            current_text.splitlines(),
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
        )
    )


def _coerce_permission_mode(value: PermissionMode | str) -> PermissionMode:
    if isinstance(value, PermissionMode):
        return value
    return PermissionMode.from_str(value)


async def _backend_health_check(
    *,
    backend: str,
    model: str,
    backend_factory: Any | None,
) -> tuple[CheckStatus, str, str, dict[str, Any] | None]:
    """Check backend reachability and whether the target model is available."""

    if backend != "ollama":
        return (
            CheckStatus.WARN,
            (
                f"Backend '{backend}' is not yet covered by doctor; "
                "using heuristic capability resolution."
            ),
            (
                "Use `--backend ollama` for a live connectivity check, "
                "or extend doctor for your backend."
            ),
            None,
        )

    if backend_factory is None:
        from ..llm.ollama import OllamaBackend

        backend_factory = OllamaBackend

    instance = backend_factory(model=model)
    try:
        available_models = []
        list_models = getattr(instance, "list_models", None)
        if callable(list_models):
            available_models = await list_models()
        health_check = getattr(instance, "health_check", None)
        backend_ok = bool(await health_check()) if callable(health_check) else False
        describe_model = getattr(instance, "describe_model", None)
        model_details = await describe_model() if callable(describe_model) else None

        available_names = [str(item.get("name", "")) for item in available_models]
        model_found = any(model in name or name in model for name in available_names if name)
        if backend_ok:
            return (
                CheckStatus.PASS,
                f"Ollama is reachable and model `{model}` is available.",
                "No action needed.",
                model_details if isinstance(model_details, dict) else None,
            )
        if available_names and not model_found:
            return (
                CheckStatus.FAIL,
                f"Ollama is reachable but model `{model}` is not pulled.",
                f"Run `ollama pull {model}` or choose an available model.",
                model_details if isinstance(model_details, dict) else None,
            )
        return (
            CheckStatus.FAIL,
            "Ollama is not reachable or returned no usable model metadata.",
            "Start Ollama with `ollama serve`, then verify the model is pulled.",
            model_details if isinstance(model_details, dict) else None,
        )
    except Exception as exc:
        return (
            CheckStatus.FAIL,
            f"Backend check failed: {exc}",
            "Make sure Ollama is running and reachable at the configured address.",
            None,
        )
    finally:
        close = getattr(instance, "close", None)
        if callable(close):
            await close()


def _capability_check(profile: CapabilityProfile) -> DoctorCheck:
    status = CheckStatus.WARN if any(
        "unknown model family" in note.lower() for note in profile.notes
    ) else CheckStatus.PASS
    message = capability_summary(profile)
    if profile.notes:
        message = f"{message}. {' '.join(profile.notes)}"
    remediation = (
        "Use a model with native tools for the strongest Loader behavior, "
        "or keep prompts concrete when using ReAct-only models."
    )
    return DoctorCheck(
        name="capabilities",
        status=status,
        message=message,
        remediation=remediation,
    )


def _workspace_detection_check(project_context: ProjectContext) -> DoctorCheck:
    if project_context.project_type == "unknown":
        return DoctorCheck(
            name="workspace",
            status=CheckStatus.WARN,
            message="Project type could not be detected from common markers.",
            remediation=(
                "Add a standard project marker such as `pyproject.toml`, "
                "`package.json`, `Cargo.toml`, or `go.mod`."
            ),
        )
    venv_hint = ""
    if project_context.has_venv:
        state = "active" if project_context.is_venv_active else "inactive"
        venv_hint = f" Venv: {project_context.venv_path} ({state})."
    return DoctorCheck(
        name="workspace",
        status=CheckStatus.PASS,
        message=(
            f"Detected a {project_context.project_type} project"
            f" using {project_context.package_manager}."
            f"{venv_hint}"
        ),
        remediation="No action needed.",
    )


def _write_access_check(project_root: Path) -> DoctorCheck:
    loader_root = project_root / ".loader"
    root_writable = os.access(project_root, os.W_OK)
    loader_writable = not loader_root.exists() or os.access(loader_root, os.W_OK)

    if root_writable and loader_writable:
        if loader_root.exists():
            message = "Workspace root and `.loader/` are writable."
        else:
            message = "Workspace root is writable; Loader can create `.loader/` on first run."
        return DoctorCheck(
            name="write_access",
            status=CheckStatus.PASS,
            message=message,
            remediation="No action needed.",
        )

    if not root_writable:
        return DoctorCheck(
            name="write_access",
            status=CheckStatus.FAIL,
            message="Workspace root is not writable by the current user.",
            remediation="Fix directory permissions or run Loader in a writable checkout.",
        )

    return DoctorCheck(
        name="write_access",
        status=CheckStatus.FAIL,
        message="`.loader/` exists but is not writable by the current user.",
        remediation=(
            "Fix permissions on `.loader/` so Loader can persist sessions "
            "and workflow state."
        ),
    )


def _test_build_check(project_context: ProjectContext) -> DoctorCheck:
    available = []
    if project_context.test_command:
        available.append(f"test `{project_context.test_command}`")
    if project_context.build_command:
        available.append(f"build `{project_context.build_command}`")

    if project_context.test_command and project_context.build_command:
        return DoctorCheck(
            name="commands",
            status=CheckStatus.PASS,
            message=f"Detected {' and '.join(available)} commands.",
            remediation="No action needed.",
        )

    if available:
        return DoctorCheck(
            name="commands",
            status=CheckStatus.WARN,
            message=f"Detected only {' and '.join(available)}.",
            remediation=(
                "Add standard project metadata so Loader can infer both test "
                "and build commands."
            ),
        )

    return DoctorCheck(
        name="commands",
        status=CheckStatus.WARN,
        message="No test or build command could be inferred from the workspace.",
        remediation=(
            "Add project markers and standard scripts so Loader can verify "
            "its own work reliably."
        ),
    )


def _state_health_check(project_root: Path) -> DoctorCheck:
    loader_root = project_root / ".loader"
    required_dirs = ["sessions", "state", "dod", "briefs", "plans"]
    if not loader_root.exists():
        return DoctorCheck(
            name="state",
            status=CheckStatus.WARN,
            message="`.loader/` has not been created yet, so no persisted state exists.",
            remediation=(
                "Run Loader once to create the state layout, or create "
                "`.loader/` manually if you need pre-seeded state."
            ),
        )

    missing = [name for name in required_dirs if not (loader_root / name).exists()]
    project_memory_path = loader_root / "project-memory.json"
    if project_memory_path.exists():
        try:
            parsed = json.loads(project_memory_path.read_text())
            if not isinstance(parsed, dict):
                raise ValueError("project-memory.json must contain an object")
        except Exception as exc:
            return DoctorCheck(
                name="state",
                status=CheckStatus.FAIL,
                message=f"Project memory is corrupted: {exc}",
                remediation=(
                    "Repair or remove `.loader/project-memory.json` so "
                    "Loader can parse durable memory again."
                ),
            )

    if missing:
        return DoctorCheck(
            name="state",
            status=CheckStatus.WARN,
            message=f"`.loader/` exists but is missing: {', '.join(missing)}.",
            remediation=(
                "Run the corresponding Loader workflows to materialize "
                "those directories, or create them if you are pre-seeding state."
            ),
        )

    return DoctorCheck(
        name="state",
        status=CheckStatus.PASS,
        message="State directories and project memory are present and parseable.",
        remediation="No action needed.",
    )


def _permission_mode_check(
    registry: ToolRegistry,
    permission_mode: PermissionMode,
    rule_status: PermissionConfigStatus,
) -> DoctorCheck:
    required_modes = [tool.required_permission.as_str() for tool in registry.list_tools()]
    counts = {
        "read-only": required_modes.count("read-only"),
        "workspace-write": required_modes.count("workspace-write"),
        "danger-full-access": required_modes.count("danger-full-access"),
    }
    if not rule_status.valid:
        return DoctorCheck(
            name="permissions",
            status=CheckStatus.FAIL,
            message=(
                f"Permission rules are invalid: {rule_status.error}. "
                f"Loader will fail closed until `{rule_status.source_path.name}` is fixed."
            ),
            remediation=(
                "Repair or remove `.loader/permission-rules.json`, then run "
                "`loader permissions show` to inspect the normalized policy."
            ),
        )

    prompt_message = (
        "prompting enabled"
        if permission_mode == PermissionMode.PROMPT or rule_status.rules.ask
        else "prompting disabled"
    )
    return DoctorCheck(
        name="permissions",
        status=CheckStatus.PASS,
        message=(
            f"Default permission mode is `{permission_mode.as_str()}`. "
            f"Policy rules: {rule_status.rules.counts['allow']} allow, "
            f"{rule_status.rules.counts['deny']} deny, "
            f"{rule_status.rules.counts['ask']} ask ({prompt_message}). "
            f"Tool requirements: {counts['read-only']} read-only, "
            f"{counts['workspace-write']} workspace-write, "
            f"{counts['danger-full-access']} danger-full-access."
        ),
        remediation=(
            "Use `loader permissions show` to inspect rules, or "
            "`loader permissions check <tool>` to dry-run one request."
        ),
    )


def _tool_permission_summaries(
    registry: ToolRegistry,
    permission_mode: PermissionMode,
    rule_status: PermissionConfigStatus,
) -> list[ToolPermissionSummary]:
    policy = _build_inspection_policy(
        registry,
        permission_mode,
        rule_status,
    )
    summaries = [
        ToolPermissionSummary(
            tool_name=tool.name,
            required_mode=tool.required_permission.as_str(),
            resolution=policy.authorize(
                tool.name,
                required_mode=tool.required_permission,
                arguments={},
            ).decision.value,
        )
        for tool in sorted(registry.list_tools(), key=lambda item: item.name)
    ]
    return summaries


def _build_inspection_policy(
    registry: ToolRegistry,
    permission_mode: PermissionMode,
    rule_status: PermissionConfigStatus,
):
    return build_permission_policy(
        active_mode=permission_mode,
        workspace_root=registry.workspace_root or Path.cwd(),
        tool_requirements=registry.get_tool_requirements(),
        rules=rule_status.rules if rule_status.valid else None,
    )


def _session_paths(sessions_root: Path) -> list[Path]:
    return [
        path
        for path in sessions_root.glob("*.json")
        if not path.stem.rsplit(".", 1)[-1].isdigit()
    ]


def _current_session_id(store: SessionStore) -> str | None:
    if not store.pointer_path.exists():
        return None
    try:
        payload = json.loads(store.pointer_path.read_text())
    except json.JSONDecodeError:
        return None
    session_id = payload.get("session_id")
    return session_id if isinstance(session_id, str) else None


def _load_dod(active_dod_path: str | None, *, project_root: Path) -> DefinitionOfDone | None:
    if not active_dod_path:
        return None
    path = Path(active_dod_path)
    if not path.exists():
        return None
    return DefinitionOfDoneStore(project_root).load(path)


def _recent_verification_summaries(
    *,
    timeline: list[WorkflowTimelineEntry],
    evidence: list[VerificationEvidence],
    limit: int = 3,
) -> list[VerificationSummary]:
    observed = _verification_summaries_from_timeline(timeline, limit=limit)
    if observed:
        return observed
    return _verification_summaries_from_evidence(evidence, limit=limit)


def _verification_summaries_from_timeline(
    timeline: list[WorkflowTimelineEntry],
    *,
    limit: int = 3,
) -> list[VerificationSummary]:
    summaries: list[VerificationSummary] = []
    seen: set[tuple[str, str, str, str]] = set()
    for entry in reversed(timeline):
        for observation in reversed(entry.verification_observations):
            summary = _verification_summary_from_observation(observation)
            key = (
                summary.command,
                summary.status,
                summary.kind,
                summary.detail,
            )
            if key in seen:
                continue
            seen.add(key)
            summaries.append(summary)
            if len(summaries) >= limit:
                return summaries
    return summaries


def _verification_summary_from_observation(
    observation: VerificationObservation,
) -> VerificationSummary:
    return VerificationSummary(
        command=observation.command or observation.summary,
        status=observation.status,
        kind=observation.kind or "runtime",
        detail=observation.detail or "",
        attempt=describe_verification_attempt(observation) or "",
    )


def _verification_summaries_from_evidence(
    evidence: list[VerificationEvidence],
    *,
    limit: int = 3,
) -> list[VerificationSummary]:
    summaries: list[VerificationSummary] = []
    for item in evidence[-limit:]:
        detail = _first_detail_line(item)
        summaries.append(
            VerificationSummary(
                command=item.command,
                status=(
                    VerificationObservationStatus.PASSED.value
                    if item.passed
                    else VerificationObservationStatus.FAILED.value
                ),
                kind=item.kind,
                detail=detail,
            )
        )
    return summaries


def _last_verification_result(
    *,
    dod: DefinitionOfDone | None,
    recent_verification: list[VerificationSummary],
) -> str | None:
    if dod is not None and dod.last_verification_result:
        return dod.last_verification_result
    if recent_verification:
        return recent_verification[0].status
    return None


def _first_detail_line(item: VerificationEvidence) -> str:
    for candidate in (item.stdout, item.stderr, item.output):
        for line in candidate.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
    return ""


def _normalize_permission_rules(
    rule_status: PermissionConfigStatus,
) -> dict[str, list[PermissionRuleSummary]]:
    return {
        "allow": [
            _permission_rule_summary("allow", rule)
            for rule in rule_status.rules.allow
        ],
        "deny": [
            _permission_rule_summary("deny", rule)
            for rule in rule_status.rules.deny
        ],
        "ask": [
            _permission_rule_summary("ask", rule)
            for rule in rule_status.rules.ask
        ],
    }


def _permission_rule_summary(
    disposition: str,
    rule: PermissionRule,
) -> PermissionRuleSummary:
    return PermissionRuleSummary(
        disposition=disposition,
        tool_name=rule.tool_name,
        contains=rule.contains,
        path_contains=rule.path_contains,
        summary=_rule_summary(rule) or "",
    )


def _rule_summary(rule: PermissionRule | None) -> str | None:
    if rule is None:
        return None
    parts: list[str] = []
    if rule.tool_name is not None:
        parts.append(f"tool={rule.tool_name}")
    if rule.contains is not None:
        parts.append(f"contains={rule.contains}")
    if rule.path_contains is not None:
        parts.append(f"path_contains={rule.path_contains}")
    return ", ".join(parts)
