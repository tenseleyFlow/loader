"""Runtime inspection helpers for doctor, status, and session surfaces."""

from __future__ import annotations

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
from .permissions import (
    PermissionConfigStatus,
    PermissionDecision,
    PermissionMode,
    load_permission_rules,
)
from .session import SessionSnapshot, SessionStore


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
    passed: bool
    kind: str
    detail: str


@dataclass(slots=True)
class StatusSnapshot:
    """Current runtime status read from persisted state."""

    project_root: Path
    model: str
    capability_profile: CapabilityProfile
    active_session_id: str | None
    workflow_mode: str
    active_turn_phase: str | None
    permission_mode: str
    permission_prompting_enabled: bool
    permission_rule_counts: dict[str, int]
    permission_rules_valid: bool
    prompt_format: str | None
    prompt_sections: list[str]
    current_task: str | None
    message_count: int
    active_dod_path: str | None
    dod_status: str | None
    dod_pending_items_count: int
    last_verification_result: str | None
    recent_verification: list[VerificationSummary]
    usage: dict[str, int] = field(default_factory=dict)
    compaction_count: int = 0
    project_type: str = "unknown"


@dataclass(slots=True)
class SessionSummary:
    """List view for one persisted session."""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    workflow_mode: str
    permission_mode: str
    permission_prompting_enabled: bool
    permission_rule_counts: dict[str, int]
    prompt_format: str | None
    active_turn_phase: str | None
    current_task: str | None
    active_dod_path: str | None
    dod_status: str | None
    is_current: bool = False


@dataclass(slots=True)
class SessionDetail:
    """Expanded detail view for one session."""

    snapshot: SessionSnapshot
    is_current: bool
    definition_of_done: DefinitionOfDone | None


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

    if snapshot is None:
        return StatusSnapshot(
            project_root=resolved_root,
            model=resolved_model,
            capability_profile=capability_profile,
            active_session_id=None,
            workflow_mode="execute",
            active_turn_phase=None,
            permission_mode=default_permission_mode,
            permission_prompting_enabled=(
                _coerce_permission_mode(permission_mode) == PermissionMode.PROMPT
                or bool(rule_status.rules.ask)
            ),
            permission_rule_counts=rule_status.rules.counts,
            permission_rules_valid=rule_status.valid,
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
    return StatusSnapshot(
        project_root=resolved_root,
        model=resolved_model,
        capability_profile=capability_profile,
        active_session_id=snapshot.session_id,
        workflow_mode=snapshot.workflow_mode,
        active_turn_phase=snapshot.active_turn_phase,
        permission_mode=snapshot.permission_mode or default_permission_mode,
        permission_prompting_enabled=permission_prompting_enabled,
        permission_rule_counts=permission_rule_counts,
        permission_rules_valid=rule_status.valid,
        prompt_format=snapshot.prompt_format,
        prompt_sections=list(snapshot.prompt_sections),
        current_task=snapshot.current_task,
        message_count=len(snapshot.messages),
        active_dod_path=snapshot.active_dod_path,
        dod_status=dod.status if dod else None,
        dod_pending_items_count=len(dod.pending_items) if dod else 0,
        last_verification_result=(
            dod.last_verification_result if dod else None
        ),
        recent_verification=_verification_summaries(dod.evidence if dod else []),
        usage=dict(snapshot.usage),
        compaction_count=(snapshot.compaction.count if snapshot.compaction else 0),
        project_type=project_context.project_type,
    )


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
                message_count=len(snapshot.messages),
                workflow_mode=snapshot.workflow_mode,
                permission_mode=snapshot.permission_mode,
                permission_prompting_enabled=(
                    snapshot.permission_prompting_enabled
                    or snapshot.permission_mode == "prompt"
                ),
                permission_rule_counts=dict(snapshot.permission_rule_counts),
                prompt_format=snapshot.prompt_format,
                active_turn_phase=snapshot.active_turn_phase,
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
    return SessionDetail(
        snapshot=snapshot,
        is_current=snapshot.session_id == current_session_id,
        definition_of_done=_load_dod(snapshot.active_dod_path, project_root=resolved_root),
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
                "Repair or remove `.loader/permission-rules.json` so Loader can "
                "evaluate allow/deny/ask policy safely."
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
            "Use `--permission-mode` or `loader explore` to constrain "
            "the active runtime lane."
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
    from .permissions import build_permission_policy

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


def _verification_summaries(
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
                passed=item.passed,
                kind=item.kind,
                detail=detail,
            )
        )
    return summaries


def _first_detail_line(item: VerificationEvidence) -> str:
    for candidate in (item.stdout, item.stderr, item.output):
        for line in candidate.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
    return ""
