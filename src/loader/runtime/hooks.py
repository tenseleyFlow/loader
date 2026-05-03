"""Tool lifecycle hooks for Loader runtime execution."""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ..llm.base import ToolCall
from ..tools.base import Tool, ToolRegistry
from ..tools.base import ToolResult as RegistryToolResult
from .dod import (
    DefinitionOfDoneStore,
    all_planned_artifacts_exist,
    collect_missing_declared_html_output_files,
    collect_planned_artifact_targets,
    planned_artifact_target_satisfied,
)
from .memory import MemoryStore
from .path_display import display_runtime_path
from .permissions import PermissionOverride, PermissionPolicy
from .repair_focus import (
    extract_active_repair_context,
    normalize_repair_path,
    path_matches_allowed_paths,
    path_within_allowed_roots,
)
from .rollback import RollbackPlan, create_rollback_plan_for_action, is_destructive_tool
from .safeguard_services import (
    ActionTracker,
    PreActionValidator,
    extract_shell_text_rewrite_target,
)
from .workflow import infer_output_outline_label


class HookEvent(StrEnum):
    """Lifecycle hook events for one tool call."""

    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"


class HookDecision(StrEnum):
    """Terminal and non-terminal hook decisions."""

    CONTINUE = "continue"
    DENY = "deny"
    CANCEL = "cancel"
    FAIL = "fail"


@dataclass(slots=True)
class HookContext:
    """Context passed to hook implementations."""

    tool_call: ToolCall
    tool: Tool | None
    registry: ToolRegistry
    permission_policy: PermissionPolicy
    source: str
    skip_duplicate_check: bool = False
    record_action: bool = True
    result: RegistryToolResult | None = None
    output: str | None = None
    is_error: bool = False


@dataclass(slots=True)
class HookResult:
    """Result from one hook invocation."""

    decision: HookDecision = HookDecision.CONTINUE
    message: str | None = None
    injected_messages: list[str] = field(default_factory=list)
    permission_override: PermissionOverride | None = None
    permission_reason: str | None = None
    updated_arguments: dict[str, Any] | None = None
    output_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal_state: str | None = None


@dataclass(slots=True)
class HookRunSummary:
    """Aggregated result across all hooks for one lifecycle stage."""

    tool_call: ToolCall
    decision: HookDecision = HookDecision.CONTINUE
    message: str | None = None
    injected_messages: list[str] = field(default_factory=list)
    permission_override: PermissionOverride | None = None
    permission_reason: str | None = None
    output_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal_state: str | None = None


class ToolHook(Protocol):
    """Protocol for async tool lifecycle hooks."""

    async def pre_tool_use(self, context: HookContext) -> HookResult: ...

    async def post_tool_use(self, context: HookContext) -> HookResult: ...

    async def post_tool_use_failure(self, context: HookContext) -> HookResult: ...


class BaseToolHook:
    """Default no-op implementation for tool hooks."""

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        return HookResult()

    async def post_tool_use(self, context: HookContext) -> HookResult:
        return HookResult()

    async def post_tool_use_failure(self, context: HookContext) -> HookResult:
        return HookResult()


class FilePathAliasHook(BaseToolHook):
    """Normalize common file-path aliases before validation and execution."""

    _FILE_TOOLS = frozenset({"read", "write", "edit", "patch"})
    _ALIASES = ("filepath", "filePath", "file", "filename", "path")

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.tool_call.name not in self._FILE_TOOLS:
            return HookResult()

        arguments = context.tool_call.arguments
        file_path = str(arguments.get("file_path", "")).strip()
        if file_path:
            return HookResult()

        for alias in self._ALIASES:
            candidate = arguments.get(alias)
            if not str(candidate or "").strip():
                continue

            updated_arguments = dict(arguments)
            updated_arguments["file_path"] = candidate
            for cleanup_key in self._ALIASES:
                updated_arguments.pop(cleanup_key, None)
            return HookResult(updated_arguments=updated_arguments)

        return HookResult()


class SearchPathAliasHook(BaseToolHook):
    """Normalize common search-path aliases before validation and execution."""

    _SEARCH_TOOLS = frozenset({"glob", "grep"})
    _ALIASES = ("directory", "dir", "folder")

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.tool_call.name not in self._SEARCH_TOOLS:
            return HookResult()

        arguments = context.tool_call.arguments
        path = str(arguments.get("path", "")).strip()
        if path:
            return HookResult()

        for alias in self._ALIASES:
            candidate = arguments.get(alias)
            if not str(candidate or "").strip():
                continue

            updated_arguments = dict(arguments)
            updated_arguments["path"] = candidate
            for cleanup_key in self._ALIASES:
                updated_arguments.pop(cleanup_key, None)
            return HookResult(updated_arguments=updated_arguments)

        if context.tool_call.name == "glob":
            normalized_arguments = self._normalize_glob_pattern_path(arguments)
            if normalized_arguments is not None:
                return HookResult(updated_arguments=normalized_arguments)

        return HookResult()

    def _normalize_glob_pattern_path(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return None

        parent = ""
        basename = ""
        if pattern.startswith(("/", "~", "./", "../")):
            pattern_path = Path(pattern)
            parent = str(pattern_path.parent).strip()
            basename = pattern_path.name.strip()
        else:
            implicit = self._split_implicit_glob_parent(pattern)
            if implicit is None:
                return None
            parent, basename = implicit
        if not parent or not basename:
            return None
        if any(token in parent for token in ("*", "?", "[")):
            return None

        updated_arguments = dict(arguments)
        updated_arguments["path"] = parent
        updated_arguments["pattern"] = basename
        return updated_arguments

    def _split_implicit_glob_parent(self, pattern: str) -> tuple[str, str] | None:
        if "/" not in pattern:
            return None

        parts = [segment for segment in pattern.split("/") if segment]
        while parts and self._is_wildcard_segment(parts[0]):
            parts.pop(0)
        if len(parts) < 2:
            return None

        parent_parts = parts[:-1]
        basename = parts[-1].strip()
        if not basename or not parent_parts:
            return None
        if any(self._segment_contains_glob(segment) for segment in parent_parts):
            return None
        return "/".join(parent_parts), basename

    def _is_wildcard_segment(self, segment: str) -> bool:
        return bool(segment) and all(char in "*?[]" for char in segment)

    def _segment_contains_glob(self, segment: str) -> bool:
        return any(token in segment for token in ("*", "?", "["))


class RelativePathContextHook(BaseToolHook):
    """Recover relative file/search paths against recently-used external directories."""

    _FILE_TOOLS = frozenset({"read", "write", "edit", "patch"})
    _SEARCH_TOOLS = frozenset({"glob", "grep"})

    def __init__(self, action_tracker: ActionTracker, workspace_root: Path) -> None:
        self.action_tracker = action_tracker
        self.workspace_root = workspace_root.expanduser().resolve()

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        argument_key = self._argument_key(context.tool_call.name)
        if argument_key is None:
            return HookResult()

        arguments = context.tool_call.arguments
        raw_path = str(arguments.get(argument_key, "")).strip()
        if not raw_path:
            return HookResult()

        require_existing = context.tool_call.name in {"read", "glob", "grep", "edit", "patch"}
        resolved: str | None = None
        injected_messages: list[str] = []
        if raw_path.startswith("/"):
            resolved = self._resolve_workspace_mirror_path(
                raw_path,
                require_existing=require_existing,
            )
            if resolved is not None:
                injected_messages.append(
                    self._workspace_mirror_correction_message(raw_path, resolved)
                )
        elif not raw_path.startswith("~"):
            resolved = self._resolve_recent_context_path(
                raw_path,
                require_existing=require_existing,
                prefer_external_ancestor=context.tool_call.name in self._SEARCH_TOOLS,
            )
        if resolved is None:
            return HookResult()

        updated_arguments = dict(arguments)
        updated_arguments[argument_key] = resolved
        return HookResult(
            updated_arguments=updated_arguments,
            injected_messages=injected_messages,
        )

    def _argument_key(self, tool_name: str) -> str | None:
        if tool_name in self._FILE_TOOLS:
            return "file_path"
        if tool_name in self._SEARCH_TOOLS:
            return "path"
        return None

    def _resolve_recent_context_path(
        self,
        raw_path: str,
        *,
        require_existing: bool,
        prefer_external_ancestor: bool,
    ) -> str | None:
        workspace_candidate = (self.workspace_root / raw_path).expanduser()
        if workspace_candidate.exists():
            if prefer_external_ancestor:
                anchored = self._resolve_recent_context_ancestor(
                    raw_path,
                    require_existing=require_existing,
                )
                if anchored is not None:
                    return anchored
            return None

        for base_dir in self.action_tracker.recent_path_contexts():
            candidate = (Path(base_dir) / raw_path).expanduser()
            if require_existing:
                if candidate.exists():
                    return str(candidate)
                continue
            if candidate.exists() or candidate.parent.exists():
                return str(candidate)
        if prefer_external_ancestor:
            return self._resolve_recent_context_ancestor(
                raw_path,
                require_existing=require_existing,
            )
        return None

    def _resolve_recent_context_ancestor(
        self,
        raw_path: str,
        *,
        require_existing: bool,
    ) -> str | None:
        raw_parts = tuple(part for part in Path(raw_path).parts if part not in {"."})
        if not raw_parts:
            return None

        for base_dir in self.action_tracker.recent_path_contexts():
            base_path = Path(base_dir).expanduser()
            try:
                resolved_base = base_path.resolve(strict=False)
            except Exception:
                resolved_base = base_path
            if resolved_base == self.workspace_root:
                continue
            try:
                resolved_base.relative_to(self.workspace_root)
                continue
            except ValueError:
                pass

            matched = self._match_recent_context_ancestor(
                resolved_base,
                raw_parts,
            )
            if matched is None:
                continue
            if require_existing and not matched.exists():
                continue
            return str(matched)
        return None

    def _match_recent_context_ancestor(
        self,
        base_path: Path,
        raw_parts: tuple[str, ...],
    ) -> Path | None:
        candidates = [base_path, *base_path.parents]
        for candidate in candidates:
            if len(candidate.parts) < len(raw_parts):
                continue
            if candidate.parts[-len(raw_parts) :] == raw_parts:
                return candidate
        return None

    def _resolve_workspace_mirror_path(
        self,
        raw_path: str,
        *,
        require_existing: bool,
    ) -> str | None:
        candidate = Path(raw_path).expanduser()
        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            resolved = candidate

        try:
            relative = resolved.relative_to(self.workspace_root)
        except ValueError:
            return None
        if not relative.parts:
            return None

        anchor = relative.parts[0]
        for base_dir in self.action_tracker.recent_path_contexts():
            base_path = Path(base_dir).expanduser()
            try:
                resolved_base = base_path.resolve(strict=False)
            except Exception:
                resolved_base = base_path
            if resolved_base == self.workspace_root:
                continue
            try:
                resolved_base.relative_to(self.workspace_root)
                continue
            except ValueError:
                pass

            try:
                anchor_index = resolved_base.parts.index(anchor)
            except ValueError:
                continue
            if anchor_index <= 0:
                continue

            anchor_root = Path(*resolved_base.parts[: anchor_index + 1])
            remapped = Path(*resolved_base.parts[:anchor_index]).joinpath(*relative.parts)
            if remapped == resolved:
                continue
            if require_existing:
                if remapped.exists():
                    return str(remapped)
                continue
            if remapped.exists() or remapped.parent.exists() or anchor_root.exists():
                return str(remapped)
        return None

    def _workspace_mirror_correction_message(self, raw_path: str, resolved_path: str) -> str:
        raw_name = Path(str(raw_path)).name or str(raw_path)
        resolved_root = self._describe_anchor_root(resolved_path)
        return (
            "[Path anchor correction] A repo-local mirror path was remapped to the established "
            f"output root under `{resolved_root}`. Keep future file/search tool calls on that "
            f"external root and use `{raw_name}` there instead of re-anchoring work to the "
            "workspace checkout."
        )

    def _describe_anchor_root(self, path_value: str) -> str:
        resolved = Path(path_value).expanduser()
        try:
            candidate = resolved.resolve(strict=False)
        except Exception:
            candidate = resolved

        parts = candidate.parts
        if "Loader" in parts:
            loader_index = parts.index("Loader")
            return str(Path(*parts[: loader_index + 1]))
        return str(candidate.parent)


_OBSERVATION_TOOLS = frozenset({"read", "glob", "grep", "bash"})
_MUTATION_TOOLS = frozenset({"write", "edit", "patch", "bash"})
_READ_ONLY_BASH_PREFIXES = frozenset(
    {"ls", "pwd", "find", "stat", "cat", "head", "tail", "rg", "grep"}
)
_MUTATING_BASH_FRAGMENTS = (
    " >",
    ">>",
    "| tee",
    "touch ",
    "mkdir ",
    "rm ",
    "mv ",
    "cp ",
    "sed -i",
    "perl -pi",
    "git add",
    "git commit",
    "git apply",
)


def _extract_observation_paths(tool_call: ToolCall) -> list[str]:
    arguments = tool_call.arguments
    if tool_call.name == "read":
        file_path = str(arguments.get("file_path", "")).strip()
        return [file_path] if file_path else []

    if tool_call.name in {"glob", "grep"}:
        candidates: list[str] = []
        search_path = str(arguments.get("path", "")).strip()
        if search_path:
            anchored_path = _derive_search_anchor(search_path, str(arguments.get("pattern", "")).strip())
            candidates.append(anchored_path or search_path)
        pattern = str(arguments.get("pattern", "")).strip()
        if not search_path and pattern.startswith(("/", "~")):
            candidates.append(str(Path(pattern).expanduser().parent))
        return candidates

    command = str(arguments.get("command", "")).strip()
    if not _is_read_only_bash(command):
        return []
    return _extract_bash_paths(command)


def _is_read_only_bash(command: str) -> bool:
    normalized = " ".join(command.split())
    if not normalized:
        return False
    if extract_shell_text_rewrite_target(normalized) is not None:
        return False
    if any(fragment in normalized for fragment in _MUTATING_BASH_FRAGMENTS):
        return False
    try:
        argv = shlex.split(normalized)
    except ValueError:
        return False
    if not argv:
        return False
    return argv[0] in _READ_ONLY_BASH_PREFIXES


def _extract_bash_paths(command: str) -> list[str]:
    try:
        argv = shlex.split(command)
    except ValueError:
        return []
    observed: list[str] = []
    for token in argv[1:]:
        candidate = token.strip()
        if not candidate or candidate.startswith("-"):
            continue
        if candidate.startswith(("/", "~")):
            observed.append(candidate)
    return observed


def _derive_search_anchor(search_path: str, pattern: str) -> str:
    normalized_search_path = str(search_path or "").strip()
    normalized_pattern = str(pattern or "").strip()
    if not normalized_search_path or not normalized_pattern:
        return normalized_search_path

    literal_segments: list[str] = []
    for segment in normalized_pattern.split("/"):
        cleaned = segment.strip()
        if not cleaned or cleaned == ".":
            continue
        if any(token in cleaned for token in ("*", "?", "[")):
            continue
        literal_segments.append(cleaned)

    if not literal_segments:
        return normalized_search_path

    if "." in literal_segments[-1]:
        literal_segments = literal_segments[:-1]
    if not literal_segments:
        return normalized_search_path

    try:
        anchored = Path(normalized_search_path).expanduser().joinpath(*literal_segments)
    except (OSError, RuntimeError, ValueError):
        return normalized_search_path
    return str(anchored)


def _extract_mutation_paths(tool_call: ToolCall) -> list[str]:
    arguments = tool_call.arguments
    if tool_call.name in {"write", "edit", "patch"}:
        file_path = str(arguments.get("file_path", "")).strip()
        return [file_path] if file_path else []

    if tool_call.name != "bash":
        return []

    command = str(arguments.get("command", "")).strip()
    if not command or not _is_mutating_bash(command):
        return []
    target = extract_shell_text_rewrite_target(command)
    return [target] if target else []


def _is_mutating_bash(command: str) -> bool:
    normalized = " ".join(command.split())
    if not normalized:
        return False
    if extract_shell_text_rewrite_target(normalized) is not None:
        return True
    if any(fragment in normalized for fragment in _MUTATING_BASH_FRAGMENTS):
        return True
    try:
        argv = shlex.split(normalized)
    except ValueError:
        return False
    if not argv:
        return False
    return argv[0] in {"touch", "mkdir", "rm", "mv", "cp", "chmod", "chown"}


def _repair_declared_output_paths(repair: Any, *, project_root: Path) -> set[str]:
    declared_outputs: set[str] = set()
    for root in getattr(repair, "allowed_roots", ()) or ():
        normalized_root = normalize_repair_path(root)
        if not normalized_root:
            continue
        for path in collect_missing_declared_html_output_files(
            target=Path(normalized_root),
            project_root=project_root,
        ):
            declared_outputs.add(normalize_repair_path(str(path)))
    return declared_outputs


def _repair_uses_artifact_set_as_source_of_truth(repair: Any) -> bool:
    return any(
        "source of truth" in str(line).lower()
        for line in getattr(repair, "repair_lines", ())
    )


class ActiveRepairScopeHook(BaseToolHook):
    """Keep fix-mode observations anchored to the active artifact set."""

    _MAX_SOURCE_OF_TRUTH_OBSERVATIONS = 4

    def __init__(
        self,
        *,
        dod_store: DefinitionOfDoneStore,
        project_root: Path,
        session: Any,
    ) -> None:
        self.dod_store = dod_store
        self.project_root = project_root
        self.session = session
        self._source_of_truth_scope_key: tuple[str, ...] | None = None
        self._source_of_truth_observation_count = 0

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.tool_call.name not in _OBSERVATION_TOOLS:
            return HookResult()
        if context.source == "verification":
            return HookResult()

        repair = self._active_repair_context()
        if repair is None:
            return HookResult()

        observed_paths = _extract_observation_paths(context.tool_call)
        if not observed_paths:
            return HookResult()
        declared_output_paths = _repair_declared_output_paths(
            repair,
            project_root=self.project_root,
        )
        in_allowed_roots = bool(repair.allowed_roots) and all(
            path_within_allowed_roots(path, repair.allowed_roots) for path in observed_paths
        )
        source_of_truth_scope = (
            _repair_uses_artifact_set_as_source_of_truth(repair) and in_allowed_roots
        )
        if source_of_truth_scope:
            self._sync_source_of_truth_scope(repair.allowed_roots)
            if (
                self._source_of_truth_observation_count
                >= self._MAX_SOURCE_OF_TRUTH_OBSERVATIONS
            ):
                return HookResult(
                    decision=HookDecision.DENY,
                    message=(
                        "[Blocked - repair audit loop: the active repair artifact set has "
                        "already been inspected several times without a concrete mutation.] "
                        f"Suggestion: make one concrete edit, patch, or write to "
                        f"`{repair.artifact_path}` or create the next missing repair target "
                        "instead of more rereads."
                    ),
                    terminal_state="blocked",
                )
        if repair.allowed_paths:
            if all(path_matches_allowed_paths(path, repair.allowed_paths) for path in observed_paths):
                return HookResult()
            if declared_output_paths and all(
                normalize_repair_path(path) in declared_output_paths
                for path in observed_paths
            ):
                return HookResult()
            if source_of_truth_scope:
                return HookResult()
            if context.tool_call.name in {"glob", "grep", "bash"} and repair.allowed_roots:
                if all(path_within_allowed_roots(path, repair.allowed_roots) for path in observed_paths):
                    return HookResult()

            allowed_preview = ", ".join(f"`{path}`" for path in repair.allowed_paths[:3])
            if len(repair.allowed_paths) > 3:
                allowed_preview += ", ..."
            declared_preview = ", ".join(
                f"`{Path(path).name or path}`"
                for path in sorted(declared_output_paths)[:3]
            )
            if len(declared_output_paths) > 3:
                declared_preview += ", ..."
            suggestion_suffix = (
                f" Declared sibling outputs currently allowed inside this repair set include: {declared_preview}."
                if declared_preview
                else ""
            )
            return HookResult(
                decision=HookDecision.DENY,
                message=(
                    "[Blocked - active repair scope: verification already identified "
                    f"`{repair.artifact_path}` as the current repair target. "
                    "Stay on the concrete repair files until that repair passes.] "
                    "Suggestion: inspect or edit only "
                    f"{allowed_preview} and do not reopen unrelated reference materials."
                    f"{suggestion_suffix}"
                ),
                terminal_state="blocked",
            )

        if not repair.allowed_roots:
            return HookResult()
        if all(path_within_allowed_roots(path, repair.allowed_roots) for path in observed_paths):
            return HookResult()

        roots_preview = ", ".join(f"`{root}`" for root in repair.allowed_roots[:2])
        if len(repair.allowed_roots) > 2:
            roots_preview += ", ..."
        return HookResult(
            decision=HookDecision.DENY,
            message=(
                "[Blocked - active repair scope: verification already identified "
                f"`{repair.artifact_path}` as the current repair target. "
                "Stay inside the current artifact set until that repair passes.] "
                "Suggestion: inspect or edit files under "
                f"{roots_preview} and do not reopen unrelated reference materials."
            ),
            terminal_state="blocked",
        )

    async def post_tool_use(self, context: HookContext) -> HookResult:
        if context.source == "verification":
            return HookResult()
        if context.tool_call.name in _MUTATION_TOOLS:
            self._reset_source_of_truth_scope()
            return HookResult()
        if context.tool_call.name not in _OBSERVATION_TOOLS:
            return HookResult()

        repair = self._active_repair_context()
        if repair is None or not _repair_uses_artifact_set_as_source_of_truth(repair):
            self._reset_source_of_truth_scope()
            return HookResult()

        observed_paths = _extract_observation_paths(context.tool_call)
        if not observed_paths:
            return HookResult()
        if not repair.allowed_roots or not all(
            path_within_allowed_roots(path, repair.allowed_roots) for path in observed_paths
        ):
            return HookResult()

        self._sync_source_of_truth_scope(repair.allowed_roots)
        self._source_of_truth_observation_count += 1
        return HookResult()

    def _active_repair_context(self):
        dod_path = getattr(self.session, "active_dod_path", None)
        if not dod_path:
            return None
        path = Path(str(dod_path))
        if not path.exists():
            return None
        dod = self.dod_store.load(path)
        if dod.status == "done":
            return None
        return extract_active_repair_context(getattr(self.session, "messages", []))

    def _sync_source_of_truth_scope(self, allowed_roots: tuple[str, ...]) -> None:
        normalized = tuple(sorted(normalize_repair_path(root) for root in allowed_roots))
        if self._source_of_truth_scope_key == normalized:
            return
        self._source_of_truth_scope_key = normalized
        self._source_of_truth_observation_count = 0

    def _reset_source_of_truth_scope(self) -> None:
        self._source_of_truth_scope_key = None
        self._source_of_truth_observation_count = 0


class ActiveRepairMutationScopeHook(BaseToolHook):
    """Keep repair-phase mutations pinned to the concrete repair targets."""

    def __init__(
        self,
        *,
        dod_store: DefinitionOfDoneStore,
        project_root: Path,
        session: Any,
    ) -> None:
        self.dod_store = dod_store
        self.project_root = project_root
        self.session = session

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.tool_call.name not in _MUTATION_TOOLS:
            return HookResult()
        if context.source == "verification":
            return HookResult()

        repair = self._active_repair_context()
        if repair is None or not repair.allowed_paths:
            return HookResult()
        allowed_paths = {normalize_repair_path(path) for path in repair.allowed_paths}

        mutation_paths = _extract_mutation_paths(context.tool_call)
        if not mutation_paths:
            if context.tool_call.name == "bash" and _is_mutating_bash(
                str(context.tool_call.arguments.get("command", "")).strip()
            ):
                return HookResult(
                    decision=HookDecision.DENY,
                    message=(
                        "[Blocked - active repair mutation scope: the current repair already "
                        f"identifies `{repair.artifact_path}` as the concrete target.] "
                        "Suggestion: use write/edit/patch directly on one of the active repair "
                        "files instead of a broad shell mutation."
                    ),
                    terminal_state="blocked",
                )
            return HookResult()
        normalized_mutation_paths = [
            normalize_repair_path(path) for path in mutation_paths if str(path).strip()
        ]
        allowed_declared_outputs = _repair_declared_output_paths(
            repair,
            project_root=self.project_root,
        )

        if normalized_mutation_paths and all(
            path in allowed_paths for path in normalized_mutation_paths
        ):
            return HookResult()
        if normalized_mutation_paths and all(
            path in allowed_paths or path in allowed_declared_outputs
            for path in normalized_mutation_paths
        ):
            return HookResult()

        allowed_preview = ", ".join(f"`{path}`" for path in repair.allowed_paths[:3])
        if len(repair.allowed_paths) > 3:
            allowed_preview += ", ..."
        declared_preview = ", ".join(
            f"`{Path(path).name or path}`"
            for path in sorted(allowed_declared_outputs)[:3]
        )
        if len(allowed_declared_outputs) > 3:
            declared_preview += ", ..."
        suggestion_suffix = (
            f" Declared sibling outputs currently allowed inside this repair set include: {declared_preview}."
            if declared_preview
            else ""
        )
        return HookResult(
            decision=HookDecision.DENY,
            message=(
                "[Blocked - active repair mutation scope: verification already identified "
                f"`{repair.artifact_path}` as the current repair target.] Suggestion: keep "
                f"mutations on the active repair files only: {allowed_preview}."
                f"{suggestion_suffix}"
            ),
            terminal_state="blocked",
        )

    def _active_repair_context(self):
        dod_path = getattr(self.session, "active_dod_path", None)
        if not dod_path:
            return None
        path = Path(str(dod_path))
        if not path.exists():
            return None
        dod = self.dod_store.load(path)
        if dod.status == "done":
            return None
        return extract_active_repair_context(getattr(self.session, "messages", []))

class LateReferenceDriftHook(BaseToolHook):
    """Block reopening old reference paths once planned artifacts are well underway."""

    _MIN_COMPLETED_FILES = 3
    _MAX_COMPLETED_SCOPE_OBSERVATIONS = 4
    _REFERENCE_STUDY_HINTS = (
        "examine",
        "inspect",
        "study",
        "cadence",
        "format",
        "structure",
        "reference",
    )

    def __init__(self, *, dod_store: DefinitionOfDoneStore, project_root: Path, session: Any) -> None:
        self.dod_store = dod_store
        self.project_root = project_root
        self.session = session
        self._completed_scope_key: tuple[str, ...] | None = None
        self._completed_scope_observation_count = 0

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.tool_call.name not in _OBSERVATION_TOOLS:
            return HookResult()
        if context.source == "verification":
            return HookResult()

        completed_scope = self._completed_artifact_scope()
        if completed_scope is not None:
            observed_paths = _extract_observation_paths(context.tool_call)
            if not observed_paths:
                return HookResult()
            if all(path_within_allowed_roots(path, completed_scope) for path in observed_paths):
                self._sync_completed_scope_state(completed_scope)
                if (
                    context.source != "verification"
                    and self._completed_scope_observation_count
                    >= self._MAX_COMPLETED_SCOPE_OBSERVATIONS
                ):
                    roots_preview = ", ".join(f"`{root}`" for root in completed_scope[:2])
                    if len(completed_scope) > 2:
                        roots_preview += ", ..."
                    return HookResult(
                        decision=HookDecision.DENY,
                        message=(
                            "[Blocked - post-build audit loop: all explicitly planned artifacts "
                            "already exist and the current output set has already been inspected "
                            "several times.] Suggestion: move to verification now or make one "
                            "concrete edit for a specific mismatch inside "
                            f"{roots_preview} instead of more rereads."
                        ),
                        terminal_state="blocked",
                    )
                return HookResult()

            roots_preview = ", ".join(f"`{root}`" for root in completed_scope[:2])
            if len(completed_scope) > 2:
                roots_preview += ", ..."
            return HookResult(
                decision=HookDecision.DENY,
                message=(
                    "[Blocked - completed artifact set scope: all explicitly planned artifacts "
                    "already exist.] Suggestion: stay within the current output roots under "
                    f"{roots_preview} and use those files as the source of truth instead of "
                    "reopening earlier reference materials."
                ),
                terminal_state="blocked",
            )

        late_stage = self._late_stage_missing_artifact()
        if late_stage is None:
            return HookResult()
        missing_artifact, planned_roots = late_stage
        observed_paths = _extract_observation_paths(context.tool_call)
        if not observed_paths:
            return HookResult()
        if all(path_within_allowed_roots(path, planned_roots) for path in observed_paths):
            return HookResult()

        roots_preview = ", ".join(f"`{root}`" for root in planned_roots[:2])
        if len(planned_roots) > 2:
            roots_preview += ", ..."
        return HookResult(
            decision=HookDecision.DENY,
            message=(
                "[Blocked - late reference drift: several planned artifacts already exist and "
                f"`{missing_artifact}` is still missing.] Suggestion: finish the next missing "
                f"artifact inside {roots_preview} before reopening earlier reference materials."
            ),
            terminal_state="blocked",
        )

    def _late_stage_missing_artifact(self) -> tuple[str, tuple[str, ...]] | None:
        dod_path = getattr(self.session, "active_dod_path", None)
        if not dod_path:
            return None
        path = Path(str(dod_path))
        if not path.exists():
            return None
        dod = self.dod_store.load(path)
        if dod.status == "done":
            return None

        planned_targets = collect_planned_artifact_targets(
            dod,
            project_root=self.project_root,
        )
        if not planned_targets:
            return None

        missing_label = ""
        completed_files = 0
        planned_roots: list[str] = []
        seen_roots: set[str] = set()
        for target, expect_directory in planned_targets:
            satisfied = planned_artifact_target_satisfied(
                dod,
                target=target,
                expect_directory=expect_directory,
                project_root=self.project_root,
            )
            if not expect_directory:
                if satisfied:
                    completed_files += 1
                elif not missing_label:
                    missing_label = str(target)
                root = str(target.parent)
            else:
                if not satisfied and not missing_label:
                    missing_label = str(target)
                root = str(target)
            if root not in seen_roots:
                planned_roots.append(root)
                seen_roots.add(root)

        if not missing_label:
            return None
        minimum_completed_files = self._MIN_COMPLETED_FILES
        if completed_files >= 1 and self._reference_study_completed(dod):
            minimum_completed_files = 1
        if completed_files < minimum_completed_files:
            return None
        return missing_label, tuple(planned_roots)

    def _reference_study_completed(self, dod) -> bool:
        for item in dod.completed_items:
            text = str(item).strip().lower()
            if not text:
                continue
            if any(hint in text for hint in self._REFERENCE_STUDY_HINTS):
                return True
        return False

    async def post_tool_use(self, context: HookContext) -> HookResult:
        if context.tool_call.name in _MUTATION_TOOLS:
            self._reset_completed_scope_state()
            return HookResult()
        if context.tool_call.name not in _OBSERVATION_TOOLS:
            return HookResult()
        if context.source == "verification":
            return HookResult()

        completed_scope = self._completed_artifact_scope()
        if completed_scope is None:
            self._reset_completed_scope_state()
            return HookResult()

        observed_paths = _extract_observation_paths(context.tool_call)
        if not observed_paths:
            return HookResult()
        if not all(path_within_allowed_roots(path, completed_scope) for path in observed_paths):
            return HookResult()

        self._sync_completed_scope_state(completed_scope)
        self._completed_scope_observation_count += 1
        return HookResult()

    def _completed_artifact_scope(self) -> tuple[str, ...] | None:
        dod_path = getattr(self.session, "active_dod_path", None)
        if not dod_path:
            return None
        path = Path(str(dod_path))
        if not path.exists():
            return None
        dod = self.dod_store.load(path)
        if dod.status in {"done", "fixing"}:
            return None

        planned_targets = collect_planned_artifact_targets(
            dod,
            project_root=self.project_root,
        )
        if not planned_targets:
            return None
        if not all_planned_artifacts_exist(dod, project_root=self.project_root):
            return None

        planned_roots: list[str] = []
        seen_roots: set[str] = set()
        for target, expect_directory in planned_targets:
            root = str(target if expect_directory else target.parent)
            if root in seen_roots:
                continue
            seen_roots.add(root)
            planned_roots.append(root)
        return tuple(planned_roots)

    def _sync_completed_scope_state(self, completed_scope: tuple[str, ...]) -> None:
        normalized = tuple(sorted(completed_scope))
        if self._completed_scope_key == normalized:
            return
        self._completed_scope_key = normalized
        self._completed_scope_observation_count = 0

    def _reset_completed_scope_state(self) -> None:
        self._completed_scope_key = None
        self._completed_scope_observation_count = 0


class MissingPlannedOutputReadHook(BaseToolHook):
    """Block rereads of planned outputs that have not been created yet."""

    def __init__(
        self,
        *,
        dod_store: DefinitionOfDoneStore,
        project_root: Path,
        session: Any,
    ) -> None:
        self.dod_store = dod_store
        self.project_root = project_root
        self.session = session

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.tool_call.name != "read":
            return HookResult()
        if context.source == "verification":
            return HookResult()

        missing_output = self._missing_planned_output_path(context.tool_call)
        if missing_output is None:
            return HookResult()

        target_path, dod = missing_output
        message_lines = [
            (
                "[Blocked - missing planned output artifact: "
                f"`{target_path}` has not been created yet.]"
            ),
            (
                "Suggestion: create it now with one "
                f"`write(file_path=\"{display_runtime_path(target_path)}\", content=\"...\")` "
                "call instead of reading it first."
            ),
        ]

        outline_label = infer_output_outline_label(
            dod,
            target_path,
            project_root=self.project_root,
        )
        if outline_label:
            message_lines.append(
                f"Use the existing outline label `{outline_label}` so the new file matches the current artifact graph."
            )

        sibling_hint = self._existing_sibling_html_hint(target_path)
        if sibling_hint:
            message_lines.append(sibling_hint)

        return HookResult(
            decision=HookDecision.DENY,
            message=" ".join(message_lines),
            terminal_state="blocked",
        )

    def _missing_planned_output_path(
        self,
        tool_call: ToolCall,
    ) -> tuple[Path, Any] | None:
        dod_path = getattr(self.session, "active_dod_path", None)
        if not dod_path:
            return None
        path = Path(str(dod_path))
        if not path.exists():
            return None
        dod = self.dod_store.load(path)
        if dod.status in {"done", "fixing"}:
            return None

        raw_path = str(tool_call.arguments.get("file_path") or "").strip()
        if not raw_path:
            return None
        target_path = Path(raw_path).expanduser().resolve(strict=False)
        if target_path.exists():
            return None

        planned_targets = collect_planned_artifact_targets(
            dod,
            project_root=self.project_root,
        )
        if not planned_targets:
            return None

        for planned_target, expect_directory in planned_targets:
            if expect_directory:
                continue
            if planned_target != target_path:
                continue
            if planned_artifact_target_satisfied(
                dod,
                target=planned_target,
                expect_directory=False,
                project_root=self.project_root,
            ):
                return None
            return target_path, dod

        for planned_target, _ in planned_targets:
            if target_path not in collect_missing_declared_html_output_files(
                target=planned_target,
                project_root=self.project_root,
            ):
                continue
            return target_path, dod
        return None

    def _existing_sibling_html_hint(self, target_path: Path) -> str | None:
        if target_path.suffix.lower() not in {".html", ".htm"}:
            return None
        parent = target_path.parent
        if not parent.is_dir():
            return None
        siblings = sorted(
            child for child in parent.iterdir() if child.is_file() and child != target_path
        )
        html_siblings = [
            child for child in siblings if child.suffix.lower() in {".html", ".htm"}
        ]
        if not html_siblings:
            return None
        reference = html_siblings[-1]
        return (
            "Reuse the overall structure and navigation pattern from "
            f"`{reference.name}` as the starting pattern for this file."
        )


class HookManager:
    """Runs tool hooks across Loader's three lifecycle events."""

    def __init__(self, hooks: Iterable[ToolHook] | None = None) -> None:
        self.hooks = list(hooks or [])

    async def run_pre_tool_use(self, context: HookContext) -> HookRunSummary:
        return await self._run_event(HookEvent.PRE_TOOL_USE, context)

    async def run_post_tool_use(self, context: HookContext) -> HookRunSummary:
        return await self._run_event(HookEvent.POST_TOOL_USE, context)

    async def run_post_tool_use_failure(self, context: HookContext) -> HookRunSummary:
        return await self._run_event(HookEvent.POST_TOOL_USE_FAILURE, context)

    async def _run_event(
        self,
        event: HookEvent,
        context: HookContext,
    ) -> HookRunSummary:
        summary = HookRunSummary(tool_call=context.tool_call)
        for hook in self.hooks:
            if event == HookEvent.PRE_TOOL_USE:
                result = await hook.pre_tool_use(context)
            elif event == HookEvent.POST_TOOL_USE:
                result = await hook.post_tool_use(context)
            else:
                result = await hook.post_tool_use_failure(context)

            summary.injected_messages.extend(result.injected_messages)
            summary.metadata.update(result.metadata)
            if result.permission_override is not None:
                summary.permission_override = result.permission_override
                summary.permission_reason = result.permission_reason
            if result.output_override is not None:
                summary.output_override = result.output_override
                context.output = result.output_override
            if result.updated_arguments is not None:
                updated_call = ToolCall(
                    id=context.tool_call.id,
                    name=context.tool_call.name,
                    arguments=dict(result.updated_arguments),
                )
                context.tool_call = updated_call
                summary.tool_call = updated_call
            if result.message is not None:
                summary.message = result.message
            if result.terminal_state is not None:
                summary.terminal_state = result.terminal_state
            if result.decision != HookDecision.CONTINUE:
                summary.decision = result.decision
                return summary

        return summary


class DuplicateActionHook(BaseToolHook):
    """Pre-hook that cancels already-completed duplicate actions."""

    def __init__(self, action_tracker: ActionTracker) -> None:
        self.action_tracker = action_tracker

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.skip_duplicate_check:
            return HookResult()
        is_duplicate, reason = self.action_tracker.check_tool_call(
            context.tool_call.name,
            context.tool_call.arguments,
        )
        if not is_duplicate:
            return HookResult()
        return HookResult(
            decision=HookDecision.CANCEL,
            message=f"[Skipped - duplicate action: {reason}]",
            terminal_state="duplicate",
        )


class ActionValidationHook(BaseToolHook):
    """Pre-hook that blocks invalid or dangerous tool arguments."""

    def __init__(self, validator: PreActionValidator) -> None:
        self.validator = validator

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        validation = self.validator.validate(
            context.tool_call.name,
            context.tool_call.arguments,
        )
        if validation.valid:
            messages: list[str] = []
            if validation.reason and validation.severity == "warning":
                messages.append(f"[Validation warning] {validation.reason}")
            return HookResult(injected_messages=messages)

        message = f"[Blocked - {validation.reason}]"
        if validation.suggestion:
            message += f" Suggestion: {validation.suggestion}"
        return HookResult(
            decision=HookDecision.DENY,
            message=message,
            terminal_state="blocked",
        )


class RollbackTrackingHook(BaseToolHook):
    """Pre-hook that tracks rollback actions for destructive tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        rollback_plan: RollbackPlan | None,
    ) -> None:
        self.registry = registry
        self.rollback_plan = rollback_plan

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if self.rollback_plan is None:
            return HookResult()
        if not is_destructive_tool(
            context.tool_call.name,
            context.tool_call.arguments,
        ):
            return HookResult()

        async def read_file_for_backup(path: str) -> str:
            read_result = await self.registry.execute("read", file_path=path)
            return read_result.output if not read_result.is_error else ""

        rollback_action = await create_rollback_plan_for_action(
            context.tool_call.name,
            context.tool_call.arguments,
            read_file_for_backup,
        )
        if rollback_action is None:
            return HookResult()

        self.rollback_plan.actions.append(rollback_action)
        return HookResult(metadata={"rollback_action": rollback_action})


class ActionHistoryHook(BaseToolHook):
    """Post-hook that records successful actions for deduplication and loop checks."""

    def __init__(self, action_tracker: ActionTracker) -> None:
        self.action_tracker = action_tracker

    async def post_tool_use(self, context: HookContext) -> HookResult:
        if not context.record_action:
            return HookResult()
        self.action_tracker.record_tool_call(
            context.tool_call.name,
            context.tool_call.arguments,
        )
        return HookResult()


class MemoryLifecycleHook(BaseToolHook):
    """Mirror durable memory updates into the session notepad."""

    async def post_tool_use(self, context: HookContext) -> HookResult:
        if context.result is None or context.result.is_error:
            return HookResult()

        store = MemoryStore(context.permission_policy.workspace_root)
        if context.tool_call.name == "project_memory_add_note":
            category = str(context.tool_call.arguments.get("category", "")).strip()
            content = str(context.tool_call.arguments.get("content", "")).strip()
            if category and content:
                store.append_notepad_working(
                    f"Remembered note [{category}]: {content}"
                )
        elif context.tool_call.name == "project_memory_add_directive":
            directive = str(context.tool_call.arguments.get("directive", "")).strip()
            priority = str(
                context.tool_call.arguments.get("priority", "normal")
            ).strip()
            if directive:
                store.append_notepad_working(
                    f"Remembered directive [{priority}]: {directive}"
                )
        return HookResult()


def build_default_tool_hooks(
    *,
    action_tracker: ActionTracker,
    validator: PreActionValidator,
    registry: ToolRegistry,
    rollback_plan: RollbackPlan | None,
    workspace_root: Path,
    session: Any,
) -> HookManager:
    """Build Loader's default tool hook stack for one runtime turn."""

    return HookManager(
        [
            FilePathAliasHook(),
            SearchPathAliasHook(),
            RelativePathContextHook(action_tracker, workspace_root),
            ActiveRepairScopeHook(
                dod_store=DefinitionOfDoneStore(workspace_root),
                project_root=workspace_root,
                session=session,
            ),
            ActiveRepairMutationScopeHook(
                dod_store=DefinitionOfDoneStore(workspace_root),
                project_root=workspace_root,
                session=session,
            ),
            LateReferenceDriftHook(
                dod_store=DefinitionOfDoneStore(workspace_root),
                project_root=workspace_root,
                session=session,
            ),
            MissingPlannedOutputReadHook(
                dod_store=DefinitionOfDoneStore(workspace_root),
                project_root=workspace_root,
                session=session,
            ),
            DuplicateActionHook(action_tracker),
            ActionValidationHook(validator),
            RollbackTrackingHook(registry, rollback_plan),
            ActionHistoryHook(action_tracker),
            MemoryLifecycleHook(),
        ]
    )
