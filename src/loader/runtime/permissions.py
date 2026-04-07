"""Permission modes, rules, and policy evaluation for tool execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any


class PermissionMode(IntEnum):
    """Runtime permission levels for Loader tool execution."""

    READ_ONLY = 1
    WORKSPACE_WRITE = 2
    DANGER_FULL_ACCESS = 3
    PROMPT = 4
    ALLOW = 5

    def as_str(self) -> str:
        """Return the user-facing string value for this mode."""
        return {
            self.READ_ONLY: "read-only",
            self.WORKSPACE_WRITE: "workspace-write",
            self.DANGER_FULL_ACCESS: "danger-full-access",
            self.PROMPT: "prompt",
            self.ALLOW: "allow",
        }[self]

    @classmethod
    def from_str(cls, value: str) -> PermissionMode:
        """Parse a user-facing permission mode string."""
        normalized = value.strip().lower()
        mapping = {
            "read-only": cls.READ_ONLY,
            "workspace-write": cls.WORKSPACE_WRITE,
            "danger-full-access": cls.DANGER_FULL_ACCESS,
            "prompt": cls.PROMPT,
            "allow": cls.ALLOW,
        }
        if normalized not in mapping:
            raise ValueError(f"Unknown permission mode: {value}")
        return mapping[normalized]


class PermissionOverride(StrEnum):
    """Hook-provided authorization override."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionDecision(StrEnum):
    """Final policy decision for one tool call."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class PermissionRuleDisposition(StrEnum):
    """Disposition associated with a rule list."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(slots=True)
class PermissionRule:
    """Conservative matcher for one permission rule."""

    tool_name: str | None = None
    contains: str | None = None
    path_contains: str | None = None
    raw: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PermissionRule:
        """Parse a rule from JSON-compatible data."""

        unknown_keys = sorted(set(data) - {"tool", "contains", "path_contains"})
        if unknown_keys:
            joined = ", ".join(unknown_keys)
            raise ValueError(f"Unknown permission rule fields: {joined}")

        tool_name = _normalize_optional_text(data.get("tool"))
        contains = _normalize_optional_text(data.get("contains"))
        path_contains = _normalize_optional_text(data.get("path_contains"))
        if not any((tool_name, contains, path_contains)):
            raise ValueError(
                "Permission rules require at least one of `tool`, `contains`, or `path_contains`"
            )

        raw_parts: list[str] = []
        if tool_name:
            raw_parts.append(f"tool={tool_name}")
        if contains:
            raw_parts.append(f"contains={contains}")
        if path_contains:
            raw_parts.append(f"path_contains={path_contains}")
        return cls(
            tool_name=tool_name,
            contains=contains,
            path_contains=path_contains,
            raw=", ".join(raw_parts),
        )

    def matches(self, request: PermissionRequest) -> bool:
        """Return whether this rule matches the current request."""

        if self.tool_name is not None and self.tool_name != request.tool_name:
            return False
        if self.contains is not None and self.contains not in request.input_summary:
            return False
        if self.path_contains is not None:
            if request.path_hint is None or self.path_contains not in request.path_hint:
                return False
        return True


@dataclass(slots=True)
class PermissionRuleSet:
    """Full rule bundle loaded for the current runtime."""

    allow: list[PermissionRule] = field(default_factory=list)
    deny: list[PermissionRule] = field(default_factory=list)
    ask: list[PermissionRule] = field(default_factory=list)
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render a JSON-compatible representation."""

        return {
            "allow": [_rule_to_dict(rule) for rule in self.allow],
            "deny": [_rule_to_dict(rule) for rule in self.deny],
            "ask": [_rule_to_dict(rule) for rule in self.ask],
        }

    @property
    def counts(self) -> dict[str, int]:
        """Return simple counts by rule bucket."""

        return {
            "allow": len(self.allow),
            "deny": len(self.deny),
            "ask": len(self.ask),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        source_path: Path | None = None,
    ) -> PermissionRuleSet:
        """Parse a rule set from JSON-compatible data."""

        unknown_keys = sorted(set(data) - {"allow", "deny", "ask"})
        if unknown_keys:
            joined = ", ".join(unknown_keys)
            raise ValueError(f"Unknown permission rule sections: {joined}")

        parsed: dict[str, list[PermissionRule]] = {}
        for bucket in ("allow", "deny", "ask"):
            raw_rules = data.get(bucket, [])
            if raw_rules is None:
                raw_rules = []
            if not isinstance(raw_rules, list):
                raise ValueError(f"`{bucket}` rules must be a list")
            parsed[bucket] = []
            for item in raw_rules:
                if not isinstance(item, dict):
                    raise ValueError(f"`{bucket}` rules must contain objects")
                parsed[bucket].append(PermissionRule.from_dict(item))

        return cls(
            allow=parsed["allow"],
            deny=parsed["deny"],
            ask=parsed["ask"],
            source_path=source_path,
        )


@dataclass(slots=True)
class PermissionRequest:
    """Full authorization request for one tool invocation."""

    tool_name: str
    input_summary: str
    active_mode: PermissionMode
    required_mode: PermissionMode
    path_hint: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class PermissionOutcome:
    """Authorization result for one tool call."""

    decision: PermissionDecision
    active_mode: PermissionMode
    required_mode: PermissionMode
    reason: str | None = None
    matched_rule: PermissionRule | None = None
    matched_disposition: PermissionRuleDisposition | None = None
    request: PermissionRequest | None = None

    @property
    def allowed(self) -> bool:
        """Return whether the tool call is authorized to run."""
        return self.decision == PermissionDecision.ALLOW


@dataclass(slots=True)
class PermissionPolicy:
    """Permission policy based on active mode, tool requirements, and rules."""

    active_mode: PermissionMode
    tool_requirements: dict[str, PermissionMode]
    workspace_root: Path
    rules: PermissionRuleSet = field(default_factory=PermissionRuleSet)

    def required_mode_for(self, tool_name: str) -> PermissionMode:
        """Return the default required mode for a tool."""
        return self.tool_requirements.get(tool_name, PermissionMode.DANGER_FULL_ACCESS)

    def rule_counts(self) -> dict[str, int]:
        """Return rule counts by disposition."""
        return self.rules.counts

    @property
    def prompting_enabled(self) -> bool:
        """Return whether the active policy can trigger interactive approval."""
        return self.active_mode == PermissionMode.PROMPT or bool(self.rules.ask)

    def authorize(
        self,
        tool_name: str,
        *,
        required_mode: PermissionMode | None = None,
        override: PermissionOverride | None = None,
        override_reason: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> PermissionOutcome:
        """Authorize a tool call under the active runtime mode."""

        resolved_required_mode = required_mode or self.required_mode_for(tool_name)
        request = PermissionRequest(
            tool_name=tool_name,
            input_summary=summarize_permission_input(tool_name, arguments or {}),
            active_mode=self.active_mode,
            required_mode=resolved_required_mode,
            path_hint=permission_path_hint(arguments or {}),
            reason=override_reason,
        )

        deny_rule = self._find_matching_rule(self.rules.deny, request)
        if deny_rule is not None:
            return PermissionOutcome(
                decision=PermissionDecision.DENY,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=(
                    f"tool '{tool_name}' denied by rule '{deny_rule.raw}'"
                ),
                matched_rule=deny_rule,
                matched_disposition=PermissionRuleDisposition.DENY,
                request=request,
            )

        ask_rule = self._find_matching_rule(self.rules.ask, request)
        allow_rule = self._find_matching_rule(self.rules.allow, request)

        if override == PermissionOverride.DENY:
            return PermissionOutcome(
                decision=PermissionDecision.DENY,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=override_reason or f"tool '{tool_name}' denied by hook",
                request=request,
            )

        if override == PermissionOverride.ASK:
            return PermissionOutcome(
                decision=PermissionDecision.ASK,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=(
                    override_reason
                    or f"tool '{tool_name}' requires approval due to hook guidance"
                ),
                request=request,
            )

        if ask_rule is not None:
            return PermissionOutcome(
                decision=PermissionDecision.ASK,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=(
                    f"tool '{tool_name}' requires approval due to ask rule "
                    f"'{ask_rule.raw}'"
                ),
                matched_rule=ask_rule,
                matched_disposition=PermissionRuleDisposition.ASK,
                request=request,
            )

        if override == PermissionOverride.ALLOW:
            return PermissionOutcome(
                decision=PermissionDecision.ALLOW,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=override_reason,
                request=request,
            )

        if allow_rule is not None:
            return PermissionOutcome(
                decision=PermissionDecision.ALLOW,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=(
                    f"tool '{tool_name}' allowed by rule '{allow_rule.raw}'"
                ),
                matched_rule=allow_rule,
                matched_disposition=PermissionRuleDisposition.ALLOW,
                request=request,
            )

        if self.active_mode == PermissionMode.PROMPT:
            return PermissionOutcome(
                decision=PermissionDecision.ASK,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=(
                    f"tool '{tool_name}' requires approval in prompt mode"
                ),
                request=request,
            )

        if self.active_mode == PermissionMode.ALLOW:
            return PermissionOutcome(
                decision=PermissionDecision.ALLOW,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                request=request,
            )

        if self.active_mode >= resolved_required_mode:
            return PermissionOutcome(
                decision=PermissionDecision.ALLOW,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                request=request,
            )

        return PermissionOutcome(
            decision=PermissionDecision.DENY,
            active_mode=self.active_mode,
            required_mode=resolved_required_mode,
            reason=(
                f"{tool_name} requires {resolved_required_mode.as_str()} "
                f"but Loader is running in {self.active_mode.as_str()} mode"
            ),
            request=request,
        )

    @staticmethod
    def _find_matching_rule(
        rules: list[PermissionRule],
        request: PermissionRequest,
    ) -> PermissionRule | None:
        for rule in rules:
            if rule.matches(request):
                return rule
        return None


@dataclass(slots=True)
class PermissionConfigStatus:
    """Inspection result for persisted permission rules."""

    valid: bool
    source_path: Path
    rules: PermissionRuleSet
    error: str | None = None


def build_permission_policy(
    *,
    active_mode: PermissionMode,
    workspace_root: Path,
    tool_requirements: dict[str, PermissionMode],
    rules: PermissionRuleSet | None = None,
) -> PermissionPolicy:
    """Create a permission policy with a canonical workspace root."""

    return PermissionPolicy(
        active_mode=active_mode,
        tool_requirements=dict(tool_requirements),
        workspace_root=workspace_root.expanduser().resolve(),
        rules=rules or PermissionRuleSet(),
    )


def default_permission_rules_path(project_root: Path) -> Path:
    """Return the canonical workspace-local permission rule file."""

    return project_root / ".loader" / "permission-rules.json"


def load_permission_rules(
    project_root: Path | str,
) -> PermissionConfigStatus:
    """Load workspace-local permission rules, failing closed on invalid JSON."""

    resolved_root = Path(project_root).expanduser().resolve()
    source_path = default_permission_rules_path(resolved_root)
    if not source_path.exists():
        return PermissionConfigStatus(
            valid=True,
            source_path=source_path,
            rules=PermissionRuleSet(source_path=source_path),
        )

    try:
        payload = json.loads(source_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("permission-rules.json must contain an object")
        rules = PermissionRuleSet.from_dict(payload, source_path=source_path)
        return PermissionConfigStatus(
            valid=True,
            source_path=source_path,
            rules=rules,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return PermissionConfigStatus(
            valid=False,
            source_path=source_path,
            rules=PermissionRuleSet(source_path=source_path),
            error=str(exc),
        )


def summarize_permission_input(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Build a compact, deterministic input summary for policy matching."""

    ordered_parts: list[str] = []
    for key in sorted(arguments):
        value = arguments[key]
        rendered = _render_permission_value(value)
        ordered_parts.append(f"{key}={rendered}")
    if not ordered_parts:
        return tool_name
    return f"{tool_name} " + " ".join(ordered_parts)


def permission_path_hint(arguments: dict[str, Any]) -> str | None:
    """Extract the best path hint from tool arguments."""

    for key in ("file_path", "path", "cwd", "directory"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _render_permission_value(value: Any) -> str:
    if isinstance(value, str):
        compact = " ".join(value.split())
        if len(compact) > 120:
            return compact[:117] + "..."
        return compact
    if isinstance(value, int | float | bool):
        return str(value)
    if isinstance(value, list):
        rendered = ", ".join(_render_permission_value(item) for item in value[:5])
        if len(value) > 5:
            rendered += ", ..."
        return f"[{rendered}]"
    if isinstance(value, dict):
        return "{...}"
    return repr(value)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Permission rule values must be strings")
    normalized = value.strip()
    return normalized or None


def _rule_to_dict(rule: PermissionRule) -> dict[str, str]:
    payload: dict[str, str] = {}
    if rule.tool_name is not None:
        payload["tool"] = rule.tool_name
    if rule.contains is not None:
        payload["contains"] = rule.contains
    if rule.path_contains is not None:
        payload["path_contains"] = rule.path_contains
    return payload
