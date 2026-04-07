"""Permission modes and policy evaluation for tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path


class PermissionMode(IntEnum):
    """Runtime permission levels for Loader tool execution."""

    READ_ONLY = 1
    WORKSPACE_WRITE = 2
    DANGER_FULL_ACCESS = 3

    def as_str(self) -> str:
        """Return the user-facing string value for this mode."""
        return {
            self.READ_ONLY: "read-only",
            self.WORKSPACE_WRITE: "workspace-write",
            self.DANGER_FULL_ACCESS: "danger-full-access",
        }[self]

    @classmethod
    def from_str(cls, value: str) -> PermissionMode:
        """Parse a user-facing permission mode string."""
        normalized = value.strip().lower()
        mapping = {
            "read-only": cls.READ_ONLY,
            "workspace-write": cls.WORKSPACE_WRITE,
            "danger-full-access": cls.DANGER_FULL_ACCESS,
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


@dataclass(slots=True)
class PermissionOutcome:
    """Authorization result for one tool call."""

    decision: PermissionDecision
    active_mode: PermissionMode
    required_mode: PermissionMode
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        """Return whether the tool call is authorized to run."""
        return self.decision == PermissionDecision.ALLOW


@dataclass(slots=True)
class PermissionPolicy:
    """Permission policy based on active mode and tool requirements."""

    active_mode: PermissionMode
    tool_requirements: dict[str, PermissionMode]
    workspace_root: Path

    def required_mode_for(self, tool_name: str) -> PermissionMode:
        """Return the default required mode for a tool."""
        return self.tool_requirements.get(tool_name, PermissionMode.DANGER_FULL_ACCESS)

    def authorize(
        self,
        tool_name: str,
        *,
        required_mode: PermissionMode | None = None,
        override: PermissionOverride | None = None,
        override_reason: str | None = None,
    ) -> PermissionOutcome:
        """Authorize a tool call under the active runtime mode."""

        resolved_required_mode = required_mode or self.required_mode_for(tool_name)

        if override == PermissionOverride.DENY:
            return PermissionOutcome(
                decision=PermissionDecision.DENY,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=override_reason or f"tool '{tool_name}' denied by hook",
            )

        if override == PermissionOverride.ASK:
            return PermissionOutcome(
                decision=PermissionDecision.ASK,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=override_reason or f"tool '{tool_name}' requires approval",
            )

        if override == PermissionOverride.ALLOW:
            return PermissionOutcome(
                decision=PermissionDecision.ALLOW,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
                reason=override_reason,
            )

        if self.active_mode >= resolved_required_mode:
            return PermissionOutcome(
                decision=PermissionDecision.ALLOW,
                active_mode=self.active_mode,
                required_mode=resolved_required_mode,
            )

        return PermissionOutcome(
            decision=PermissionDecision.DENY,
            active_mode=self.active_mode,
            required_mode=resolved_required_mode,
            reason=(
                f"{tool_name} requires {resolved_required_mode.as_str()} "
                f"but Loader is running in {self.active_mode.as_str()} mode"
            ),
        )


def build_permission_policy(
    *,
    active_mode: PermissionMode,
    workspace_root: Path,
    tool_requirements: dict[str, PermissionMode],
) -> PermissionPolicy:
    """Create a permission policy with a canonical workspace root."""

    return PermissionPolicy(
        active_mode=active_mode,
        tool_requirements=dict(tool_requirements),
        workspace_root=workspace_root.expanduser().resolve(),
    )
