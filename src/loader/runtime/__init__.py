"""Runtime primitives for Loader's turn engine."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AgentEvent",
    "CapabilityProfile",
    "ConversationSession",
    "DefinitionOfDone",
    "DefinitionOfDoneStore",
    "HookManager",
    "PermissionMode",
    "PermissionPolicy",
    "RuntimeTraceEvent",
    "RuntimeTracer",
    "TurnSummary",
    "VerificationEvidence",
    "build_permission_policy",
    "resolve_backend_capability_profile",
    "resolve_capability_profile",
]


def __getattr__(name: str) -> Any:
    """Lazily expose runtime exports to avoid import cycles."""

    if name in {
        "CapabilityProfile",
        "resolve_backend_capability_profile",
        "resolve_capability_profile",
    }:
        module = import_module(".capabilities", __name__)
    elif name in {
        "DefinitionOfDone",
        "DefinitionOfDoneStore",
        "VerificationEvidence",
    }:
        module = import_module(".dod", __name__)
    elif name in {"AgentEvent", "TurnSummary"}:
        module = import_module(".events", __name__)
    elif name in {"PermissionMode", "PermissionPolicy", "build_permission_policy"}:
        module = import_module(".permissions", __name__)
    elif name in {"HookManager"}:
        module = import_module(".hooks", __name__)
    elif name in {"ConversationSession"}:
        module = import_module(".session", __name__)
    elif name in {"RuntimeTraceEvent", "RuntimeTracer"}:
        module = import_module(".tracing", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value
