"""Shared helpers for runtime-owner metadata."""

from __future__ import annotations

from typing import Any

_KNOWN_RUNTIME_OWNER_PATHS = {
    "Agent": "public-agent",
    "RuntimeHandle": "runtime-handle",
}


def normalize_runtime_owner_type(value: Any) -> str | None:
    """Coerce persisted runtime-owner types into optional text."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_runtime_owner_path(
    value: Any,
    *,
    owner_type: str | None = None,
) -> str | None:
    """Coerce persisted runtime-owner paths into canonical text."""

    if value is not None:
        text = str(value).strip()
        if text:
            return text
    if owner_type is None:
        return None
    return _KNOWN_RUNTIME_OWNER_PATHS.get(owner_type, _camel_to_kebab(owner_type))


def build_runtime_owner_metadata(source: Any) -> dict[str, str | None]:
    """Build canonical runtime-owner metadata from one shell owner."""

    owner_type = (
        normalize_runtime_owner_type(source)
        if isinstance(source, str)
        else normalize_runtime_owner_type(type(source).__name__)
    )
    return {
        "owner_type": owner_type,
        "owner_path": normalize_runtime_owner_path(None, owner_type=owner_type),
    }


def format_runtime_owner_label(
    owner_type: str | None,
    owner_path: str | None,
) -> str | None:
    """Render one compact human-readable runtime-owner label."""

    normalized_type = normalize_runtime_owner_type(owner_type)
    normalized_path = normalize_runtime_owner_path(owner_path, owner_type=normalized_type)
    if normalized_type and normalized_path:
        return f"{normalized_path} ({normalized_type})"
    return normalized_path or normalized_type


def classify_runtime_owner_boundary(
    owner_type: str | None,
    owner_path: str | None,
) -> str | None:
    """Classify the persisted owner boundary for operator surfaces."""

    normalized_type = normalize_runtime_owner_type(owner_type)
    normalized_path = normalize_runtime_owner_path(owner_path, owner_type=normalized_type)
    if normalized_path == "runtime-handle" or normalized_type == "RuntimeHandle":
        return "runtime-first"
    if normalized_path == "public-agent" or normalized_type == "Agent":
        return "public-compat"
    return None


def format_runtime_boundary_label(
    owner_type: str | None,
    owner_path: str | None,
) -> str | None:
    """Render one concise operator-facing runtime boundary label."""

    boundary = classify_runtime_owner_boundary(owner_type, owner_path)
    owner = format_runtime_owner_label(owner_type, owner_path)
    if boundary and owner:
        return f"{boundary} via {owner}"
    return boundary or owner


def _camel_to_kebab(value: str) -> str:
    """Convert one CamelCase-ish class name into kebab-case."""

    chars: list[str] = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0:
            chars.append("-")
        chars.append(char.lower())
    return "".join(chars)
