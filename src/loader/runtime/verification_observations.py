"""Typed verification observations captured during runtime verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class VerificationObservationStatus(StrEnum):
    """How one verification observation resolved at runtime."""

    PLANNED = "planned"
    PENDING = "pending"
    STALE = "stale"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    MISSING = "missing"


@dataclass(slots=True)
class VerificationObservation:
    """One typed verification fact captured near execution time."""

    status: str
    summary: str
    command: str | None = None
    kind: str | None = None
    exit_code: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize one observation for persisted runtime state."""

        return {
            "status": self.status,
            "summary": self.summary,
            "command": self.command,
            "kind": self.kind,
            "exit_code": self.exit_code,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationObservation:
        """Load one persisted observation."""

        return cls(
            status=normalize_verification_observation_status(data.get("status")),
            summary=str(data.get("summary", "")),
            command=_optional_text(data.get("command")),
            kind=_optional_text(data.get("kind")),
            exit_code=_optional_int(data.get("exit_code")),
            detail=_optional_text(data.get("detail")),
        )


def normalize_verification_observation_status(value: Any) -> str:
    """Coerce persisted observation statuses into the canonical enum set."""

    if value is None:
        return VerificationObservationStatus.MISSING.value
    normalized = str(value).strip().lower()
    for candidate in VerificationObservationStatus:
        if candidate.value == normalized:
            return candidate.value
    return VerificationObservationStatus.MISSING.value


def normalize_verification_observations(value: Any) -> list[VerificationObservation]:
    """Coerce persisted observation payloads into typed entries."""

    if not isinstance(value, list):
        return []
    entries: list[VerificationObservation] = []
    for item in value:
        if isinstance(item, dict):
            entries.append(VerificationObservation.from_dict(item))
    return entries


def summarize_verification_observations(
    entries: list[VerificationObservation],
    *,
    max_items: int | None = None,
) -> list[str]:
    """Project typed verification observations into concise summaries."""

    summaries: list[str] = []
    limit = len(entries) if max_items is None else max_items
    for entry in entries[:limit]:
        summary = entry.summary.strip()
        if summary and summary not in summaries:
            summaries.append(summary)
    return summaries


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
