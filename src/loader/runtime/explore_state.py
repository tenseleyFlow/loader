"""Lightweight persisted continuity for the read-only explore lane."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..llm.base import Message

EXPLORE_STATE_VERSION = 1
DEFAULT_EXPLORE_MAX_MESSAGES = 12


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_messages(value: Any) -> list[Message]:
    if not isinstance(value, list):
        return []
    messages: list[Message] = []
    for item in value:
        if isinstance(item, dict):
            messages.append(Message.from_persisted_dict(item))
    return messages


@dataclass(slots=True)
class ExploreSnapshot:
    """Persisted read-only explore transcript state."""

    updated_at: str = field(default_factory=_utc_now)
    turn_count: int = 0
    model_name: str | None = None
    messages: list[Message] = field(default_factory=list)
    last_history_mode: str | None = None
    last_query: str | None = None
    last_response: str | None = None
    version: int = EXPLORE_STATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "model_name": self.model_name,
            "messages": [message.to_persisted_dict() for message in self.messages],
            "last_history_mode": self.last_history_mode,
            "last_query": self.last_query,
            "last_response": self.last_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExploreSnapshot:
        return cls(
            version=int(data.get("version", EXPLORE_STATE_VERSION)),
            updated_at=str(data.get("updated_at", _utc_now())),
            turn_count=int(data.get("turn_count", 0)),
            model_name=_optional_text(data.get("model_name")),
            messages=_normalize_messages(data.get("messages")),
            last_history_mode=_optional_text(data.get("last_history_mode")),
            last_query=_optional_text(data.get("last_query")),
            last_response=_optional_text(data.get("last_response")),
        )


class ExploreStateStore:
    """Persist and load lightweight explore-lane continuity."""

    def __init__(
        self,
        project_root: Path | str,
        *,
        max_messages: int = DEFAULT_EXPLORE_MAX_MESSAGES,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.max_messages = max_messages

    @property
    def state_path(self) -> Path:
        return self.project_root / ".loader" / "state" / "explore.json"

    def load(self) -> ExploreSnapshot | None:
        path = self.state_path
        if not path.exists():
            return None
        try:
            return ExploreSnapshot.from_dict(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def save(self, snapshot: ExploreSnapshot) -> ExploreSnapshot:
        snapshot.messages = list(snapshot.messages)[-self.max_messages :]
        snapshot.updated_at = _utc_now()
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n")
        return snapshot

    def clear(self) -> None:
        path = self.state_path
        if path.exists():
            path.unlink()
