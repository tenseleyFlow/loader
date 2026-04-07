"""Persisted prompt-contract history for inspection surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class PromptSnapshot:
    """One persisted prompt contract."""

    timestamp: str
    workflow_mode: str
    permission_mode: str
    current_task: str | None
    prompt_format: str
    prompt_sections: list[str] = field(default_factory=list)
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "workflow_mode": self.workflow_mode,
            "permission_mode": self.permission_mode,
            "current_task": self.current_task,
            "prompt_format": self.prompt_format,
            "prompt_sections": list(self.prompt_sections),
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptSnapshot:
        return cls(
            timestamp=str(data.get("timestamp", "")),
            workflow_mode=str(data.get("workflow_mode", "execute")),
            permission_mode=str(data.get("permission_mode", "workspace-write")),
            current_task=_optional_text(data.get("current_task")),
            prompt_format=str(data.get("prompt_format", "native")),
            prompt_sections=[
                str(item) for item in data.get("prompt_sections", []) if str(item).strip()
            ],
            content=str(data.get("content", "")),
        )

    @classmethod
    def create(
        cls,
        *,
        workflow_mode: str,
        permission_mode: str,
        current_task: str | None,
        prompt_format: str,
        prompt_sections: list[str],
        content: str,
    ) -> PromptSnapshot:
        return cls(
            timestamp=_utc_now(),
            workflow_mode=workflow_mode,
            permission_mode=permission_mode,
            current_task=_optional_text(current_task),
            prompt_format=prompt_format,
            prompt_sections=list(prompt_sections),
            content=content,
        )

    def matches_contract(self, other: PromptSnapshot) -> bool:
        """Return whether two prompt snapshots represent the same contract."""

        return (
            self.workflow_mode == other.workflow_mode
            and self.permission_mode == other.permission_mode
            and self.current_task == other.current_task
            and self.prompt_format == other.prompt_format
            and self.prompt_sections == other.prompt_sections
            and self.content == other.content
        )


def normalize_prompt_history(value: Any) -> list[PromptSnapshot]:
    """Coerce persisted prompt history into prompt snapshots."""

    if not isinstance(value, list):
        return []
    snapshots: list[PromptSnapshot] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        snapshot = PromptSnapshot.from_dict(item)
        if snapshot.content:
            snapshots.append(snapshot)
    return snapshots


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
