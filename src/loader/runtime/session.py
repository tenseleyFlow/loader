"""Conversation session ownership, persistence, and resume support."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..llm.base import Message
from .compaction import (
    DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD,
    DEFAULT_COMPACTION_KEEP_LAST_MESSAGES,
    SessionCompactionResult,
    compact_session_messages,
    estimate_message_tokens,
)

SESSION_VERSION = 1
DEFAULT_ROTATE_AFTER_BYTES = 256 * 1024
MAX_ROTATED_FILES = 3


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_session_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def normalize_usage(usage: dict[str, int] | None) -> dict[str, int]:
    """Map backend-specific usage keys onto Loader's canonical usage fields."""

    usage = usage or {}
    normalized: dict[str, int] = {}
    key_map = {
        "prompt_tokens": "input_tokens",
        "input_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "output_tokens": "output_tokens",
        "cache_creation_tokens": "cache_creation_tokens",
        "cache_read_tokens": "cache_read_tokens",
        "tool_calls": "tool_calls",
        "iterations": "iterations",
        "turns": "turns",
    }
    for key, value in usage.items():
        target_key = key_map.get(key)
        if target_key is None:
            continue
        normalized[target_key] = normalized.get(target_key, 0) + int(value)
    return normalized


@dataclass(slots=True)
class SessionCompaction:
    """Metadata describing the latest transcript compaction."""

    count: int = 0
    removed_message_count: int = 0
    summary: str = ""
    original_input_tokens: int = 0
    compressed_input_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "removed_message_count": self.removed_message_count,
            "summary": self.summary,
            "original_input_tokens": self.original_input_tokens,
            "compressed_input_tokens": self.compressed_input_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionCompaction:
        return cls(
            count=int(data.get("count", 0)),
            removed_message_count=int(data.get("removed_message_count", 0)),
            summary=str(data.get("summary", "")),
            original_input_tokens=int(data.get("original_input_tokens", 0)),
            compressed_input_tokens=int(data.get("compressed_input_tokens", 0)),
        )


@dataclass(slots=True)
class SessionSnapshot:
    """Persisted session snapshot stored under `.loader/sessions/`."""

    session_id: str
    created_at: str
    updated_at: str
    messages: list[Message]
    usage: dict[str, int] = field(default_factory=dict)
    active_dod_path: str | None = None
    current_task: str | None = None
    workflow_mode: str = "execute"
    permission_mode: str = "workspace-write"
    compaction: SessionCompaction | None = None
    version: int = SESSION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [message.to_persisted_dict() for message in self.messages],
            "usage": dict(self.usage),
            "active_dod_path": self.active_dod_path,
            "current_task": self.current_task,
            "workflow_mode": self.workflow_mode,
            "permission_mode": self.permission_mode,
            "compaction": self.compaction.to_dict() if self.compaction else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSnapshot:
        return cls(
            version=int(data.get("version", SESSION_VERSION)),
            session_id=str(data["session_id"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            messages=[
                Message.from_persisted_dict(message)
                for message in data.get("messages", [])
            ],
            usage=normalize_usage(data.get("usage")),
            active_dod_path=data.get("active_dod_path"),
            current_task=data.get("current_task"),
            workflow_mode=str(data.get("workflow_mode", "execute")),
            permission_mode=str(data.get("permission_mode", "workspace-write")),
            compaction=(
                SessionCompaction.from_dict(data["compaction"])
                if data.get("compaction")
                else None
            ),
        )


class SessionStore:
    """Persist and load session snapshots under `.loader/`."""

    def __init__(
        self,
        project_root: Path,
        *,
        rotate_after_bytes: int = DEFAULT_ROTATE_AFTER_BYTES,
        max_rotated_files: int = MAX_ROTATED_FILES,
    ) -> None:
        self.project_root = project_root
        self.loader_root = project_root / ".loader"
        self.sessions_root = self.loader_root / "sessions"
        self.state_root = self.loader_root / "state"
        self.pointer_path = self.state_root / "current_session.json"
        self.rotate_after_bytes = rotate_after_bytes
        self.max_rotated_files = max_rotated_files

    def ensure_layout(self) -> None:
        """Ensure the session/state layout exists on disk."""

        self.sessions_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        """Return the canonical save path for one session."""

        return self.sessions_root / f"{session_id}.json"

    def save(self, snapshot: SessionSnapshot) -> Path:
        """Persist one session snapshot atomically with rotation."""

        self.ensure_layout()
        path = self.session_path(snapshot.session_id)
        payload = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
        self._rotate_if_needed(path, payload)
        _atomic_write(path, payload)
        _atomic_write(
            self.pointer_path,
            json.dumps(
                {
                    "session_id": snapshot.session_id,
                    "updated_at": snapshot.updated_at,
                    "path": str(path),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        return path

    def load(self, session_id: str) -> SessionSnapshot:
        """Load one named session snapshot."""

        path = self.session_path(session_id)
        return SessionSnapshot.from_dict(json.loads(path.read_text()))

    def load_latest(self) -> SessionSnapshot | None:
        """Load the latest active session if any exists."""

        self.ensure_layout()
        if self.pointer_path.exists():
            pointer = json.loads(self.pointer_path.read_text())
            session_id = pointer.get("session_id")
            if isinstance(session_id, str):
                path = self.session_path(session_id)
                if path.exists():
                    return SessionSnapshot.from_dict(json.loads(path.read_text()))

        candidates = sorted(
            path
            for path in self.sessions_root.glob("*.json")
            if not path.stem.rsplit(".", 1)[-1].isdigit()
        )
        if not candidates:
            return None
        path = max(candidates, key=lambda item: item.stat().st_mtime)
        return SessionSnapshot.from_dict(json.loads(path.read_text()))

    def _rotate_if_needed(self, path: Path, payload: str) -> None:
        """Rotate older persisted snapshots when the session file grows large."""

        if not path.exists():
            return
        size_after_write = max(path.stat().st_size, len(payload.encode("utf-8")))
        if size_after_write < self.rotate_after_bytes:
            return

        for index in range(self.max_rotated_files, 0, -1):
            rotated_path = path.with_suffix(f".{index}.json")
            if not rotated_path.exists():
                continue
            if index == self.max_rotated_files:
                rotated_path.unlink()
            else:
                rotated_path.replace(path.with_suffix(f".{index + 1}.json"))

        path.replace(path.with_suffix(".1.json"))


@dataclass
class ConversationSession:
    """Owns the conversation history used for turn execution."""

    system_message_factory: Callable[[], Message]
    few_shot_factory: Callable[[], list[Message]]
    project_root: Path
    messages: list[Message] = field(default_factory=list)
    session_id: str = field(default_factory=_generate_session_id)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    usage_totals: dict[str, int] = field(default_factory=dict)
    active_dod_path: str | None = None
    current_task: str | None = None
    workflow_mode: str = "execute"
    permission_mode: str = "workspace-write"
    compaction: SessionCompaction | None = None
    rotate_after_bytes: int = DEFAULT_ROTATE_AFTER_BYTES
    max_rotated_files: int = MAX_ROTATED_FILES
    auto_compaction_input_tokens_threshold: int = DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD
    compaction_keep_last_messages: int = DEFAULT_COMPACTION_KEEP_LAST_MESSAGES

    def __post_init__(self) -> None:
        self.store = SessionStore(
            self.project_root,
            rotate_after_bytes=self.rotate_after_bytes,
            max_rotated_files=self.max_rotated_files,
        )

    @property
    def storage_path(self) -> Path:
        """Return the current persisted session snapshot path."""

        return self.store.session_path(self.session_id)

    def build_request_messages(self) -> list[Message]:
        """Build the full request transcript for the backend."""

        request_messages = [self.system_message_factory()]
        if len(self.messages) <= 2:
            request_messages.extend(self.few_shot_factory())
        request_messages.extend(self.messages)
        return request_messages

    def append(self, message: Message) -> None:
        """Append one message to the session and persist the snapshot."""

        self.messages.append(message)
        self.touch()
        self.persist()

    def extend(self, messages: list[Message]) -> None:
        """Append many messages to the session and persist the snapshot."""

        self.messages.extend(messages)
        self.touch()
        self.persist()

    def clear(self) -> None:
        """Reset the session history in place."""

        self.messages.clear()
        self.active_dod_path = None
        self.current_task = None
        self.workflow_mode = "execute"
        self.compaction = None
        self.usage_totals = {}
        self.touch()
        self.persist()

    def touch(self) -> None:
        """Refresh the session update timestamp."""

        self.updated_at = _utc_now()

    def update_runtime_state(
        self,
        *,
        active_dod_path: str | None = None,
        current_task: str | None = None,
        workflow_mode: str | None = None,
        permission_mode: str | None = None,
    ) -> None:
        """Update persisted runtime state that lives beside the messages."""

        if active_dod_path is not None:
            self.active_dod_path = active_dod_path
        if current_task is not None:
            self.current_task = current_task
        if workflow_mode is not None:
            self.workflow_mode = workflow_mode
        if permission_mode is not None:
            self.permission_mode = permission_mode
        self.touch()
        self.persist()

    def maybe_compact(self) -> SessionCompactionResult | None:
        """Compact the transcript when the current request grows too large."""

        request_messages = self.build_request_messages()
        estimated_input_tokens = estimate_message_tokens(request_messages)
        if estimated_input_tokens < self.auto_compaction_input_tokens_threshold:
            return None

        result = compact_session_messages(
            self.messages,
            keep_last_messages=self.compaction_keep_last_messages,
            previous_summary=self.compaction.summary if self.compaction else None,
            current_task=self.current_task,
            original_input_tokens=estimated_input_tokens,
        )
        if result is None:
            return None

        self.messages[:] = result.messages
        self.compaction = SessionCompaction(
            count=1 if self.compaction is None else self.compaction.count + 1,
            removed_message_count=result.removed_message_count,
            summary=result.summary,
            original_input_tokens=result.original_input_tokens,
            compressed_input_tokens=result.compressed_input_tokens,
        )
        self.touch()
        self.persist()
        return result

    def record_turn_usage(
        self,
        usage: dict[str, int],
        *,
        tool_calls: int,
        iterations: int,
    ) -> dict[str, int]:
        """Accumulate normalized usage totals across the current session."""

        totals = normalize_usage(usage)
        if "tool_calls" not in totals:
            totals["tool_calls"] = tool_calls
        if "iterations" not in totals:
            totals["iterations"] = iterations
        totals["turns"] = totals.get("turns", 0) + 1
        for key, value in totals.items():
            self.usage_totals[key] = self.usage_totals.get(key, 0) + value
        self.touch()
        self.persist()
        return dict(self.usage_totals)

    def persist(self) -> Path:
        """Write the current session snapshot to disk."""

        snapshot = SessionSnapshot(
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            messages=list(self.messages),
            usage=dict(self.usage_totals),
            active_dod_path=self.active_dod_path,
            current_task=self.current_task,
            workflow_mode=self.workflow_mode,
            permission_mode=self.permission_mode,
            compaction=self.compaction,
        )
        return self.store.save(snapshot)

    @classmethod
    def load(
        cls,
        *,
        project_root: Path,
        system_message_factory: Callable[[], Message],
        few_shot_factory: Callable[[], list[Message]],
        session_id: str | None = None,
        rotate_after_bytes: int = DEFAULT_ROTATE_AFTER_BYTES,
        max_rotated_files: int = MAX_ROTATED_FILES,
        auto_compaction_input_tokens_threshold: int = DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD,
        compaction_keep_last_messages: int = DEFAULT_COMPACTION_KEEP_LAST_MESSAGES,
    ) -> ConversationSession | None:
        """Load the latest or named conversation session from disk."""

        store = SessionStore(
            project_root,
            rotate_after_bytes=rotate_after_bytes,
            max_rotated_files=max_rotated_files,
        )
        snapshot = store.load(session_id) if session_id else store.load_latest()
        if snapshot is None:
            return None
        instance = cls.__new__(cls)
        instance.system_message_factory = system_message_factory
        instance.few_shot_factory = few_shot_factory
        instance.project_root = project_root
        instance.messages = list(snapshot.messages)
        instance.session_id = snapshot.session_id
        instance.created_at = snapshot.created_at
        instance.updated_at = snapshot.updated_at
        instance.usage_totals = dict(snapshot.usage)
        instance.active_dod_path = snapshot.active_dod_path
        instance.current_task = snapshot.current_task
        instance.workflow_mode = snapshot.workflow_mode
        instance.permission_mode = snapshot.permission_mode
        instance.compaction = snapshot.compaction
        instance.rotate_after_bytes = rotate_after_bytes
        instance.max_rotated_files = max_rotated_files
        instance.auto_compaction_input_tokens_threshold = (
            auto_compaction_input_tokens_threshold
        )
        instance.compaction_keep_last_messages = compaction_keep_last_messages
        instance.store = store
        return instance


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content)
    temp_path.replace(path)
