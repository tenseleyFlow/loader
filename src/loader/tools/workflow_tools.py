"""Workflow-oriented tools for task tracking and user clarification."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..runtime.permissions import PermissionMode
from .base import Tool, ToolResult

UserQuestionHandler = Callable[[str, list[str] | None], Awaitable[str]]

TODO_STATUSES = {"pending", "in_progress", "completed"}


@dataclass(slots=True)
class TodoItem:
    """Structured todo item compatible with Loader workflow state."""

    content: str
    active_form: str
    status: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TodoItem:
        active_form = str(
            data.get("active_form")
            or data.get("activeForm")
            or data.get("active")
            or ""
        ).strip()
        return cls(
            content=str(data.get("content", "")).strip(),
            active_form=active_form,
            status=str(data.get("status", "")).strip().lower(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "content": self.content,
            "active_form": self.active_form,
            "status": self.status,
        }


class TodoWriteTool(Tool):
    """Persist the current task list under `.loader/`."""

    required_permission = PermissionMode.READ_ONLY

    def __init__(self, workspace_root: Path | str | None = None) -> None:
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve() if workspace_root else None
        )

    @property
    def name(self) -> str:
        return "TodoWrite"

    def set_workspace_root(self, workspace_root: Path | None) -> None:
        self.workspace_root = workspace_root

    @property
    def description(self) -> str:
        return (
            "Persist the current task list under .loader/todos/. "
            "Use it to track pending, in-progress, and completed work items."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Current task list for the active workflow.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Short task description in base form.",
                            },
                            "active_form": {
                                "type": "string",
                                "description": "Progressive-tense form, e.g. 'Running tests'.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "Current todo status.",
                            },
                        },
                        "required": ["content", "active_form", "status"],
                    },
                }
            },
            "required": ["todos"],
        }

    async def execute(
        self,
        todos: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ToolResult:
        try:
            items = [TodoItem.from_dict(todo) for todo in todos]
            self._validate_items(items)
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)

        store_path = self._store_path()
        old_todos = await asyncio.to_thread(self._read_existing_items, store_path)

        all_done = all(item.status == "completed" for item in items)
        persisted_items = [] if all_done else [item.to_dict() for item in items]

        store_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            store_path.write_text,
            json.dumps(persisted_items, indent=2, sort_keys=True),
        )

        verification_nudge_needed = (
            all_done
            and len(items) >= 3
            and not any("verif" in item.content.lower() for item in items)
        )

        payload = {
            "old_todos": old_todos,
            "new_todos": [item.to_dict() for item in items],
            "verification_nudge_needed": verification_nudge_needed,
            "store_path": str(store_path),
        }
        return ToolResult(
            output=json.dumps(payload, indent=2, sort_keys=True),
            metadata=payload,
        )

    def _store_path(self) -> Path:
        root = self.workspace_root or Path.cwd()
        return root / ".loader" / "todos" / "active.json"

    def _read_existing_items(self, store_path: Path) -> list[dict[str, Any]]:
        if not store_path.exists():
            return []
        raw = json.loads(store_path.read_text())
        if not isinstance(raw, list):
            return []
        items: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                items.append(TodoItem.from_dict(item).to_dict())
        return items

    def _validate_items(self, items: list[TodoItem]) -> None:
        if not items:
            raise ValueError("todos must not be empty")
        for item in items:
            if not item.content:
                raise ValueError("todo content must not be empty")
            if not item.active_form:
                raise ValueError("todo active_form must not be empty")
            if item.status not in TODO_STATUSES:
                raise ValueError(
                    "todo status must be one of pending, in_progress, or completed"
                )


class AskUserQuestionTool(Tool):
    """Ask the user one structured question and capture the answer."""

    required_permission = PermissionMode.READ_ONLY

    @property
    def name(self) -> str:
        return "AskUserQuestion"

    @property
    def description(self) -> str:
        return "Ask the user one question and wait for their response."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The exact question to present to the user.",
                },
                "options": {
                    "type": "array",
                    "description": "Optional short answer choices.",
                    "items": {"type": "string"},
                },
            },
            "required": ["question"],
        }

    async def execute(
        self,
        question: str,
        options: list[str] | None = None,
        user_response_handler: UserQuestionHandler | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        normalized_question = question.strip()
        normalized_options = [
            str(option).strip()
            for option in (options or [])
            if str(option).strip()
        ]

        if not normalized_question:
            return ToolResult("question must not be empty", is_error=True)
        if user_response_handler is None:
            return ToolResult(
                "AskUserQuestion requires a user_response_handler callback",
                is_error=True,
            )

        answer = (
            await user_response_handler(
                normalized_question,
                normalized_options or None,
            )
        ).strip()
        resolved_answer = self._resolve_answer(answer, normalized_options or None)
        payload = {
            "question": normalized_question,
            "options": normalized_options or None,
            "answer": resolved_answer,
            "status": "answered",
        }
        return ToolResult(
            output=json.dumps(payload, indent=2, sort_keys=True),
            metadata=payload,
        )

    @staticmethod
    def _resolve_answer(answer: str, options: list[str] | None) -> str:
        if not options:
            return answer
        try:
            index = int(answer) - 1
        except ValueError:
            return answer
        if 0 <= index < len(options):
            return options[index]
        return answer
