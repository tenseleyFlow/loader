"""Tests for workflow-oriented tools introduced in Sprint 04."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loader.tools.workflow_tools import AskUserQuestionTool, TodoWriteTool


@pytest.mark.asyncio
async def test_todo_write_persists_and_returns_previous_state(tmp_path: Path) -> None:
    tool = TodoWriteTool(tmp_path)

    first = await tool.execute(
        todos=[
            {
                "content": "Create runtime router",
                "active_form": "Creating runtime router",
                "status": "in_progress",
            }
        ]
    )
    second = await tool.execute(
        todos=[
            {
                "content": "Create runtime router",
                "active_form": "Creating runtime router",
                "status": "completed",
            }
        ]
    )

    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    store_path = tmp_path / ".loader" / "todos" / "active.json"

    assert first.is_error is False
    assert first_payload["old_todos"] == []
    assert second_payload["old_todos"] == first_payload["new_todos"]
    assert json.loads(store_path.read_text()) == []


@pytest.mark.asyncio
async def test_todo_write_rejects_invalid_payloads_and_sets_verification_nudge(
    tmp_path: Path,
) -> None:
    tool = TodoWriteTool(tmp_path)

    empty = await tool.execute(todos=[])
    blank = await tool.execute(
        todos=[
            {
                "content": "  ",
                "active_form": "Reviewing plan",
                "status": "pending",
            }
        ]
    )
    nudged = await tool.execute(
        todos=[
            {
                "content": "Implement router",
                "active_form": "Implementing router",
                "status": "completed",
            },
            {
                "content": "Write tests",
                "active_form": "Writing tests",
                "status": "completed",
            },
            {
                "content": "Update docs",
                "active_form": "Updating docs",
                "status": "completed",
            },
        ]
    )

    assert empty.is_error is True
    assert "todos must not be empty" in empty.output
    assert blank.is_error is True
    assert "todo content must not be empty" in blank.output
    assert json.loads(nudged.output)["verification_nudge_needed"] is True


@pytest.mark.asyncio
async def test_ask_user_question_uses_callback_and_resolves_numbered_options() -> None:
    tool = AskUserQuestionTool()

    async def answer(question: str, options: list[str] | None) -> str:
        assert "Which path" in question
        assert options == ["Plan first", "Execute now"]
        return "2"

    result = await tool.execute(
        question="Which path should we take?",
        options=["Plan first", "Execute now"],
        user_response_handler=answer,
    )

    payload = json.loads(result.output)
    assert result.is_error is False
    assert payload["answer"] == "Execute now"
    assert payload["status"] == "answered"


@pytest.mark.asyncio
async def test_ask_user_question_requires_callback() -> None:
    tool = AskUserQuestionTool()

    result = await tool.execute(question="Need an answer?")

    assert result.is_error is True
    assert "user_response_handler" in result.output
