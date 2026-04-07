"""Tests for transcript compaction and summary compression."""

from __future__ import annotations

from loader.llm.base import Message, Role
from loader.runtime.compaction import (
    SummaryCompressionBudget,
    compact_session_messages,
    compress_summary,
)


def test_compress_summary_dedupes_lines_and_collapses_whitespace() -> None:
    summary = "\n".join(
        [
            "Conversation summary:",
            "- Scope:   compact   earlier   messages.",
            "- Scope: compact earlier messages.",
            "- Current work: finish session persistence.",
            "- Current work: finish session persistence.",
        ]
    )

    result = compress_summary(summary, budget=SummaryCompressionBudget())

    assert result.removed_duplicate_lines == 2
    assert "- Scope: compact earlier messages." in result.summary
    assert "  compact   earlier" not in result.summary


def test_compact_session_messages_preserves_recent_messages() -> None:
    messages = [
        Message(role=Role.USER, content="First task framing"),
        Message(role=Role.ASSISTANT, content="Initial plan"),
        Message(role=Role.USER, content="Focus on runtime quality"),
        Message(role=Role.ASSISTANT, content="Tracked updated files"),
        Message(role=Role.USER, content="Verify the result"),
        Message(role=Role.ASSISTANT, content="Verification passed"),
    ]

    result = compact_session_messages(
        messages,
        keep_last_messages=4,
        current_task="Improve Loader runtime continuity",
    )

    assert result is not None
    assert result.removed_message_count == 2
    assert [message.content for message in result.messages[-4:]] == [
        message.content for message in messages[-4:]
    ]
    assert result.messages[0].content.startswith("[COMPACTED CONTEXT]")
    assert "Continuation instructions:" in result.messages[0].content
