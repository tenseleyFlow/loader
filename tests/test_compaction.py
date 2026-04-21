"""Tests for transcript compaction and summary compression."""

from __future__ import annotations

from loader.llm.base import Message, Role, ToolCall
from loader.runtime.compaction import (
    SummaryCompressionBudget,
    build_session_summary,
    compact_session_messages,
    compress_summary,
    resolve_auto_compaction_input_tokens_threshold,
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


def test_build_session_summary_skips_nested_compacted_context_content() -> None:
    messages = [
        Message(
            role=Role.USER,
            content=(
                "[COMPACTED CONTEXT]\nConversation summary:\n"
                "- Scope: older work\n- Current work: old state"
            ),
        ),
        Message(role=Role.ASSISTANT, content="Read the chapter index."),
        Message(role=Role.USER, content="Update the chapter links."),
    ]

    summary = build_session_summary(
        messages,
        previous_summary="[COMPACTED CONTEXT]\nConversation summary:\n- Scope: older work",
        current_task="Repair the table of contents links",
    )

    assert "Recent user requests: [COMPACTED CONTEXT]" not in summary
    assert "Pending work: [COMPACTED CONTEXT]" not in summary
    assert "- Previously compacted context retained." in summary


def test_build_session_summary_preserves_confirmed_facts_and_next_step() -> None:
    messages = [
        Message(
            role=Role.TOOL,
            content=(
                "Observation [notepad_write_working]: Result: "
                "02-basic-syntax.html -> 02-setup.html\n"
                "03-variables-data-types.html -> 03-basics.html"
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content="Checking the index before editing it.",
            tool_calls=[
                ToolCall(
                    id="read-1",
                    name="read",
                    arguments={"file_path": "~/Loader/guides/fortran/index.html"},
                )
            ],
        ),
        Message(
            role=Role.TOOL,
            content=(
                "Observation [glob]: Result: "
                "/Users/mfwolffe/Loader/guides/fortran/chapters/01-introduction.html\n"
                "/Users/mfwolffe/Loader/guides/fortran/chapters/02-setup.html\n"
                "/Users/mfwolffe/Loader/guides/fortran/chapters/03-basics.html\n"
                "/Users/mfwolffe/Loader/guides/fortran/chapters/04-variables.html"
            ),
        ),
    ]

    summary = build_session_summary(
        messages,
        current_task=(
            "Update ~/Loader/guides/fortran/index.html with the correct chapter links."
        ),
    )

    assert "Confirmed facts:" in summary
    assert "02-basic-syntax.html -> 02-setup.html" in summary
    assert "Existing files include 01-introduction.html" in summary
    assert "Preferred next step:" in summary
    assert "`~/Loader/guides/fortran/index.html`" in summary


def test_compact_session_messages_uses_single_continuation_instruction_block() -> None:
    messages = [
        Message(role=Role.USER, content="Task framing"),
        Message(role=Role.ASSISTANT, content="Initial plan"),
        Message(role=Role.USER, content="Keep going"),
        Message(role=Role.ASSISTANT, content="Still working"),
        Message(role=Role.USER, content="Use the known mapping"),
    ]

    result = compact_session_messages(
        messages,
        keep_last_messages=2,
        current_task="Repair the table of contents links",
    )

    assert result is not None
    assert result.messages[0].content.count("Continuation instructions:") == 1


def test_resolve_auto_compaction_threshold_uses_context_window_as_upper_bound() -> None:
    assert resolve_auto_compaction_input_tokens_threshold(
        100_000,
        context_window=131_072,
    ) == 98_304
    assert resolve_auto_compaction_input_tokens_threshold(
        100_000,
        context_window=262_144,
    ) == 100_000
    assert resolve_auto_compaction_input_tokens_threshold(
        100_000,
        context_window=8_192,
    ) == 12_000
