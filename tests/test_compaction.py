"""Tests for transcript compaction and summary compression."""

from __future__ import annotations

from loader.llm.base import Message, Role, ToolCall
from loader.runtime.compaction import (
    SummaryCompressionBudget,
    build_session_summary,
    compact_session_messages,
    compress_summary,
    infer_preferred_next_step,
    resolve_auto_compaction_input_tokens_threshold,
    summarize_confirmed_facts,
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
            role=Role.ASSISTANT,
            content="Inspecting the setup chapter title.",
            tool_calls=[
                ToolCall(
                    id="read-2",
                    name="read",
                    arguments={"file_path": "~/Loader/guides/fortran/chapters/02-setup.html"},
                )
            ],
        ),
        Message.tool_result_message(
            tool_call_id="read-2",
            display_content=(
                "   1\t<!DOCTYPE html>\n"
                "   2\t<html>\n"
                "  61\t<h1>Chapter 2: Setting Up Fortran</h1>\n"
                "  62\t</html>\n"
            ),
            result_content=(
                "   1\t<!DOCTYPE html>\n"
                "   2\t<html>\n"
                "  61\t<h1>Chapter 2: Setting Up Fortran</h1>\n"
                "  62\t</html>\n"
            ),
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
    assert "02-setup.html = Chapter 2: Setting Up Fortran" in summary
    assert "Preferred next step:" in summary
    assert "`~/Loader/guides/fortran/index.html`" in summary


def test_summarize_confirmed_facts_extracts_chapter_titles_from_read_results() -> None:
    messages = [
        Message(
            role=Role.ASSISTANT,
            content="I will inspect the chapter files.",
            tool_calls=[
                ToolCall(
                    id="read-1",
                    name="read",
                    arguments={"file_path": "/tmp/fortran/chapters/01-introduction.html"},
                ),
                ToolCall(
                    id="read-2",
                    name="read",
                    arguments={"file_path": "/tmp/fortran/chapters/02-setup.html"},
                ),
            ],
        ),
        Message.tool_result_message(
            tool_call_id="read-1",
            display_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
            result_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
        ),
        Message.tool_result_message(
            tool_call_id="read-2",
            display_content="<title>Chapter 2: Setting Up Fortran</title>\n",
            result_content="<title>Chapter 2: Setting Up Fortran</title>\n",
        ),
    ]

    confirmed_facts = summarize_confirmed_facts(messages, max_items=2)

    assert confirmed_facts is not None
    assert "Chapter titles confirmed:" in confirmed_facts
    assert "01-introduction.html = Chapter 1: Introduction to Fortran" in confirmed_facts
    assert "02-setup.html = Chapter 2: Setting Up Fortran" in confirmed_facts


def test_infer_preferred_next_step_uses_confirmed_chapter_pairs() -> None:
    messages = [
        Message(
            role=Role.ASSISTANT,
            content="I should inspect the chapter and then update the index.",
            tool_calls=[
                ToolCall(
                    id="read-index",
                    name="read",
                    arguments={"file_path": "/tmp/fortran/index.html"},
                ),
                ToolCall(
                    id="read-1",
                    name="read",
                    arguments={"file_path": "/tmp/fortran/chapters/01-introduction.html"},
                ),
            ],
        ),
        Message.tool_result_message(
            tool_call_id="read-1",
            display_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
            result_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
        ),
    ]

    next_step = infer_preferred_next_step(
        messages,
        current_task="Update /tmp/fortran/index.html so the chapter list matches the real files.",
    )

    assert next_step == (
        "Update `/tmp/fortran/index.html` using the confirmed chapter file/title pairs "
        "instead of rereading files."
    )


def test_infer_preferred_next_step_uses_latest_verification_gap() -> None:
    messages = [
        Message(
            role=Role.ASSISTANT,
            content="I should inspect the chapter and then update the index.",
            tool_calls=[
                ToolCall(
                    id="read-index",
                    name="read",
                    arguments={"file_path": "/tmp/fortran/index.html"},
                ),
                ToolCall(
                    id="read-1",
                    name="read",
                    arguments={"file_path": "/tmp/fortran/chapters/01-introduction.html"},
                ),
                ToolCall(
                    id="verify-1",
                    name="bash",
                    arguments={"command": "python3 - <<'PY'\n...\nPY"},
                ),
            ],
        ),
        Message.tool_result_message(
            tool_call_id="read-1",
            display_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
            result_content="<h1>Chapter 1: Introduction to Fortran</h1>\n",
        ),
        Message.tool_result_message(
            tool_call_id="verify-1",
            display_content=(
                "Missing links:\n"
                "chapters/05-control-structures.html -> missing\n"
                "chapters/06-input-output.html -> missing\n"
            ),
            result_content=(
                "Missing links:\n"
                "chapters/05-control-structures.html -> missing\n"
                "chapters/06-input-output.html -> missing\n"
            ),
            is_error=True,
        ),
    ]

    confirmed_facts = summarize_confirmed_facts(messages, max_items=2)
    next_step = infer_preferred_next_step(
        messages,
        current_task="Update /tmp/fortran/index.html so the chapter list matches the real files.",
    )

    assert confirmed_facts is not None
    assert "Verification gaps: missing TOC links chapters/05-control-structures.html" in confirmed_facts
    assert next_step == (
        "Update `/tmp/fortran/index.html` to fix the specific verification failures "
        "(missing TOC links chapters/05-control-structures.html, "
        "chapters/06-input-output.html) instead of restarting discovery."
    )


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
