"""Transcript compaction and summary compression for Loader sessions."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ..llm.base import Message, Role

DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD = 100_000
DEFAULT_COMPACTION_KEEP_LAST_MESSAGES = 4
DEFAULT_MAX_CHARS = 1_200
DEFAULT_MAX_LINES = 24
DEFAULT_MAX_LINE_CHARS = 160


@dataclass(slots=True)
class SummaryCompressionBudget:
    """Budget for priority-aware summary compression."""

    max_chars: int = DEFAULT_MAX_CHARS
    max_lines: int = DEFAULT_MAX_LINES
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS


@dataclass(slots=True)
class SummaryCompressionResult:
    """Outcome from compressing a multi-line summary."""

    summary: str
    original_chars: int
    compressed_chars: int
    original_lines: int
    compressed_lines: int
    removed_duplicate_lines: int
    omitted_lines: int
    truncated: bool


@dataclass(slots=True)
class SessionCompactionResult:
    """Outcome from compacting a persisted session transcript."""

    messages: list[Message]
    summary: str
    removed_message_count: int
    preserved_message_count: int
    original_input_tokens: int
    compressed_input_tokens: int
    compression: SummaryCompressionResult


def estimate_message_tokens(messages: list[Message]) -> int:
    """Estimate input tokens for a transcript using a simple local heuristic."""

    total_chars = 0
    for message in messages:
        total_chars += len(message.content)
        total_chars += sum(len(tool_call.name) for tool_call in message.tool_calls)
        total_chars += sum(len(str(tool_call.arguments)) for tool_call in message.tool_calls)
        total_chars += sum(len(tool_result.content) for tool_result in message.tool_results)
    return max(1, total_chars // 4)


def compress_summary(
    summary: str,
    budget: SummaryCompressionBudget | None = None,
) -> SummaryCompressionResult:
    """Compress summary text using claw-style line priorities and budgets."""

    budget = budget or SummaryCompressionBudget()
    original_chars = len(summary)
    original_lines = len(summary.splitlines())

    normalized_lines: list[str] = []
    seen: set[str] = set()
    removed_duplicate_lines = 0
    for raw_line in summary.splitlines():
        normalized = _collapse_inline_whitespace(raw_line)
        if not normalized:
            continue
        truncated = _truncate_line(normalized, budget.max_line_chars)
        dedupe_key = truncated.lower()
        if dedupe_key in seen:
            removed_duplicate_lines += 1
            continue
        seen.add(dedupe_key)
        normalized_lines.append(truncated)

    if not normalized_lines or budget.max_chars <= 0 or budget.max_lines <= 0:
        return SummaryCompressionResult(
            summary="",
            original_chars=original_chars,
            compressed_chars=0,
            original_lines=original_lines,
            compressed_lines=0,
            removed_duplicate_lines=removed_duplicate_lines,
            omitted_lines=len(normalized_lines),
            truncated=original_chars > 0,
        )

    selected_indexes: list[int] = []
    for priority in range(4):
        for index, line in enumerate(normalized_lines):
            if index in selected_indexes or _line_priority(line) != priority:
                continue
            candidate = [normalized_lines[item] for item in selected_indexes] + [line]
            if len(candidate) > budget.max_lines:
                continue
            if _joined_char_count(candidate) > budget.max_chars:
                continue
            selected_indexes.append(index)

    selected_lines = [normalized_lines[index] for index in selected_indexes]
    if not selected_lines:
        selected_lines = [_truncate_line(normalized_lines[0], budget.max_chars)]

    omitted_lines = len(normalized_lines) - len(selected_lines)
    omission_notice = f"- ... {omitted_lines} additional line(s) omitted."
    if omitted_lines > 0:
        candidate = selected_lines + [omission_notice]
        if (
            len(candidate) <= budget.max_lines
            and _joined_char_count(candidate) <= budget.max_chars
        ):
            selected_lines.append(omission_notice)

    compressed_summary = "\n".join(selected_lines)
    return SummaryCompressionResult(
        summary=compressed_summary,
        original_chars=original_chars,
        compressed_chars=len(compressed_summary),
        original_lines=original_lines,
        compressed_lines=len(selected_lines),
        removed_duplicate_lines=removed_duplicate_lines,
        omitted_lines=omitted_lines,
        truncated=compressed_summary != summary.strip(),
    )


def compact_session_messages(
    messages: list[Message],
    *,
    keep_last_messages: int = DEFAULT_COMPACTION_KEEP_LAST_MESSAGES,
    budget: SummaryCompressionBudget | None = None,
    previous_summary: str | None = None,
    current_task: str | None = None,
    original_input_tokens: int | None = None,
) -> SessionCompactionResult | None:
    """Compact older messages into one continuation summary message."""

    if len(messages) <= keep_last_messages:
        return None

    removed_messages = messages[:-keep_last_messages]
    preserved_messages = list(messages[-keep_last_messages:])
    summary_text = build_session_summary(
        removed_messages,
        previous_summary=previous_summary,
        current_task=current_task,
    )
    compression = compress_summary(summary_text, budget=budget)
    summary_message = Message(
        role=Role.USER,
        content=(
            "[COMPACTED CONTEXT]\n"
            f"{compression.summary}\n\n"
            "Continuation instructions:\n"
            "- Continue from the preserved recent messages.\n"
            "- Honor the active DoD, workflow mode, and permission mode.\n"
            "- Do not ask the user to repeat already-captured context unless essential."
        ),
    )
    compacted_messages = [summary_message, *preserved_messages]
    input_tokens = original_input_tokens or estimate_message_tokens(messages)
    compressed_input_tokens = estimate_message_tokens(compacted_messages)
    return SessionCompactionResult(
        messages=compacted_messages,
        summary=summary_message.content,
        removed_message_count=len(removed_messages),
        preserved_message_count=len(preserved_messages),
        original_input_tokens=input_tokens,
        compressed_input_tokens=compressed_input_tokens,
        compression=compression,
    )


def build_session_summary(
    messages: list[Message],
    *,
    previous_summary: str | None = None,
    current_task: str | None = None,
) -> str:
    """Build a structured session summary before compression."""

    user_messages = [
        _collapse_inline_whitespace(message.content)
        for message in messages
        if message.role == Role.USER and message.content.strip()
    ]
    assistant_messages = [
        _collapse_inline_whitespace(message.content)
        for message in messages
        if message.role == Role.ASSISTANT and message.content.strip()
    ]
    tool_names = [
        tool_call.name
        for message in messages
        for tool_call in message.tool_calls
        if tool_call.name
    ]
    key_files = _extract_key_files(messages)

    scope = current_task or (user_messages[0] if user_messages else "Continue the current task.")
    current_work = assistant_messages[-1] if assistant_messages else (user_messages[-1] if user_messages else "Resume the latest work state.")
    pending_work = user_messages[-1] if user_messages else current_work
    recent_requests = "; ".join(user_messages[-3:]) if user_messages else "None captured."
    tool_summary = ", ".join(name for name, _ in Counter(tool_names).most_common(5)) if tool_names else "None yet."
    file_summary = ", ".join(key_files[:6]) if key_files else "None referenced."

    lines = [
        "Conversation summary:",
        f"- Scope: {scope}",
        f"- Current work: {current_work}",
        f"- Pending work: {pending_work}",
        f"- Key files referenced: {file_summary}",
        f"- Tools mentioned: {tool_summary}",
        f"- Recent user requests: {recent_requests}",
    ]
    if previous_summary:
        previous_line = _collapse_inline_whitespace(previous_summary.splitlines()[0])
        lines.append(f"- Previously compacted context: {previous_line}")
    lines.extend(
        [
            f"- Newly compacted context: {len(messages)} earlier message(s) summarized.",
            "Continuation instructions:",
            "- Continue from the preserved recent messages.",
            "- Keep the active DoD, workflow mode, and permission mode aligned.",
            "- Avoid redoing completed work unless verification fails.",
        ]
    )
    return "\n".join(lines)


def _extract_key_files(messages: list[Message]) -> list[str]:
    pattern = re.compile(r"(?:/|\.{1,2}/|[A-Za-z0-9_.-]+/)[A-Za-z0-9_./-]+\.[A-Za-z0-9]+")
    files: list[str] = []
    for message in messages:
        for match in pattern.findall(message.content):
            if match not in files:
                files.append(match)
        for tool_call in message.tool_calls:
            for key in ("file_path", "path", "cwd"):
                value = tool_call.arguments.get(key)
                if isinstance(value, str) and value and value not in files:
                    files.append(value)
    return files


def _collapse_inline_whitespace(line: str) -> str:
    return " ".join(line.split())


def _truncate_line(line: str, max_chars: int) -> str:
    if max_chars <= 0 or len(line) <= max_chars:
        return line
    if max_chars == 1:
        return "..."
    return f"{line[: max_chars - 3]}..."


def _joined_char_count(lines: list[str]) -> int:
    return sum(len(line) for line in lines) + max(0, len(lines) - 1)


def _line_priority(line: str) -> int:
    if line in {"Summary:", "Conversation summary:"} or _is_core_detail(line):
        return 0
    if line.endswith(":"):
        return 1
    if line.startswith("- "):
        return 2
    return 3


def _is_core_detail(line: str) -> bool:
    return any(
        line.startswith(prefix)
        for prefix in (
            "- Scope:",
            "- Current work:",
            "- Pending work:",
            "- Key files referenced:",
            "- Tools mentioned:",
            "- Recent user requests:",
            "- Previously compacted context:",
            "- Newly compacted context:",
        )
    )
