"""Transcript compaction and summary compression for Loader sessions."""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..llm.base import Message, Role, ToolCall

DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD = 100_000
MIN_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD = 12_000
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


def resolve_auto_compaction_input_tokens_threshold(
    configured_threshold: int,
    *,
    context_window: int | None = None,
) -> int:
    """Resolve one compaction threshold from config and model context."""

    threshold = max(1, int(configured_threshold))
    if context_window is None or context_window <= 0:
        return threshold

    context_bound = max(
        MIN_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD,
        int(context_window * 0.75),
    )
    context_bound = min(DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD, context_bound)
    return min(threshold, context_bound)


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
        if (
            message.role == Role.USER
            and message.content.strip()
            and not _is_compacted_context_message(message.content)
        )
    ]
    assistant_messages = [
        _collapse_inline_whitespace(message.content)
        for message in messages
        if (
            message.role == Role.ASSISTANT
            and message.content.strip()
            and not _is_compacted_context_message(message.content)
        )
    ]
    tool_names = [
        tool_call.name
        for message in messages
        for tool_call in message.tool_calls
        if tool_call.name
    ]
    key_files = extract_key_files(messages)
    confirmed_facts = summarize_confirmed_facts(messages)
    preferred_next_step = infer_preferred_next_step(
        messages,
        current_task=current_task,
    )

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
    if confirmed_facts:
        lines.append(f"- Confirmed facts: {confirmed_facts}")
    if preferred_next_step:
        lines.append(f"- Preferred next step: {preferred_next_step}")
    if previous_summary:
        lines.append("- Previously compacted context retained.")
    lines.append(f"- Newly compacted context: {len(messages)} earlier message(s) summarized.")
    return "\n".join(lines)


def extract_key_files(messages: list[Message], *, limit: int | None = 6) -> list[str]:
    pattern = re.compile(
        r"(?:~/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+|"
        r"/(?:Users|home|tmp|var|private)/(?:[A-Za-z0-9_. -]+/)+[A-Za-z0-9_.-]+\.[A-Za-z0-9]+|"
        r"(?:\.{1,2}/|[A-Za-z0-9_.-]+/)[A-Za-z0-9_./-]+\.[A-Za-z0-9]+)"
    )
    files: list[str] = []
    for message in messages:
        if _is_compacted_context_message(message.content):
            continue
        for match in pattern.findall(message.content):
            normalized = _normalize_path_candidate(match)
            if normalized and normalized not in files:
                files.append(normalized)
        for tool_call in message.tool_calls:
            for key in ("file_path", "path", "cwd"):
                value = tool_call.arguments.get(key)
                if not isinstance(value, str):
                    continue
                normalized = _normalize_path_candidate(value)
                if normalized and normalized not in files:
                    files.append(normalized)
        if limit is not None and len(files) >= limit:
            return files[:limit]
    return files


def summarize_confirmed_facts(messages: list[Message], *, max_items: int = 2) -> str | None:
    """Summarize recent confirmed discoveries from successful tool results."""

    facts = _collect_confirmed_facts(messages)

    if not facts:
        return None
    return " | ".join(facts[:max_items])


def infer_preferred_next_step(
    messages: list[Message],
    *,
    current_task: str | None = None,
) -> str | None:
    """Infer one concrete next step from the task and recent transcript."""

    if summarize_confirmed_facts(messages, max_items=1) is None:
        return None

    target_path = _choose_target_path(messages, current_task=current_task)
    has_confirmed_titles = _summarize_html_title_discovery(messages) is not None
    verification_gap = _summarize_latest_html_verification_gap(messages)
    if target_path:
        if verification_gap:
            return (
                f"Update `{target_path}` to fix the specific verification failures "
                f"({verification_gap}) instead of restarting discovery."
            )
        if has_confirmed_titles:
            return (
                f"Update `{target_path}` using the confirmed chapter file/title pairs "
                "instead of rereading files."
            )
        return (
            f"Update `{target_path}` using the confirmed findings instead of "
            "restarting earlier discovery steps."
        )
    return "Continue from the confirmed findings instead of restarting earlier discovery."


def _collapse_inline_whitespace(line: str) -> str:
    return " ".join(line.split())


def _is_compacted_context_message(content: str) -> bool:
    return content.lstrip().startswith("[COMPACTED CONTEXT]")


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
            "- Confirmed facts:",
            "- Preferred next step:",
            "- Previously compacted context:",
            "- Newly compacted context:",
        )
    )


def _normalize_path_candidate(value: str) -> str | None:
    text = str(value).strip().rstrip(".,;:")
    if not text:
        return None
    if text.startswith(("~/", "./", "../")):
        return text
    if text.startswith(("/Users/", "/home/", "/tmp/", "/var/", "/private/")):
        return text
    if "/" in text and not text.startswith("/") and "." in Path(text).name:
        return text
    return None


def _observed_tool_name(content: str) -> str | None:
    match = re.match(r"Observation \[([^\]]+)\]:", content.strip())
    if match:
        return match.group(1).strip()
    return None


def _collect_confirmed_facts(messages: list[Message]) -> list[str]:
    facts: list[str] = []
    tool_calls_by_id = {
        tool_call.id: tool_call
        for message in messages
        for tool_call in message.tool_calls
    }

    explicit_mapping_fact = _collect_explicit_mapping_fact(
        messages,
        tool_calls_by_id=tool_calls_by_id,
    )
    if explicit_mapping_fact:
        facts.append(explicit_mapping_fact)

    verification_gap_fact = _collect_html_verification_gap_fact(
        messages,
        tool_calls_by_id=tool_calls_by_id,
    )
    if verification_gap_fact:
        facts.append(verification_gap_fact)

    title_fact = _summarize_html_title_discovery(
        messages,
        tool_calls_by_id=tool_calls_by_id,
    )
    if title_fact:
        facts.append(title_fact)

    file_fact = _collect_html_file_discovery_fact(
        messages,
        tool_calls_by_id=tool_calls_by_id,
    )
    if file_fact:
        facts.append(file_fact)

    return facts


def _collect_explicit_mapping_fact(
    messages: list[Message],
    *,
    tool_calls_by_id: dict[str, ToolCall],
) -> str | None:
    mappings: list[str] = []
    for message in messages:
        if message.role != Role.TOOL or _is_compacted_context_message(message.content):
            continue
        if any(result.is_error for result in message.tool_results):
            continue

        tool_name = _resolve_tool_name(
            message,
            tool_calls_by_id=tool_calls_by_id,
        )
        if tool_name not in {
            "notepad_write_working",
            "notepad_append",
            "notepad_write_priority",
            "notepad_write_manual",
        }:
            continue

        payload = "\n".join(
            result.content.strip()
            for result in message.tool_results
            if result.content.strip()
        ) or message.content
        pairs = re.findall(
            r"([A-Za-z0-9_.-]+\.html)\s*->\s*([A-Za-z0-9_.-]+\.html)",
            payload,
        )
        for left, right in pairs:
            mapping = f"{left} -> {right}"
            if mapping not in mappings:
                mappings.append(mapping)

    if not mappings:
        return None

    preview = ", ".join(mappings[:4])
    if len(mappings) > 4:
        preview += ", ..."
    return f"Filename mappings confirmed: {preview}"


def _summarize_html_mappings(payload: str) -> str | None:
    pairs = re.findall(
        r"([A-Za-z0-9_.-]+\.html)\s*->\s*([A-Za-z0-9_.-]+\.html)",
        payload,
    )
    unique_pairs: list[str] = []
    for left, right in pairs:
        mapping = f"{left} -> {right}"
        if mapping not in unique_pairs:
            unique_pairs.append(mapping)
    if not unique_pairs:
        return None
    preview = ", ".join(unique_pairs[:4])
    if len(unique_pairs) > 4:
        preview += ", ..."
    return f"Filename mappings confirmed: {preview}"


def _summarize_html_title_discovery(
    messages: list[Message],
    *,
    max_pairs: int = 4,
    tool_calls_by_id: dict[str, ToolCall] | None = None,
) -> str | None:
    if tool_calls_by_id is None:
        tool_calls_by_id = {
            tool_call.id: tool_call
            for message in messages
            for tool_call in message.tool_calls
        }

    confirmed_pairs: list[str] = []
    for message in messages:
        if message.role != Role.TOOL or _is_compacted_context_message(message.content):
            continue
        if any(result.is_error for result in message.tool_results):
            continue

        tool_call = next(
            (
                tool_calls_by_id.get(result.tool_call_id)
                for result in message.tool_results
                if result.tool_call_id in tool_calls_by_id
            ),
            None,
        )
        if tool_call is None or tool_call.name != "read":
            continue

        raw_path = tool_call.arguments.get("file_path")
        if not isinstance(raw_path, str):
            continue
        normalized_path = _normalize_path_candidate(raw_path) or raw_path
        if Path(normalized_path).name == "index.html" or "/chapters/" not in normalized_path:
            continue

        payload = "\n".join(
            result.content.strip()
            for result in message.tool_results
            if result.content.strip()
        ) or message.content
        title = _extract_html_title(payload)
        if not title:
            continue

        pair = f"{Path(normalized_path).name} = {title}"
        if pair not in confirmed_pairs:
            confirmed_pairs.append(pair)

    if not confirmed_pairs:
        return None

    preview = ", ".join(confirmed_pairs[:max_pairs])
    if len(confirmed_pairs) > max_pairs:
        preview += ", ..."
    return f"Chapter titles confirmed: {preview}"


def _extract_html_title(payload: str) -> str | None:
    for pattern in (
        r"<h1[^>]*>(.*?)</h1>",
        r"<title[^>]*>(.*?)</title>",
    ):
        match = re.search(pattern, payload, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        title = re.sub(r"<[^>]+>", " ", match.group(1))
        title = _collapse_inline_whitespace(html.unescape(title))
        if title:
            return title
    return None


def _collect_html_file_discovery_fact(
    messages: list[Message],
    *,
    tool_calls_by_id: dict[str, ToolCall],
) -> str | None:
    filenames: list[str] = []
    for message in messages:
        if message.role != Role.TOOL or _is_compacted_context_message(message.content):
            continue
        if any(result.is_error for result in message.tool_results):
            continue

        tool_name = _resolve_tool_name(
            message,
            tool_calls_by_id=tool_calls_by_id,
        )
        if tool_name not in {"glob", "bash"}:
            continue

        payload = "\n".join(
            result.content.strip()
            for result in message.tool_results
            if result.content.strip()
        ) or message.content
        matches = re.findall(r"([A-Za-z0-9_.-]+\.html)", payload)
        for name in matches:
            if name not in filenames:
                filenames.append(name)

    if len(filenames) < 3:
        return None

    preview = ", ".join(filenames[:6])
    if len(filenames) > 6:
        preview += ", ..."
    return f"Existing files include {preview}"


def _collect_html_verification_gap_fact(
    messages: list[Message],
    *,
    tool_calls_by_id: dict[str, ToolCall],
) -> str | None:
    gap = _summarize_latest_html_verification_gap(
        messages,
        tool_calls_by_id=tool_calls_by_id,
    )
    if not gap:
        return None
    return f"Verification gaps: {gap}"


def _summarize_latest_html_verification_gap(
    messages: list[Message],
    *,
    max_items: int = 2,
    tool_calls_by_id: dict[str, ToolCall] | None = None,
) -> str | None:
    if tool_calls_by_id is None:
        tool_calls_by_id = {
            tool_call.id: tool_call
            for message in messages
            for tool_call in message.tool_calls
        }

    for message in reversed(messages):
        if message.role != Role.TOOL or _is_compacted_context_message(message.content):
            continue
        if not any(result.is_error for result in message.tool_results):
            continue
        tool_name = _resolve_tool_name(
            message,
            tool_calls_by_id=tool_calls_by_id,
        )
        if tool_name != "bash":
            continue

        payload = "\n".join(
            result.content.strip()
            for result in message.tool_results
            if result.content.strip()
        ) or message.content
        gap = _extract_html_verification_gap(payload, max_items=max_items)
        if gap:
            return gap

    return None


def _extract_html_verification_gap(payload: str, *, max_items: int = 2) -> str | None:
    missing: list[str] = []
    mismatches: list[str] = []
    mode: str | None = None

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered == "missing links:":
            mode = "missing"
            continue
        if lowered == "title mismatches:":
            mode = "mismatch"
            continue
        if mode == "missing" and "->" in line:
            href = line.split("->", 1)[0].strip()
            if href and href not in missing:
                missing.append(href)
            continue
        if mode == "mismatch" and "!=" in line:
            if line not in mismatches:
                mismatches.append(line)

    parts: list[str] = []
    if missing:
        preview = ", ".join(missing[:max_items])
        if len(missing) > max_items:
            preview += ", ..."
        parts.append(f"missing TOC links {preview}")
    if mismatches:
        preview = ", ".join(mismatches[:max_items])
        if len(mismatches) > max_items:
            preview += ", ..."
        parts.append(f"title mismatches {preview}")
    return "; ".join(parts) if parts else None


def _summarize_html_file_discovery(payload: str) -> str | None:
    filenames = re.findall(r"([A-Za-z0-9_.-]+\.html)", payload)
    unique_names: list[str] = []
    for name in filenames:
        if name not in unique_names:
            unique_names.append(name)
    if len(unique_names) < 3:
        return None
    preview = ", ".join(unique_names[:6])
    if len(unique_names) > 6:
        preview += ", ..."
    return f"Existing files include {preview}"


def _resolve_tool_name(
    message: Message,
    *,
    tool_calls_by_id: dict[str, ToolCall],
) -> str | None:
    observed = _observed_tool_name(message.content)
    if observed:
        return observed

    for result in message.tool_results:
        tool_call = tool_calls_by_id.get(result.tool_call_id)
        if tool_call is not None:
            return tool_call.name
    return None


def _choose_target_path(
    messages: list[Message],
    *,
    current_task: str | None = None,
) -> str | None:
    candidates: Counter[str] = Counter()
    for message in messages:
        for tool_call in message.tool_calls:
            if tool_call.name not in {"read", "write", "edit", "patch"}:
                continue
            raw_path = tool_call.arguments.get("file_path")
            if not isinstance(raw_path, str):
                continue
            normalized = _normalize_path_candidate(raw_path)
            if not normalized:
                continue
            path_name = Path(normalized).name
            if path_name == "index.html":
                candidates[normalized] += 10
            elif path_name.endswith(".html") and "/chapters/" not in normalized:
                candidates[normalized] += 4

    if candidates:
        return candidates.most_common(1)[0][0]

    if not current_task:
        return None
    current_task_paths = extract_key_files([Message(role=Role.USER, content=current_task)], limit=3)
    for path in current_task_paths:
        if Path(path).name == "index.html":
            return path
    return current_task_paths[0] if current_task_paths else None
