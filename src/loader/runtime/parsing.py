"""Runtime-owned parsing utilities for tool calls and tool results."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..llm.base import ToolCall


@dataclass
class ParsedResponse:
    """Parsed LLM response with potential tool calls."""

    content: str
    tool_calls: list[ToolCall]
    is_final_answer: bool = False


def _extract_arguments(data: dict) -> dict:
    """Extract arguments from tool call data, handling various key names."""

    for key in ["arguments", "parameters", "args", "params"]:
        if key in data:
            value = data[key]
            return value if isinstance(value, dict) else {}
    return {}


def _parse_bracket_args(args_str: str) -> dict:
    """Parse arguments from bracketed tool call format."""

    args = {}
    pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|([^,\]]+?))\s*(?:,|$)'

    for match in re.finditer(pattern, args_str):
        key = match.group(1)
        value = match.group(2) or match.group(3) or match.group(4)
        if value is not None:
            args[key] = value.strip()

    return args


def _tool_name_map(allowed_tool_names: Iterable[str] | None) -> dict[str, str] | None:
    """Build a case-insensitive map of allowed tool names."""

    if allowed_tool_names is None:
        return None
    return {name.casefold(): name for name in allowed_tool_names}


def _canonicalize_tool_name(
    name: str,
    tool_names: dict[str, str] | None,
    *,
    lowercase_default: bool = False,
) -> str | None:
    """Return the canonical tool name for parsing, if allowed."""

    if tool_names is None:
        return name.lower() if lowercase_default else name
    return tool_names.get(name.casefold())


def _extract_json_tool_calls(
    text: str,
    tool_names: dict[str, str] | None = None,
) -> tuple[list[ToolCall], list[tuple[int, int]]]:
    """Extract bare JSON tool calls, including nested argument structures."""

    decoder = json.JSONDecoder()
    tool_calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    index = 0

    while index < len(text):
        start = text.find("{", index)
        if start == -1:
            break

        try:
            data, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue

        end = start + consumed
        if isinstance(data, dict):
            name = data.get("name", "")
            arguments = _extract_arguments(data)
            canonical_name = (
                _canonicalize_tool_name(name, tool_names)
                if isinstance(name, str)
                else None
            )
            if (
                canonical_name
                and any(key in data for key in ("arguments", "parameters", "args", "params"))
            ):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{len(tool_calls)}",
                        name=canonical_name,
                        arguments=arguments,
                    )
                )
                spans.append((start, end))
                index = end
                continue

        index = start + 1

    return tool_calls, spans


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove non-overlapping spans from text."""

    if not spans:
        return text

    parts: list[str] = []
    previous_end = 0
    for start, end in spans:
        parts.append(text[previous_end:start])
        previous_end = end
    parts.append(text[previous_end:])
    return "".join(parts)


def parse_tool_calls(
    text: str,
    *,
    allowed_tool_names: Iterable[str] | None = None,
) -> ParsedResponse:
    """Parse tool calls from LLM text output."""

    tool_calls: list[ToolCall] = []
    content = text
    is_final = False
    final_content = ""
    tool_names = _tool_name_map(allowed_tool_names)

    final_match = re.search(
        r"Final Answer:\s*(.+?)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        is_final = True
        final_content = final_match.group(1).strip()

    tool_call_pattern = r"(?:</tool_call>\s*)?<tool_call>\s*(\{.*?\})\s*</tool_call>"
    xml_spans: list[tuple[int, int]] = []
    for index, match in enumerate(re.finditer(tool_call_pattern, text, re.DOTALL)):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

        name = data.get("name", "")
        arguments = _extract_arguments(data)
        canonical_name = (
            _canonicalize_tool_name(name, tool_names)
            if isinstance(name, str)
            else None
        )
        if canonical_name:
            tool_calls.append(
                ToolCall(
                    id=f"call_{index}",
                    name=canonical_name,
                    arguments=arguments,
                )
            )
            xml_spans.append(match.span())

    content = _remove_spans(content, xml_spans)

    if not tool_calls:
        tool_calls, spans = _extract_json_tool_calls(text, tool_names)
        if tool_calls and not is_final:
            content = _remove_spans(content, spans)

    if not tool_calls:
        bracket_pattern = r'\[(?:calls|USE)\s+(\w+)\s+tool(?:\s+with)?[:\s]+([^\]]+)\]'
        for index, (name, args_str) in enumerate(
            re.findall(bracket_pattern, text, re.IGNORECASE)
        ):
            canonical_name = _canonicalize_tool_name(
                name,
                tool_names,
                lowercase_default=True,
            )
            if canonical_name is None:
                continue
            args = _parse_bracket_args(args_str)
            if args:
                tool_calls.append(
                    ToolCall(
                        id=f"call_{index}",
                        name=canonical_name,
                        arguments=args,
                    )
                )
        if tool_calls:
            content = re.sub(bracket_pattern, "", content, flags=re.IGNORECASE)

    if is_final:
        content = final_content

    content = content.strip()
    content = re.sub(r"^Thought:\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^Action:\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^Observation:\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"</?tool_call>", "", content)
    content = re.sub(r"\n{3,}", "\n\n", content)

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        is_final_answer=is_final,
    )


def format_tool_result(tool_name: str, result: str, is_error: bool = False) -> str:
    """Format a tool result for inclusion in conversation."""

    prefix = "Error" if is_error else "Result"
    return f"Observation [{tool_name}]: {prefix}: {result}"
