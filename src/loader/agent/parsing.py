"""Parsing utilities for tool calls from LLM output."""

import json
import re
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
    # Try common key names for arguments
    for key in ["arguments", "parameters", "args", "params"]:
        if key in data:
            val = data[key]
            return val if isinstance(val, dict) else {}
    return {}


def _parse_bracket_args(args_str: str) -> dict:
    """Parse arguments from bracketed tool call format.

    Handles formats like:
        file_path=/tmp/test.txt, content="hello world"
        command="ls -la"
        file_path="test.py", old_string="foo", new_string="bar"

    Args:
        args_str: The arguments string (everything after "tool with:" or "tool:")

    Returns:
        Dictionary of parsed arguments
    """
    args = {}

    # Pattern to match key=value pairs where value can be:
    # - quoted string (single or double quotes)
    # - unquoted value (until comma or end)
    pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|([^,\]]+?))\s*(?:,|$)'

    for match in re.finditer(pattern, args_str):
        key = match.group(1)
        # Value is in one of the capture groups (2=double quoted, 3=single quoted, 4=unquoted)
        value = match.group(2) or match.group(3) or match.group(4)
        if value is not None:
            value = value.strip()
            args[key] = value

    return args


def _extract_json_tool_calls(text: str) -> tuple[list[ToolCall], list[tuple[int, int]]]:
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
            if (
                isinstance(name, str)
                and name
                and any(key in data for key in ("arguments", "parameters", "args", "params"))
            ):
                tool_calls.append(ToolCall(
                    id=f"call_{len(tool_calls)}",
                    name=name,
                    arguments=arguments,
                ))
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


def parse_tool_calls(text: str) -> ParsedResponse:
    """Parse tool calls from LLM text output.

    Supports multiple formats:
    1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. {"name": "...", "arguments": {...}} (bare JSON)
    3. Final Answer: ... (indicates completion)

    Args:
        text: Raw LLM output

    Returns:
        ParsedResponse with extracted tool calls and cleaned content
    """
    tool_calls: list[ToolCall] = []
    content = text
    is_final = False
    final_content = ""

    # Check for Final Answer (ReAct pattern)
    final_match = re.search(
        r"Final Answer:\s*(.+?)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        is_final = True
        final_content = final_match.group(1).strip()

    # Pattern 1: <tool_call>...</tool_call> blocks (also handle malformed </tool_call> at start)
    tool_call_pattern = r"(?:</tool_call>\s*)?<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(tool_call_pattern, text, re.DOTALL)

    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = _extract_arguments(data)

            if name:
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=name,
                    arguments=arguments,
                ))
        except json.JSONDecodeError:
            continue

    # Remove tool call blocks from content (including malformed ones)
    content = re.sub(r"</tool_call>\s*", "", content)
    content = re.sub(r"<tool_call>\s*\{[^}]*\}\s*</tool_call>", "", content, flags=re.DOTALL)

    # Pattern 2: Bare JSON if no tool_call tags found
    if not tool_calls:
        tool_calls, spans = _extract_json_tool_calls(text)
        if tool_calls and not is_final:
            content = _remove_spans(content, spans)

    # Pattern 3: Bracketed format [calls/USE tool with/: key=value, ...]
    # Examples:
    #   [calls write tool with: file_path=/tmp/test.txt, content="hello"]
    #   [USE bash tool: command="ls -la"]
    if not tool_calls:
        bracket_pattern = r'\[(?:calls|USE)\s+(\w+)\s+tool(?:\s+with)?[:\s]+([^\]]+)\]'
        for i, (name, args_str) in enumerate(re.findall(bracket_pattern, text, re.IGNORECASE)):
            args = _parse_bracket_args(args_str)
            if args:
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=name.lower(),
                    arguments=args,
                ))
        # Remove bracketed tool calls from content
        if tool_calls:
            content = re.sub(bracket_pattern, "", content, flags=re.IGNORECASE)

    if is_final:
        content = final_content

    # Clean up content
    content = content.strip()

    # Remove "Thought:" and "Action:" labels if present (ReAct artifacts)
    content = re.sub(r"^Thought:\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^Action:\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^Observation:\s*", "", content, flags=re.MULTILINE)

    # Remove any remaining orphaned tool_call tags
    content = re.sub(r"</?tool_call>", "", content)

    # Clean up multiple newlines
    content = re.sub(r"\n{3,}", "\n\n", content)

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        is_final_answer=is_final,
    )


def format_tool_result(tool_name: str, result: str, is_error: bool = False) -> str:
    """Format a tool result for inclusion in conversation.

    Args:
        tool_name: Name of the tool that was executed
        result: The tool's output
        is_error: Whether the result is an error

    Returns:
        Formatted result string
    """
    prefix = "Error" if is_error else "Result"
    return f"Observation [{tool_name}]: {prefix}: {result}"
