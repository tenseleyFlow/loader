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

    # Check for Final Answer (ReAct pattern)
    final_match = re.search(
        r"Final Answer:\s*(.+?)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if final_match:
        is_final = True
        # Extract just the final answer as content
        content = final_match.group(1).strip()

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
        # More flexible pattern that handles both "arguments" and "parameters"
        bare_json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"(?:arguments|parameters)"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        for i, (name, args_str) in enumerate(re.findall(bare_json_pattern, text)):
            try:
                args = json.loads(args_str)
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=name,
                    arguments=args,
                ))
            except json.JSONDecodeError:
                continue
        # Remove bare JSON tool calls from content
        if tool_calls:
            content = re.sub(bare_json_pattern, "", content)

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
