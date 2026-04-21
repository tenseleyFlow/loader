"""Runtime-owned parsing utilities for tool calls and tool results."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable
from dataclasses import dataclass

from ..llm.base import ToolCall


@dataclass
class ParsedResponse:
    """Parsed LLM response with potential tool calls."""

    content: str
    tool_calls: list[ToolCall]
    is_final_answer: bool = False


_TOOL_NAME_ALIASES = {
    "bashcommand": "bash",
    "editfile": "edit",
    "globfile": "glob",
    "globfiles": "glob",
    "patchfile": "patch",
    "readfile": "read",
    "writefile": "write",
}


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


def _normalized_tool_key(name: str) -> str:
    """Collapse separators and case so near-miss tool names can still match."""

    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def _normalized_allowed_tool_map(tool_names: dict[str, str] | None) -> dict[str, str] | None:
    """Build a separator-insensitive tool-name map."""

    if tool_names is None:
        return None
    return {
        _normalized_tool_key(canonical_name): canonical_name
        for canonical_name in tool_names.values()
    }


def _canonicalize_tool_name(
    name: str,
    tool_names: dict[str, str] | None,
    *,
    lowercase_default: bool = False,
) -> str | None:
    """Return the canonical tool name for parsing, if allowed."""

    if tool_names is None:
        return name.lower() if lowercase_default else name

    direct_match = tool_names.get(name.casefold())
    if direct_match is not None:
        return direct_match

    normalized_allowed = _normalized_allowed_tool_map(tool_names)
    if normalized_allowed is None:
        return None

    normalized_name = _normalized_tool_key(name)
    normalized_match = normalized_allowed.get(normalized_name)
    if normalized_match is not None:
        return normalized_match

    alias_target = _TOOL_NAME_ALIASES.get(normalized_name)
    if alias_target is None:
        return None

    direct_alias_match = tool_names.get(alias_target.casefold())
    if direct_alias_match is not None:
        return direct_alias_match
    return normalized_allowed.get(_normalized_tool_key(alias_target))


def canonicalize_tool_name(
    name: str,
    *,
    allowed_tool_names: Iterable[str] | None = None,
    lowercase_default: bool = False,
) -> str | None:
    """Public helper for backend/native tool-call normalization."""

    return _canonicalize_tool_name(
        name,
        _tool_name_map(allowed_tool_names),
        lowercase_default=lowercase_default,
    )


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


def _extract_function_tag_tool_calls(
    text: str,
    tool_names: dict[str, str] | None = None,
) -> tuple[list[ToolCall], list[tuple[int, int]]]:
    """Extract tool calls from ``<function=name><parameter=k>v</parameter></function>`` format.

    Several Ollama model renderers (qwen3-coder, qwen2) emit this format
    when the tool schema is too large for the native tool-calling path.
    """

    pattern = r"<function=(\w+)>(.*?)</function>"
    param_pattern = r"<parameter=(\w+)>\s*(.*?)\s*</parameter>"
    tool_calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []

    for match in re.finditer(pattern, text, re.DOTALL):
        raw_name = match.group(1)
        canonical = _canonicalize_tool_name(raw_name, tool_names)
        if canonical is None:
            continue
        body = match.group(2)
        arguments: dict[str, str] = {}
        for param_match in re.finditer(param_pattern, body, re.DOTALL):
            arguments[param_match.group(1)] = param_match.group(2).strip()
        if arguments:
            tool_calls.append(
                ToolCall(
                    id=f"call_{len(tool_calls)}",
                    name=canonical,
                    arguments=arguments,
                )
            )
            spans.append(match.span())

    return tool_calls, spans


def _parse_fenced_tool_arguments(
    tool_name: str,
    command_line: str,
) -> dict[str, str] | None:
    """Convert one simple fenced command line into Loader tool arguments."""

    try:
        argv = shlex.split(command_line)
    except ValueError:
        return None
    if len(argv) < 2:
        return None

    payload = command_line[len(argv[0]) :].strip()
    if tool_name == "read" and len(argv) == 2:
        return {"file_path": argv[1]}
    if tool_name == "glob" and len(argv) == 2:
        return {"pattern": argv[1]}
    if tool_name == "bash" and payload:
        return {"command": payload}
    return None


def _extract_fenced_command_tool_calls(
    text: str,
    tool_names: dict[str, str] | None = None,
) -> tuple[list[ToolCall], list[tuple[int, int]]]:
    """Recover simple one-line fenced tool commands from local-model prose."""

    fence_pattern = r"```(?:[^\n`]*)\n(.*?)```"
    tool_calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []

    for match in re.finditer(fence_pattern, text, re.DOTALL):
        body = match.group(1).strip()
        if not body or "\n" in body:
            continue
        raw_name = body.split(None, 1)[0]
        canonical_name = _canonicalize_tool_name(
            raw_name,
            tool_names,
            lowercase_default=True,
        )
        if canonical_name is None:
            continue
        arguments = _parse_fenced_tool_arguments(canonical_name, body)
        if not arguments:
            continue
        tool_calls.append(
            ToolCall(
                id=f"call_{len(tool_calls)}",
                name=canonical_name,
                arguments=arguments,
            )
        )
        spans.append(match.span())

    return tool_calls, spans


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

    # Try <function=name> format first (qwen3-coder fallback)
    func_calls, func_spans = _extract_function_tag_tool_calls(text, tool_names)
    if func_calls:
        tool_calls = func_calls
        content = _remove_spans(content, func_spans)

    tool_call_pattern = r"(?:</tool_call>\s*)?<tool_call>\s*(\{.*?\})\s*</tool_call>"
    xml_spans: list[tuple[int, int]] = []
    if not tool_calls:
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

    if not tool_calls:
        fenced_calls, fenced_spans = _extract_fenced_command_tool_calls(
            text,
            tool_names,
        )
        if fenced_calls:
            tool_calls = fenced_calls
            content = _remove_spans(content, fenced_spans)

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
