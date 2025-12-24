"""Prompt templates for the agent."""

import os
from typing import Any


def format_tool_descriptions(tools: list[dict[str, Any]]) -> str:
    """Format tool schemas as text descriptions for ReAct prompting."""
    lines = []
    for tool in tools:
        name = tool["name"]
        desc = tool.get("description", "")
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])

        param_strs = []
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req = "(required)" if pname in required else "(optional)"
            param_strs.append(f"    - {pname}: {ptype} {req} - {pdesc}")

        param_block = "\n".join(param_strs) if param_strs else "    (no parameters)"
        lines.append(f"- {name}: {desc}\n  Parameters:\n{param_block}")

    return "\n\n".join(lines)


SYSTEM_PROMPT = """You are Loader, a helpful AI coding assistant running locally on the user's machine.

Current working directory: {cwd}

You have access to tools to help accomplish tasks. Use them when needed to read files, search code, run commands, or make changes.

## Guidelines

- Be concise but thorough
- When editing code, always read the file first to understand context
- Explain what you're doing and why
- If a tool call fails, explain the error and try an alternative approach
- When you're done, provide a clear summary of what you accomplished
- Do NOT output raw JSON or tool call syntax in your responses - just use the tools naturally
"""


REACT_SYSTEM_PROMPT = """You are Loader, a helpful AI coding assistant running locally on the user's machine.

Current working directory: {cwd}

You have access to the following tools:

{tool_descriptions}

## ReAct Format

You solve problems using the ReAct (Reasoning + Acting) pattern. For each step:

1. **Thought**: Reason about what to do next
2. **Action**: Use a tool (or provide final answer)
3. **Observation**: See the result (provided by the system)

Format your responses like this:

Thought: I need to understand the current code before making changes.
Action: <tool_call>
{{"name": "read", "arguments": {{"file_path": "src/main.py"}}}}
</tool_call>

Then wait for the Observation before continuing.

When you have enough information to answer, use:

Thought: I now have all the information I need.
Final Answer: [your response to the user]

## Guidelines

- Always think before acting
- Read files before editing them
- Be concise but thorough
- If something fails, reason about why and try alternatives
"""


def build_system_prompt(
    tools: list[dict[str, Any]],
    use_react: bool = False,
) -> str:
    """Build the system prompt with tool descriptions.

    Args:
        tools: List of tool schemas
        use_react: If True, use ReAct-style prompting

    Returns:
        Formatted system prompt
    """
    cwd = os.getcwd()

    if use_react:
        # ReAct mode needs tool descriptions in the prompt
        tool_descriptions = format_tool_descriptions(tools)
        return REACT_SYSTEM_PROMPT.format(
            cwd=cwd,
            tool_descriptions=tool_descriptions,
        )
    else:
        # Native mode - tools are passed via API, not in prompt
        return SYSTEM_PROMPT.format(cwd=cwd)
