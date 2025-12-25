"""Prompt templates for the agent."""

import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..context.project import ProjectContext


# Project-specific guidance templates
PROJECT_TIPS = {
    "python": """
## Python Project Tips

- Use the detected test command to run tests
- Check pyproject.toml or setup.py for project configuration
- Look for type hints and use mypy/pyright conventions if present
- Follow PEP 8 style unless the project uses a different style
""",
    "node": """
## Node.js Project Tips

- Check package.json for available scripts
- Look for tsconfig.json if TypeScript is used
- Check for ESLint/Prettier configs for style
- Dependencies are in node_modules (don't edit these)
""",
    "rust": """
## Rust Project Tips

- Use `cargo` for all build/test/run operations
- Check Cargo.toml for dependencies and features
- Look for clippy.toml or .clippy.conf for lint rules
- Run `cargo fmt` to format code
""",
    "go": """
## Go Project Tips

- Use `go` commands for build/test/run
- Check go.mod for module name and dependencies
- Follow standard Go project layout
- Run `go fmt` to format code
""",
}

PACKAGE_MANAGER_TIPS = {
    "uv": """
- This project uses **uv** for package management
- Use `uv run` to execute commands in the project environment
- Use `uv add` to add dependencies, `uv remove` to remove them
- The lockfile is `uv.lock`
""",
    "poetry": """
- This project uses **Poetry** for package management
- Use `poetry run` to execute commands in the virtualenv
- Use `poetry add` to add dependencies
- The lockfile is `poetry.lock`
""",
    "pipenv": """
- This project uses **Pipenv** for package management
- Use `pipenv run` to execute commands in the virtualenv
- Use `pipenv install` to add dependencies
- The lockfile is `Pipfile.lock`
""",
    "yarn": """
- This project uses **Yarn** for package management
- Use `yarn` commands instead of npm
- The lockfile is `yarn.lock`
""",
    "pnpm": """
- This project uses **pnpm** for package management
- Use `pnpm` commands instead of npm
- The lockfile is `pnpm-lock.yaml`
""",
}

TEST_FRAMEWORK_TIPS = {
    "pytest": """
- Tests use **pytest**
- Look for conftest.py for shared fixtures
- Use `-v` for verbose output, `-x` to stop on first failure
- Use `-k "pattern"` to run specific tests
""",
    "jest": """
- Tests use **Jest**
- Look for jest.config.js for configuration
- Use `--watch` for interactive mode
- Use `--coverage` to see coverage report
""",
    "vitest": """
- Tests use **Vitest**
- Configuration in vitest.config.ts or vite.config.ts
- Use `--ui` for interactive browser mode
""",
}


def get_project_specific_tips(context: "ProjectContext") -> str:
    """Generate project-specific tips based on detected context."""
    tips = []

    # Project type tips
    if context.project_type in PROJECT_TIPS:
        tips.append(PROJECT_TIPS[context.project_type])

    # Package manager tips
    if context.package_manager in PACKAGE_MANAGER_TIPS:
        tips.append(PACKAGE_MANAGER_TIPS[context.package_manager])

    # Test framework tips
    if context.test_framework in TEST_FRAMEWORK_TIPS:
        tips.append(TEST_FRAMEWORK_TIPS[context.test_framework])

    # Virtual environment tip
    if context.has_venv:
        if context.is_venv_active:
            tips.append(f"- Virtual environment `{context.venv_path}` is **active**")
        else:
            tips.append(f"- Virtual environment found at `{context.venv_path}` (not active)")
            if context.package_manager not in ("poetry", "uv", "pipenv"):
                tips.append(f"  - Activate with: `source {context.venv_path}/bin/activate`")

    return "\n".join(tips)


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


SYSTEM_PROMPT = """You are Loader, an AI coding agent running locally on the user's machine.

Current working directory: {cwd}

## CRITICAL INSTRUCTION: USE TOOLS, DO NOT DESCRIBE

You MUST use your tools to complete tasks. NEVER output code blocks for the user to copy.

WRONG (chatbot behavior - DO NOT DO THIS):
```
Here's how to create the file:
```bash
mkdir -p ~/Project/site
```
Save this to index.html:
```html
<html>...</html>
```
```

CORRECT (agent behavior - ALWAYS DO THIS):
I'll create the directory and file now.
[calls bash tool with: mkdir -p ~/Project/site]
[calls write tool with: file_path=~/Project/site/index.html, content=<html>...</html>]
Done. Created ~/Project/site/index.html.

## You Have These Tools - USE THEM

- `bash`: Execute shell commands (mkdir, git, npm, etc.)
- `write`: Create new files with content
- `edit`: Modify existing files
- `read`: Read file contents
- `glob`: Find files by pattern
- `grep`: Search file contents

## Rules

1. **EXECUTE, don't describe**: When asked to do something, USE TOOLS to do it immediately
2. **No code blocks for action**: Never show bash commands or file contents for the user to copy
3. **Read before edit**: Always read a file before modifying it
4. **Verify after action**: After making changes, confirm success
5. **Be concise**: Brief reasoning, then action, then short summary
6. **COMPLETE the task**: Don't stop after creating files - install deps, run tests, start servers
7. **Follow through**: If creating a project, fully initialize it (npm install, pip install, etc.)
8. **Demonstrate results**: Run what you created to prove it works

## Examples of Correct Behavior

User: "Create a hello.py file that prints hello world"
You: I'll create that file now.
[USE write tool: file_path="hello.py", content="print('hello world')"]
Created hello.py.

User: "Run the tests"
You: Running tests now.
[USE bash tool: command="pytest"]
Tests passed (or: 2 tests failed, here's the output...)

User: "Add a new function to utils.py"
You: Let me read the file first.
[USE read tool: file_path="utils.py"]
Now I'll add the function.
[USE edit tool: file_path="utils.py", old_string="...", new_string="..."]
Added the function to utils.py.

## What NOT To Do

- Do NOT say "you can run this command: ..."
- Do NOT say "create a file with this content: ..."
- Do NOT show code in markdown blocks for the user to copy
- Do NOT explain how to do something - just DO IT

You are an AGENT that EXECUTES tasks, not a chatbot that gives advice.
"""


REACT_SYSTEM_PROMPT = """You are Loader, an AI coding agent. You EXECUTE tasks using tools.

Current working directory: {cwd}

## CRITICAL: YOU MUST USE TOOLS

NEVER show code blocks for users to copy. ALWAYS use tools to execute actions.

WRONG - Do not do this:
"Here's the command to run: `mkdir project`"
"Create a file with this content: ```html...```"

CORRECT - Do this instead:
"Creating the directory now."
<tool_call>
{{"name": "bash", "arguments": {{"command": "mkdir project"}}}}
</tool_call>

## Tools Available

{tool_descriptions}

## How to Call Tools

Use this exact format:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

Wait for the result, then continue or finish.

## Rules

1. USE TOOLS to do things - never just describe
2. Read files before editing them
3. Be concise: brief intro, tool call, short summary
4. If a tool fails, try a different approach
5. COMPLETE the task fully - don't stop after one step
6. For projects: create files, install deps, then RUN to verify
7. Demonstrate that your work actually functions

## Examples

User: "Create a test.py file"
Assistant: Creating the file.
<tool_call>
{{"name": "write", "arguments": {{"file_path": "test.py", "content": "# test file"}}}}
</tool_call>

User: "List files in src/"
Assistant: Listing files.
<tool_call>
{{"name": "bash", "arguments": {{"command": "ls -la src/"}}}}
</tool_call>

User: "What's in config.json?"
Assistant: Reading the file.
<tool_call>
{{"name": "read", "arguments": {{"file_path": "config.json"}}}}
</tool_call>

Remember: You are an AGENT. Execute tasks, don't explain them.
"""


def build_system_prompt(
    tools: list[dict[str, Any]],
    use_react: bool = False,
    project_context: "str | ProjectContext | None" = None,
) -> str:
    """Build the system prompt with tool descriptions.

    Args:
        tools: List of tool schemas
        use_react: If True, use ReAct-style prompting
        project_context: Optional project context (string or ProjectContext object)

    Returns:
        Formatted system prompt
    """
    cwd = os.getcwd()

    if use_react:
        tool_descriptions = format_tool_descriptions(tools)
        prompt = REACT_SYSTEM_PROMPT.format(
            cwd=cwd,
            tool_descriptions=tool_descriptions,
        )
    else:
        prompt = SYSTEM_PROMPT.format(cwd=cwd)

    # Add project context if available
    if project_context:
        # Handle both string and ProjectContext
        if isinstance(project_context, str):
            prompt += f"\n\n## Project Context\n\n{project_context}"
        else:
            # It's a ProjectContext object
            prompt += f"\n\n## Project Context\n\n{project_context.to_prompt()}"

            # Add project-specific tips
            tips = get_project_specific_tips(project_context)
            if tips.strip():
                prompt += f"\n\n{tips}"

    return prompt
