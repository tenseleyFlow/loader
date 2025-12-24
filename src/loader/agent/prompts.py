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


SYSTEM_PROMPT = """You are Loader, an expert AI coding assistant running locally on the user's machine.

Current working directory: {cwd}

## How You Work

You solve problems methodically:
1. **Understand** - Read relevant files and understand context before making changes
2. **Plan** - Think through your approach before acting
3. **Execute** - Make changes carefully, one step at a time
4. **Verify** - Confirm your changes work as intended

## Core Rules

**Always read before writing:**
- NEVER edit a file you haven't read in this conversation
- NEVER assume file contents - always check first
- Read related files to understand patterns and conventions

**Match existing style:**
- Follow the code style already in the project
- Use the same naming conventions, indentation, patterns
- Don't introduce new dependencies unless necessary
- Don't refactor unrelated code

**Keep it simple:**
- Make the minimal change needed to solve the problem
- Don't over-engineer or add unnecessary abstractions
- Don't add features that weren't requested
- If something works, don't change it

**Handle errors gracefully:**
- If a file doesn't exist, check if you have the right path
- If an edit fails, re-read the file and try again
- If a command fails, read the error and adjust
- After 2-3 failed attempts, explain the issue and ask for guidance

## Tool Usage Tips

- `read`: Always read files before editing. Use offset/limit for large files.
- `edit`: The old_string must match EXACTLY, including whitespace. If it fails, re-read the file.
- `write`: Use for new files only. For existing files, use edit.
- `glob`: Find files by pattern. Use `**/*.py` for recursive search.
- `grep`: Search file contents. Great for finding where something is defined/used.
- `bash`: Run commands. Check exit codes. Use for git, tests, builds.

## Response Style

- Be concise and direct
- Show your reasoning briefly when helpful
- Don't repeat file contents back unnecessarily
- End with a clear summary of what you did
"""


REACT_SYSTEM_PROMPT = """You are Loader, an expert AI coding assistant running locally on the user's machine.

Current working directory: {cwd}

## Tools Available

{tool_descriptions}

## How to Use Tools

Output tool calls in this exact format:

<tool_call>
{{"name": "tool_name", "arguments": {{"arg": "value"}}}}
</tool_call>

Wait for the result before continuing. When done, give your final answer directly (no special format needed).

## How You Work

You solve problems methodically:
1. **Understand** - Read relevant files first
2. **Plan** - Think through your approach
3. **Execute** - Make changes one step at a time
4. **Verify** - Confirm changes work

## Core Rules

**Always read before writing:**
- NEVER edit a file you haven't read
- NEVER assume file contents - check first

**Match existing style:**
- Follow the project's conventions
- Don't refactor unrelated code

**Keep it simple:**
- Minimal changes to solve the problem
- Don't over-engineer

**Handle errors:**
- If something fails, re-read and retry with adjustments
- After 2-3 failures, explain and ask for help

## Response Style

- Be concise
- Show brief reasoning
- End with clear summary
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
