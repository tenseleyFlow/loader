"""Prompt templates for the agent."""

import os
from typing import TYPE_CHECKING, Any

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


MODE_GUIDANCE = {
    "clarify": """
## Clarify Mode
- Ask exactly one focused question with `AskUserQuestion`
- Clarify intent, outcome, scope, or boundaries before proposing solutions
- Do not start coding or writing patch plans yet
- Keep the question high-leverage and brief
""",
    "plan": """
## Plan Mode
- Produce persistent implementation and verification planning artifacts
- Do not start writing code in this mode
- Be explicit about file touchpoints, order of work, risks, acceptance criteria, and verification commands
- Prefer concrete, repository-grounded plans over generic checklists
""",
    "execute": """
## Execute Mode
- Use tools directly to perform the task
- Read relevant files before editing them
- Keep `TodoWrite` current for multi-step work when progress tracking matters
- Concise reporting is fine, and numbered lists are allowed when they communicate plan or evidence clearly
""",
    "verify": """
## Verify Mode
- Run the planned verification commands and capture evidence
- Do not declare the task complete while any verification step is failing
- Report concrete pass/fail evidence rather than vague confidence
""",
}


SYSTEM_PROMPT = """You are Loader, an AI coding agent.

Current directory: {cwd}

## Tools Available
{tool_descriptions}

## How to Use Tools
Output a tool call in this format:
[tool: param="value", param2="value2"]

## Examples
[bash: command="mkdir project"]
[write: file_path="hello.py", content="print('hello')"]
[read: file_path="config.json"]
[edit: file_path="app.py", old_string="old", new_string="new"]
[TodoWrite: todos=[{{content="Run tests", active_form="Running tests", status="in_progress"}}]]
[AskUserQuestion: question="Which path matters more?", options=["Speed", "Correctness"]]

## Active Workflow Mode
{workflow_mode}

{mode_guidance}

## Rules
1. Follow the active workflow mode rather than improvising a different one
2. Use tools or concise prose directly instead of narrating fake tool use
3. Use the write tool for files rather than pasting long code blocks
4. Keep responses grounded in repository evidence and verification output
"""


REACT_SYSTEM_PROMPT = """You are Loader, an AI coding agent.

Current directory: {cwd}

## Tools Available
{tool_descriptions}

## How to Use Tools
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

## Examples
<tool_call>
{{"name": "bash", "arguments": {{"command": "mkdir project"}}}}
</tool_call>

<tool_call>
{{"name": "write", "arguments": {{"file_path": "hello.py", "content": "print('hello')"}}}}
</tool_call>

<tool_call>
{{"name": "read", "arguments": {{"file_path": "config.json"}}}}
</tool_call>

## Active Workflow Mode
{workflow_mode}

{mode_guidance}

## Rules
1. Follow the active workflow mode rather than improvising a different one
2. Use tools or concise prose directly instead of narrating fake tool use
3. Use the write tool for files rather than pasting long code blocks
4. Keep responses grounded in repository evidence and verification output
"""


def build_system_prompt(
    tools: list[dict[str, Any]],
    use_react: bool = False,
    project_context: "str | ProjectContext | None" = None,
    workflow_mode: str = "execute",
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
    tool_descriptions = format_tool_descriptions(tools)
    mode_guidance = MODE_GUIDANCE.get(workflow_mode, MODE_GUIDANCE["execute"])

    if use_react:
        prompt = REACT_SYSTEM_PROMPT.format(
            cwd=cwd,
            tool_descriptions=tool_descriptions,
            workflow_mode=workflow_mode,
            mode_guidance=mode_guidance,
        )
    else:
        prompt = SYSTEM_PROMPT.format(
            cwd=cwd,
            tool_descriptions=tool_descriptions,
            workflow_mode=workflow_mode,
            mode_guidance=mode_guidance,
        )

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
