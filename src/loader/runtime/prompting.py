"""Typed system-prompt builder for Loader runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..context.project import ProjectContext

PROJECT_TIPS = {
    "python": """
## Python Project Tips

- Use the detected test command to run tests
- Check `pyproject.toml` or `setup.py` for project configuration
- Look for type hints and use `mypy`/`pyright` conventions if present
- Follow PEP 8 style unless the project uses a different style
""",
    "node": """
## Node.js Project Tips

- Check `package.json` for available scripts
- Look for `tsconfig.json` if TypeScript is used
- Check for ESLint/Prettier configs for style
- Dependencies are in `node_modules` (do not edit these)
""",
    "rust": """
## Rust Project Tips

- Use `cargo` for all build/test/run operations
- Check `Cargo.toml` for dependencies and features
- Look for `clippy.toml` or `.clippy.conf` for lint rules
- Run `cargo fmt` to format code
""",
    "go": """
## Go Project Tips

- Use `go` commands for build/test/run operations
- Check `go.mod` for module name and dependencies
- Follow standard Go project layout
- Run `go fmt` to format code
""",
}

PACKAGE_MANAGER_TIPS = {
    "uv": """
- This project uses **uv** for package management
- Use `uv run` to execute commands in the project environment
- Use `uv add` to add dependencies and `uv remove` to remove them
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
- Look for `conftest.py` for shared fixtures
- Use `-v` for verbose output and `-k "pattern"` for focused runs
""",
    "jest": """
- Tests use **Jest**
- Look for `jest.config.js` for configuration
- Use `--watch` for interactive mode
- Use `--coverage` to see coverage
""",
    "vitest": """
- Tests use **Vitest**
- Configuration is usually in `vitest.config.ts` or `vite.config.ts`
- Use `--ui` for interactive browser mode
""",
}

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
- Be explicit about file touchpoints, order of work, risks, acceptance criteria,
  and verification commands
- Prefer concrete, repository-grounded plans over generic checklists
""",
    "execute": """
## Execute Mode
- Use tools directly to perform the task
- Read relevant files before editing them
- Keep `TodoWrite` current for multi-step work when progress tracking matters
- Concise reporting is fine, and numbered lists are allowed when they
  communicate plan or evidence clearly
""",
    "verify": """
## Verify Mode
- Run the planned verification commands and capture evidence
- Do not declare the task complete while any verification step is failing
- Report concrete pass/fail evidence rather than vague confidence
""",
}

SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "# Dynamic Runtime Context"


@dataclass(slots=True)
class PromptSection:
    """One named prompt section."""

    name: str
    body: str
    dynamic: bool = False

    def render(self) -> str:
        """Render one section as markdown."""

        return f"# {self.name}\n{self.body.strip()}"


@dataclass(slots=True)
class SystemPromptBuildResult:
    """Rendered prompt text plus operator-visible metadata."""

    content: str
    prompt_format: str
    section_names: list[str] = field(default_factory=list)
    dynamic_section_names: list[str] = field(default_factory=list)


def build_system_prompt_result(
    *,
    tools: list[dict[str, Any]],
    use_react: bool = False,
    project_context: str | ProjectContext | None = None,
    workflow_mode: str = "execute",
    permission_mode: str = "workspace-write",
    cwd: Path | str | None = None,
    current_task: str | None = None,
) -> SystemPromptBuildResult:
    """Build the runtime system prompt and return prompt metadata."""

    resolved_cwd = str(Path(cwd or ".").expanduser().resolve())
    prompt_format = "react" if use_react else "native"
    sections: list[PromptSection] = [
        PromptSection(
            name="Introduction",
            body=(
                "You are Loader, an AI coding agent. Help the user with software "
                "engineering work by following the runtime contract, using tools "
                "faithfully, and reporting outcomes honestly."
            ),
        ),
        PromptSection(
            name="Tools Available",
            body=format_tool_descriptions(tools),
        ),
        PromptSection(
            name="Tool Call Format",
            body=_tool_call_instructions(use_react),
        ),
        PromptSection(
            name="Rules",
            body=(
                "1. Follow the active workflow mode rather than improvising a different one\n"
                "2. Use tools or concise prose directly instead of narrating fake tool use\n"
                "3. Use the write tool for files rather than pasting long code blocks\n"
                "4. Keep responses grounded in repository evidence and verification output\n"
                "5. If verification fails or was not run, say so explicitly"
            ),
        ),
    ]

    dynamic_sections = [
        PromptSection(
            name="Runtime Config",
            body="\n".join(
                [
                    f"- Current directory: `{resolved_cwd}`",
                    f"- Permission mode: `{permission_mode}`",
                    f"- Tool call format: `{prompt_format}`",
                ]
            ),
            dynamic=True,
        ),
        PromptSection(
            name="Workflow Context",
            body="\n".join(
                line
                for line in [
                    f"- Active workflow mode: `{workflow_mode}`",
                    f"- Current task: {current_task}" if current_task else "",
                ]
                if line
            ),
            dynamic=True,
        ),
        PromptSection(
            name="Mode Guidance",
            body=MODE_GUIDANCE.get(workflow_mode, MODE_GUIDANCE["execute"]),
            dynamic=True,
        ),
    ]

    dynamic_sections.extend(_project_sections(project_context))
    rendered = _render_sections(sections, dynamic_sections)
    return SystemPromptBuildResult(
        content=rendered,
        prompt_format=prompt_format,
        section_names=[section.name for section in sections + dynamic_sections],
        dynamic_section_names=[
            section.name for section in dynamic_sections if section.dynamic
        ],
    )


def build_system_prompt(**kwargs: Any) -> str:
    """Compatibility wrapper returning only the rendered prompt text."""

    return build_system_prompt_result(**kwargs).content


def format_tool_descriptions(tools: list[dict[str, Any]]) -> str:
    """Format tool schemas as readable markdown."""

    if not tools:
        return "No tools available for this turn."

    lines = []
    for tool in tools:
        name = tool["name"]
        description = tool.get("description", "")
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])

        parameter_lines = []
        for param_name, param_info in params.items():
            param_type = param_info.get("type", "any")
            param_description = param_info.get("description", "")
            requirement = "required" if param_name in required else "optional"
            parameter_lines.append(
                f"  - `{param_name}`: {param_type} ({requirement}) - {param_description}"
            )

        if not parameter_lines:
            parameter_lines.append("  - no parameters")

        lines.append(
            "\n".join(
                [
                    f"- `{name}`: {description}",
                    "  Parameters:",
                    *parameter_lines,
                ]
            )
        )
    return "\n\n".join(lines)


def get_project_specific_tips(context: ProjectContext) -> str:
    """Generate project-specific guidance for the prompt builder."""

    tips = []
    if context.project_type in PROJECT_TIPS:
        tips.append(PROJECT_TIPS[context.project_type])
    if context.package_manager in PACKAGE_MANAGER_TIPS:
        tips.append(PACKAGE_MANAGER_TIPS[context.package_manager])
    if context.test_framework in TEST_FRAMEWORK_TIPS:
        tips.append(TEST_FRAMEWORK_TIPS[context.test_framework])
    if context.has_venv:
        if context.is_venv_active:
            tips.append(f"- Virtual environment `{context.venv_path}` is **active**")
        else:
            tips.append(f"- Virtual environment found at `{context.venv_path}` (inactive)")
    return "\n".join(tips)


def _render_sections(
    static_sections: list[PromptSection],
    dynamic_sections: list[PromptSection],
) -> str:
    rendered_sections = [section.render() for section in static_sections]
    if dynamic_sections:
        rendered_sections.append(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        rendered_sections.extend(section.render() for section in dynamic_sections)
    return "\n\n".join(rendered_sections)


def _project_sections(
    project_context: str | ProjectContext | None,
) -> list[PromptSection]:
    if not project_context:
        return []

    if isinstance(project_context, str):
        return [
            PromptSection(
                name="Project Context",
                body=project_context,
                dynamic=True,
            )
        ]

    sections = [
        PromptSection(
            name="Project Context",
            body=project_context.to_prompt(),
            dynamic=True,
        )
    ]
    tips = get_project_specific_tips(project_context)
    if tips.strip():
        sections.append(
            PromptSection(
                name="Project Tips",
                body=tips,
                dynamic=True,
            )
        )
    return sections


def _tool_call_instructions(use_react: bool) -> str:
    if use_react:
        return "\n".join(
            [
                "Wrap tool calls in `<tool_call>` tags using JSON arguments.",
                "",
                "Example:",
                "```xml",
                "<tool_call>",
                '{"name": "bash", "arguments": {"command": "pwd"}}',
                "</tool_call>",
                "```",
            ]
        )

    return "\n".join(
            [
                "Call tools directly using bracket syntax.",
                "",
                "Examples:",
                '- `[bash: command="pwd"]`',
                '- `[write: file_path="hello.py", content="print(\'hello\')"]`',
                (
                    '- `[TodoWrite: todos=[{content="Run tests", '
                    'active_form="Running tests", status="in_progress"}]]`'
                ),
            ]
        )
