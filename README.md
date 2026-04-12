# Loader

Loader is a local-first coding assistant that runs against Ollama and drives a tool-using agent loop from the terminal or a Textual TUI.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- a running Ollama server on `http://localhost:11434`
- at least one pulled Ollama model (see [MODELS.md](MODELS.md))

## Install

```bash
# Global install — use loader from any directory
uv tool install git+https://github.com/tenseleyFlow/loader
loader "write a hello world program"

# Or install from a local clone
git clone https://github.com/tenseleyFlow/loader
cd loader
uv tool install -e .
```

## Development setup

```bash
uv sync --extra dev

uv run loader                          # TUI mode
uv run loader --no-tui                 # terminal mode
uv run loader --select-model           # pick an Ollama model

uv run pytest                          # run all tests
uv run pytest tests/test_foo.py -q     # single file
uv run ruff check src tests            # lint
uv run mypy src                        # type check (strict)
```

## How it works

Loader sends your prompt to a local Ollama model with tool schemas attached. The model calls tools (read, write, edit, bash, glob, grep, git) to complete the task. A typed turn loop drives the agent cycle:

1. **Prepare** — detect project context, build system prompt, set workflow mode
2. **Assistant** — stream the model response, extract tool calls
3. **Tools** — execute the tool batch, record results to session
4. **Completion** — check definition-of-done, run verification if needed
5. **Repeat** or **finalize** based on whether the task is complete

The TUI shows tool calls with previews, an approval bar for writes outside the workspace, streaming output, and a status line with session state.

## Key options

```
--permission-mode    read-only | workspace-write (default) | danger-full-access | prompt | allow
--select-model       choose from installed Ollama models
--plan               start in plan mode (outline before coding)
--clarify            start in clarify mode (ask questions first)
--react              force text-based tool calling (for models without native support)
--ctx N              context window size (default 8192)
```

## Repository layout

- `src/loader/runtime/` — turn engine, tool execution, verification, workflow routing
- `src/loader/tools/` — tool implementations (file, shell, search, git, workflow)
- `src/loader/llm/` — Ollama backend with native tool calling and streaming
- `src/loader/ui/` — Textual TUI with tool widgets, approval bar, status line
- `src/loader/cli/` — Click CLI entry point
- `tests/` — 416 deterministic tests with scripted backend harness
- `.docs/` — sprint planning, parity checkpoints, architecture analysis

## Documentation

- [MODELS.md](MODELS.md) — recommended Ollama models
- [.docs/REPORT.md](.docs/REPORT.md) — deep analysis vs reference implementations
- [.docs/PARITY.md](.docs/PARITY.md) — runtime feature inventory
- [.docs/sprints/](.docs/sprints/index.md) — sprint planning
