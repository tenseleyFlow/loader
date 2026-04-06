# Loader

Loader is a local-first coding assistant that runs against Ollama and drives a small tool-using agent loop from the terminal or a Textual TUI.

## Current state

Loader is still early-stage. The project already has:

- a working terminal and TUI interface
- a default tool set for file reads/writes, editing, search, globbing, and shell commands
- project-context detection
- runtime safeguards, recovery logic, and heuristic completion checks

It does not yet have the stronger runtime architecture described in [`.docs/REPORT.md`](.docs/REPORT.md). The current implementation plan lives under [`.docs/sprints/`](.docs/sprints/index.md), and the Sprint 00 runtime baseline is tracked in [`.docs/PARITY.md`](.docs/PARITY.md).

## Requirements

- Python 3.11+
- `uv`
- a running local Ollama server on `http://localhost:11434`
- at least one pulled Ollama model

## Setup

```bash
uv sync
uv sync --extra dev
```

## Common commands

```bash
uv run loader
uv run loader "write a hello world program"
uv run loader --no-tui
uv run loader --select-model

uv run pytest
uv run pytest tests/test_runtime_harness.py -q
uv run ruff check src tests
uv run mypy src
```

## Repository notes

- Loader source lives under `src/loader/`
- tests live under `tests/`
- reference repos used for comparison live under `refs/` and are gitignored
- `uv run pytest` is scoped to Loader's own `tests/` directory

## Known limitations

- the agent loop is still monolithic and heuristic-heavy
- there is a known tool-result contract regression captured by the runtime harness and left red for Sprint 01
- permissions are still confirmation-based rather than mode-based
- session persistence, planning artifacts, and verification loops are not implemented yet
