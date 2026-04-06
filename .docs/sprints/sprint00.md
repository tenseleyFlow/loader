# Sprint 00: Foundation, Measurement, and Parity Harness

## Prerequisites

None. This is the stabilization sprint before major behavior work.

## Goals

Make Loader measurable and trustworthy enough to improve deliberately.

This sprint exists to prevent us from adding more agent behavior on top of:

- a monolithic runtime loop (`agent/loop.py` is 1929 LOC, `agent/reasoning.py` is 1196 LOC, `agent/safeguards.py` is 1079 LOC — together about 4200 lines of orchestration in one cluster)
- a structurally broken tool-result code path that has zero coverage
- broken default test discovery (pytest currently picks up `refs/claw-code/tests/` and fails to import the `loader` package)
- weak operational polish

## Deliverables

### 1. Failing regression test for the `tool_call_id` runtime contract bug — DO THIS FIRST

Before any harness or hygiene work, write a failing pytest case that drives the duplicate-suppression and pre-validation branches in `src/loader/agent/loop.py`.

The bug:

- `src/loader/llm/base.py:33-39` defines `Message` with `role`, `content`, `tool_calls`, `tool_results` — and **no** `tool_call_id` field. That field belongs to the separate `ToolResult` dataclass at `src/loader/llm/base.py:25-30`.
- `src/loader/agent/loop.py:885` and `:906` both construct `Message(role=Role.TOOL, content=..., tool_call_id=tool_call.id)`.
- Both call sites raise `TypeError: Message.__init__() got an unexpected keyword argument 'tool_call_id'` the first time they execute.
- They live on the duplicate-suppression branch and the pre-validation-failure branch, neither of which has any integration coverage today.

Why this is the first deliverable:

- it proves the bug is real
- it proves the harness exists
- it gives Sprint 01 a green-bar target rather than a vague refactor goal
- it answers the question "are there other bugs like this?" by forcing us to actually drive the loop

The test should fail today and remain failing until Sprint 01 fixes the message contract.

### 2. Project hygiene and product basics

- rewrite `README.md` (currently still says "FortranGoingOnForty / A tutorial on using Fortran for beginners")
- ensure `refs/` remains gitignored
- document the current runtime surface and limitations in `.docs/`

### 3. Test execution that works by default

The current state of `uv run pytest --collect-only`:

- picks up `refs/claw-code/tests/test_porting_workspace.py` (because there is no `tool.pytest.ini_options` block in `pyproject.toml`)
- fails to import the `loader` package in the resolved env
- collects 0 tests

Fix:

- add `[tool.pytest.ini_options]` to `pyproject.toml` with `testpaths = ["tests"]`
- ensure the package is importable under the test invocation path
- add `extend-exclude = ["refs"]` to the ruff configuration so refs/ does not contaminate lint runs
- add `exclude = ["refs/"]` to the mypy configuration for the same reason
- document the canonical dev/test invocation in the README and in `CLAUDE.md`
- `uv run pytest` (with no flags) must succeed out of the box

### 4. Runtime behavior harness

Create a deterministic mock backend and scenario harness for the current turn loop.

**Pattern to copy:** `refs/claw-code/rust/crates/rusty-claude-cli/tests/mock_parity_harness.rs`. claw-code already implements this exact design — a mock service plus a scripted scenario taxonomy. Port the taxonomy directly rather than inventing a parallel one.

Minimum scenarios (mirroring claw-code's list, adapted for Loader):

- simple answer with no tools (`streaming_text`)
- single read tool call (`read_file_roundtrip`)
- multi-tool turn (`multi_tool_turn_roundtrip`)
- write allowed (`write_file_allowed`)
- write denied (`write_file_denied`) — initially via skip-confirmation default; rewired to permission policy in Sprint 03
- bash success (`bash_stdout_roundtrip`)
- bash confirmation prompt approved/denied
- extracted/raw-text tool call fallback
- completion-check continuation
- duplicate action suppression (this scenario will hit the bug from deliverable 1)

### 5. Baseline behavior document

Create a parity checklist for Loader's own runtime behavior:

- what is supported
- what is flaky
- what is intentionally out of scope
- what scenarios must stay green

## Testing strategy

- the `tool_call_id` regression test fails today (pre-fix) and is committed in its failing state, gated only by the harness
- default `uv run pytest` succeeds and collects only Loader's own tests
- harness scenarios produce stable results
- at least one integration test exercises the full turn loop end-to-end
- baseline parity checklist is committed and auditable

## Definition of done

- the `tool_call_id` regression test exists and is failing for the right reason
- `README.md` correctly describes Loader
- `pytest`/`uv` workflow is defined and working
- ruff and mypy do not walk into `refs/`
- Loader has a deterministic runtime test harness with the scenario taxonomy ported from claw-code
- current runtime behavior is documented honestly
- we can measure regressions before changing the loop
