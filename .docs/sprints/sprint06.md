# Sprint 06: Doctor, Explore, Status, and Tool Surface Expansion

## Prerequisites

Sprint 05

## Goals

Turn Loader from "a loop with a TUI" into a tool with inspectable operational surfaces and a tool surface broad enough to keep the main runtime loop from being the only place every operation happens.

The references for this sprint are:

- `refs/oh-my-codex/src/cli/doctor.ts` (health checks)
- `refs/oh-my-codex/src/cli/explore.ts` (read-only inspection lane)
- `refs/claw-code/rust/crates/tools/src/lib.rs` (the 49-tool surface — Loader does not need all of them, but the categories are the right reference)

## Deliverables

### 1. `loader doctor`

Add a health-check surface that reports:

- backend connectivity (Ollama up, model pulled)
- model capability summary (resolved from the Sprint 01 capability profile)
- workspace detection (Sprint 01 project context)
- write access (workspace boundary, `.loader/` writable)
- test/build command detection (from project context)
- state/session directory health (`.loader/` exists, sessions/dod/briefs/plans dirs present, project-memory parseable)
- permission mode (current default and what each tool would resolve to)

Each check returns `pass | warn | fail` with a one-line message and a remediation hint.

`loader doctor` should be runnable without entering the main runtime loop — it is a diagnostic, not a turn.

### 2. `loader status` and `loader session`

Expose:

- `loader status` — current model, capability profile, permission mode, workflow mode, active session, recent verification/evidence state, DoD phase
- `loader session list` — sessions in `.loader/sessions/` with id, started-at, last-updated, message count, DoD status
- `loader session show <id>` — full detail for one session
- `loader session resume <id>` — wired to Sprint 05's resume support

Both commands read from `.loader/` and never invoke the LLM.

### 3. Lightweight read-only explore lane

Add an optimized read-only inspection path for:

- file lookup
- symbol lookup
- pattern discovery
- repo relationship questions ("where is X imported?", "what calls Y?")

The explore lane:

- runs in `read-only` permission mode by default (regardless of the user's session-wide setting)
- skips the verify phase from Sprint 02 (lookups have no DoD)
- skips the mode router's `clarify` and `plan` modes from Sprint 04 — it goes straight to a constrained `execute` with read-only tools only
- has its own concise system prompt focused on lookup, not action

Reference: `refs/oh-my-codex/src/cli/explore.ts:48-77` (allows only read-only git subcommands, validates against pipes/redirects/semicolons in tokenized form).

This is what keeps simple lookups out of the full execution loop and addresses the user's "spending too long on simple tasks" complaint.

### 4. Tool surface expansion

Add a first serious expansion pass for Loader tools. `TodoWrite` and `AskUserQuestion` already exist from Sprint 04. New tools in this sprint:

- **diff/patch-aware editing** — a tool that takes a structured patch (using the `StructuredPatchHunk` shape from Sprint 03's file_ops hardening) instead of raw old/new strings. Reduces the failure rate on multi-line edits.
- **git status helper** — read-only git status / log / diff / show / branch surface, similar to OMX explore's tokenized subcommand allowlist. Lives in the explore lane but also available to the main runtime.
- **memory/notepad tools** — wire the Sprint 05 memory surfaces into the tool registry so the model can call them: `project_memory_read`, `project_memory_write`, `notepad_read`, `notepad_append`.
- **structured ask-user** — a richer variant of `AskUserQuestion` that can present multiple-choice options or numbered alternatives (for plan-mode handoff and verify-mode "which fix do you want?")

Do **not** add team/subagent orchestration in this sprint. The solo runtime needs to be stable first. Multi-agent surfaces are deferred indefinitely.

### 5. CLI/TUI consolidation

By this point Loader has accumulated a lot of CLI flags and TUI surfaces. Sprint 06 should:

- consolidate the flag surface (deprecate flags that are now redundant with mode router decisions)
- update the TUI status line to surface model + capability profile + permission mode + workflow mode + DoD phase + session id, all in one line
- ensure `loader --help` is coherent

## Testing strategy

- doctor reports meaningful failures for: Ollama down, model not pulled, workspace not writable, `.loader/` corrupted
- doctor reports pass for a known-good local setup
- status/session surfaces reflect real runtime state (assert against fixtures in `.loader/`)
- explore mode handles read-only lookups without entering the full execution workflow (assert: no DoD object created, no verify phase, no mode router decision logged)
- explore mode denies write attempts even if the user has `workspace-write` set globally
- new tools are covered by both unit tests and the Sprint 00 mock harness
- the Sprint 02 baseline parity checklist (which has been growing each sprint) covers the new product surfaces

## Definition of done

- Loader is operable and inspectable from outside the main runtime loop
- simple inspection tasks are faster and cheaper via the explore lane
- the expanded tool surface reduces prompt pressure on the main loop
- doctor / status / session surfaces reflect real state
- Loader feels closer to a product and less like an experiment
- the team / multi-agent / hook-ecosystem deferrals are still deferred (and that is the right call)
