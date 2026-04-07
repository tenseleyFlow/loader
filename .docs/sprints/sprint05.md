# Sprint 05: Session State, Memory, and Compaction

## Prerequisites

Sprint 04

## Goals

Give Loader durable continuity.

This sprint should reduce re-discovery, improve multi-turn coherence, and support longer-running tasks without drowning the model in raw transcript.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/session.rs` (session persistence with rotation and compaction metadata)
- `refs/claw-code/rust/crates/runtime/src/compact.rs` (auto-compaction trigger)
- `refs/claw-code/rust/crates/runtime/src/summary_compression.rs` (priority-aware line-level summarization — model on this rather than reinventing)
- `refs/oh-my-codex/src/mcp/memory-server.ts` (project memory + notepad surfaces)

## Deliverables

### 1. Full session store under `.loader/`

The minimum `.loader/` shape was created in Sprint 02 (`.loader/dod/`). This sprint extends it to the full layout:

```text
.loader/
├── dod/                  # already exists from Sprint 02
├── briefs/               # already exists from Sprint 04
├── plans/                # already exists from Sprint 04
├── sessions/             # NEW — persisted conversation state
├── state/                # NEW — runtime state (current session pointer, etc.)
├── notepad.md            # NEW — durable working notes
└── project-memory.json   # NEW — repo conventions and user directives
```

Sessions are persisted with:

- session id
- created/updated timestamps
- messages (using the Sprint 01 typed message schema)
- compaction metadata (when applicable)
- DoD object reference (link to the active task's DoD file in `.loader/dod/`)
- usage tracking

File rotation: cap individual session files at ~256 KB and rotate, matching `refs/claw-code/rust/crates/runtime/src/session.rs:13-14`.

### 2. Resume support

Allow Loader to resume the latest or a named session:

- `loader --resume` resumes the most recent session
- `loader --resume <session-id>` resumes by id
- `loader session list` (which lands in Sprint 06) shows available sessions

Resume must restore the message history, the active DoD object, the active mode, and the active permission policy.

### 3. Working memory and project memory

Add tools/surfaces to:

- read/write project memory (`.loader/project-memory.json` — tech stack, build commands, conventions, directives, structure)
- append working notes (`.loader/notepad.md` — temporary context that survives across turns within a session)
- store user directives ("we use uv, never pip" / "tests live in tests/" / etc.)

These are MCP-style tools (sections similar to OMX's `project_memory_read` / `project_memory_add_note` / `project_memory_add_directive` / `notepad_read` / `notepad_write_*`), but they live as native Loader tools, not MCP servers. Loader does not need a full MCP runtime yet.

Each memory tool declares `read-only` permission for the read variants and `workspace-write` for the write variants (since `.loader/` is inside the workspace).

### 4. Transcript compaction

Implement session compaction modeled on `refs/claw-code/rust/crates/runtime/src/summary_compression.rs`:

- triggered automatically when input tokens exceed a threshold (default 100,000, matching claw-code's `DEFAULT_AUTO_COMPACTION_INPUT_TOKENS_THRESHOLD`)
- preserves the most recent N messages (default 4, matching claw-code)
- summarizes older context with priority-aware line-level compression:
  - dedupes identical lines
  - collapses inline whitespace
  - prioritizes "Summary:", "Current work:", "Key files referenced:", and similar core lines
  - bounded to a token/line/char budget
- emits explicit continuation instructions in the summary

The compacted session is written back to `.loader/sessions/` with metadata recording what was removed.

### 5. Usage tracking

Track per-turn and cumulative:

- input tokens
- output tokens
- cache creation tokens (when available from the backend)
- cache read tokens (when available)
- tool calls
- iterations

Usage is attached to the `TurnSummary` from Sprint 01 and accumulated across the session. Cost estimation is a stretch goal — Loader is local-first and the dollar cost is zero, but token tracking is still useful for compaction triggers and observability.

### 6. Memory hooks for the lifecycle

Use the Sprint 03 hook lifecycle for:

- a `post_tool_use` hook that updates `notepad.md` when the user explicitly invokes a "remember this" tool
- a session-finalization hook that writes the DoD evidence summary into `project-memory.json` when relevant (e.g., "the canonical test command is `uv run pytest`")

This is how the durability layer integrates with the rest of the runtime instead of bolting on.

## Testing strategy

- sessions persist and reload correctly across simulated process restarts
- resume restores message history, DoD, mode, and permission policy
- memory/notepad operations survive across turns within a session
- compacted sessions preserve recent messages exactly and produce a summary that round-trips through the priority-aware compression
- compaction triggers automatically at the token threshold
- compacted summary contains the explicit continuation instruction
- file rotation kicks in at the size cap

## Definition of done

- Loader can continue work across sessions without fully re-priming the model
- memory/state lives outside the prompt under `.loader/`
- long sessions can be compacted safely without losing the active DoD or recent messages
- multi-turn work becomes more predictable
- usage tracking is wired into the turn summary
- the file layout is stable enough that Sprint 06's product surfaces can rely on it

## Audit

### Landed

- Loader now persists full session snapshots under `.loader/sessions/` and tracks the active session pointer under `.loader/state/current_session.json`
- persisted sessions now carry typed messages, cumulative usage totals, compaction metadata, the active DoD path, the current task, workflow mode, and permission mode
- `Agent.resume_session(...)` now restores message history, the active DoD object, workflow mode, permission mode, and current task across process restarts
- the CLI now supports both `loader --resume` and `loader --resume <session-id>` by rewriting that syntax into an internal hidden option before Click parsing
- transcript compaction now triggers automatically at the configured input-token threshold, keeps the latest four messages verbatim, and inserts a claw-inspired continuation summary with priority-aware line compression
- `TurnSummary` now carries normalized per-turn usage and cumulative session usage, with streamed Ollama responses reporting prompt/output token counts when available
- Loader now exposes native `project_memory_*` and `notepad_*` tools backed by `.loader/project-memory.json` and `.loader/notepad.md`
- the hook lifecycle now mirrors successful memory writes into the notepad, and finalized DoD evidence summaries are captured into project memory when verification produced useful evidence

### Verification

- `uv run pytest -q` is green: `137 passed`
- `tests/test_session_state.py` covers persistence, resume, rotation, compaction persistence, and cumulative usage rollups
- `tests/test_compaction.py` covers priority-aware summary compression and continuation-message compaction behavior
- `tests/test_memory_tools.py` covers project-memory writes, notepad writes, lifecycle-hook mirroring, and DoD-summary capture into project memory
- `tests/test_cli_resume.py` covers `--resume` argument rewriting for latest and named-session restore
- `tests/test_runtime_harness.py` and `tests/test_workflow_runtime.py` remain green after the session/memory changes, so Sprint 05 did not regress the earlier parity baseline

### Residual debt

- session compaction summaries are runtime-authored heuristics; Loader still does not have claw-code's richer continuation semantics or OMX-style semantic memory extraction
- the DoD-to-project-memory capture is intentionally conservative and may miss higher-value repo conventions unless the evidence summary makes them explicit
- Sprint 05 restores sessions in the CLI runtime, but Sprint 06 still needs to surface session ids, listing, and inspection as first-class product commands
- cache token tracking is normalized when the backend provides it, but Loader still does not estimate cost and some backends may report fewer usage fields than Ollama
- `conversation.py` keeps growing as Sprint 05 logic lands, so Sprint 06+ should keep carving persistence/finalization concerns into smaller runtime components instead of leaving durability inside the turn monolith
