# Sprint 09: Turn State Machine, Workflow Contracts, and Prompt Preview

## Prerequisites

Sprint 08

## Goals

Turn Loader's explicit phase labels into a real runtime contract and make workflow routing explainable instead of merely observable.

Sprint 08 gave Loader a typed prompt builder, named turn phases, and permission/operator surfaces. That was the right bridge, but the audit is honest about what still hurts:

- `conversation.py` still coordinates too much through heuristic branching
- workflow routing is still implied by helper decisions more than enforced by a transition model
- phase state is visible, but not yet validated like a real state machine
- prompt metadata is inspectable, but the actual prompt contract is still hard to preview directly

The next leverage point is to stop treating state as labels attached to heuristics and start treating it as the runtime.

This sprint is about discipline:

- turn progression becomes a validated state machine
- workflow routing becomes a typed decision contract with persisted reasons
- `conversation.py` becomes thinner again by delegating transitions instead of deciding everything inline
- operators can inspect the prompt/workflow contract without triggering a live model turn

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/mcp_lifecycle_hardened.rs`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/modes/base.ts`

## Deliverables

### 1. Validated turn state machine instead of phase bookkeeping only

Sprint 08 added named phases. Sprint 09 should make those phases authoritative.

Implementation targets:

- introduce a dedicated turn-state machine under `src/loader/runtime/` with typed states, transition intents, and terminal reasons
- define explicit allowed transitions for the main turn lifecycle, for example:
  - `prepare -> assistant`
  - `assistant -> repair`
  - `assistant -> tools`
  - `assistant -> completion`
  - `tools -> critique`
  - `critique -> completion`
  - `completion -> finalize`
- fail loudly on impossible transitions in tests and tracing instead of silently drifting
- capture transition metadata such as:
  - why a transition happened
  - whether it was a normal path, reroute, retry, or recovery move
  - whether it ended in success, blocked, fix-loop reentry, or iteration exhaustion
- keep the state machine separate from model/tool side effects so transition logic is testable on its own

The goal is not ceremony. The goal is to make Loader's turn lifecycle harder to accidentally regress.

### 2. Typed workflow-routing contract with persisted reasoning

Loader already has `clarify`, `plan`, `execute`, `verify`, and `explore`, but the router still behaves more like a cluster of heuristics than a durable contract.

Implementation targets:

- replace the current loose routing outputs with a typed workflow decision object, for example:
  - selected mode
  - reason code
  - ambiguity/complexity signal
  - whether the mode is an initial route, a reentry, or a forced continuation
  - whether a downstream mode is already scheduled
- persist workflow-decision metadata in session/runtime state so inspection surfaces can explain:
  - why Loader chose clarify vs plan vs execute
  - why a verify failure returned to execute
  - why a conversational task skipped verification
- move verify/fix loop reentry onto the same contract instead of letting it remain a partial special case
- keep explore explicitly outside the main mutating workflow contract while still representing it as a typed lane

This is how Loader gets closer to OMX's “mode with a contract” behavior without copying OMX's full planning stack yet.

### 3. Slim `conversation.py` around transitions and routing

After Sprint 08, `conversation.py` is better, but it still knows too much about routing and control flow.

Implementation targets:

- extract workflow-decision evaluation and turn-transition advancement into dedicated runtime modules
- make `ConversationRuntime.run_turn(...)` primarily:
  - build turn context
  - ask the router for the next workflow decision
  - advance the state machine
  - delegate assistant/tool/repair/finalization work
  - persist the resulting turn summary
- remove remaining inline branches that duplicate completion/reentry/skip logic in multiple places
- keep all new control-flow work inside `src/loader/runtime/`, not back in `agent/loop.py`

A good outcome is that `conversation.py` reads like an orchestrator over contracts rather than a place where routing policy keeps accumulating.

### 4. First-class prompt/workflow preview surfaces

Sprint 08 made prompt and policy metadata legible. Sprint 09 should make the actual contract previewable.

Implementation targets:

- add `loader prompt show` to render the current prompt contract for a task without issuing a model request
- support previewing at least:
  - workflow mode
  - prompt format (`native` / `react`)
  - included dynamic sections
  - current permission mode
  - relevant workflow context
- keep the output operator-friendly:
  - summary metadata first
  - prompt body second
  - no hidden live side effects
- extend `loader status` / `loader session show` with the latest workflow-decision reason and latest transition summary when present

The goal is to make Loader's runtime choices debuggable before and after a turn, not only during one.

## Testing strategy

- unit coverage for:
  - allowed and disallowed turn-state transitions
  - terminal-state reasons and retry/reentry transitions
  - workflow-decision objects and persisted reason codes
  - prompt-preview rendering over multiple workflow modes
- CLI coverage for:
  - `loader prompt show`
  - workflow reason/transition fields in `loader status` and `loader session show`
- deterministic/runtime coverage for:
  - verify/fix reentry using only valid state-machine transitions
  - conversational tasks skipping verification through a typed workflow decision instead of an implicit branch
  - repair-phase retries preserving valid transition sequences
  - Sprint 00-08 parity scenarios staying green after the state-machine split
- regression coverage:
  - no turn path should bypass the state machine once the contract is introduced
  - `conversation.py` should no longer be the only place that knows whether a reroute is legal

## Definition of done

- Loader has a validated turn-state machine, not just named phase labels
- workflow routing emits typed, persisted decisions with explainable reasons and reentry metadata
- `conversation.py` is slimmer again and more coordinator-like
- operators can preview the current prompt contract without a live model call
- status/session surfaces can explain the latest workflow decision and transition outcome
- the full parity baseline remains green after the control-flow split

## Explicitly out of scope

- a full workflow editor or visual state-machine UI
- a first-class permission rule editor
- OMX-style multi-iteration consensus planning
- AST-aware, LSP-aware, or symbol-aware editing
- multi-agent or team orchestration
