# Sprint 17: Bootstrap Source Narrowing, Turn Contract Tightening, and Explore Operator UX

## Prerequisites

Sprint 16

## Goals

Finish the next real contraction after Sprint 16: stop treating the public runtime boundary as an `Agent` object by default, keep deleting in-stream repair ownership where the runtime can enforce a stronger contract instead, and give the new explore continuity story a small but real operator surface.

Sprint 16 closed an important shell-level loop. Loader now has a first-class runtime launcher contract, a smaller `agent/loop.py`, explicit compatibility-boundary proof, and persisted explore continuity. That work changed the shape of the remaining debt in a useful way:

- the runtime no longer needs `agent/loop.py` to decide chat vs decompose vs direct turn routing
- compatibility shims are now explicit and guarded instead of silently load-bearing
- explore is no longer purely one-shot
- but the runtime still starts from an `Agent`-shaped bootstrap source at the public boundary
- `agent/loop.py` still owns prompt/session factory behavior and too much entrypoint lifecycle glue
- and the old audit critique still matters in a narrower form: Loader must keep deleting or hardening in-stream repair behavior, not just moving it again

`audit.txt` is stale on specifics, test counts, and many file-level claims. It is not the roadmap anymore. But one core warning is still worth carrying into Sprint 17:

- do not keep wrapping model-misbehavior recovery in nicer files if the runtime can instead enforce a stronger explicit contract

Sprint 17 should use that warning deliberately while staying grounded in the current codebase rather than the old audit snapshot.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/runtime_context.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint16.md`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`

## Deliverables

### 1. Narrow the public bootstrap source below `Agent`

Sprint 16 gave Loader a launcher contract. Sprint 17 should stop treating that launcher contract as “an agent object with the right fields.”

Implementation targets:

- introduce a smaller public bootstrap/launcher source under `src/loader/runtime/`, likely around:
  - backend
  - registry
  - session
  - permission/capability state
  - prompt/session callbacks that still genuinely need to stay dynamic
- reduce direct `RuntimeBootstrapSource` dependence on the full `Agent` object
- keep `ConversationRuntime`, `ExploreRuntime`, and `RuntimeLauncher` constructing from that narrower source rather than from `Agent` by convention
- avoid moving fields mechanically; the point is to decide which launch-time responsibilities truly belong in the runtime boundary and which belong in the public shell

The goal is not “zero references to Agent.” The goal is to make the public runtime boundary look intentionally runtime-shaped rather than coincidentally agent-shaped.

### 2. Move prompt/session shell behavior out of `agent/loop.py`

Sprint 16 shrank entry routing. Sprint 17 should take the next obvious shell debt: prompt/session factories and lifecycle glue.

Implementation targets:

- inventory what still makes `src/loader/agent/loop.py` feel heavier than a public facade, especially:
  - system-prompt construction and snapshot persistence
  - few-shot example selection
  - session creation/replacement boilerplate
  - resume/clear lifecycle helpers that have become reusable runtime-shell behavior in practice
- move or re-home the behavior that is no longer meaningfully “agent-owned”
- keep `agent/loop.py` focused on:
  - public entrypoints
  - explicit lifecycle commands
  - UI/event wrappers
  - compatibility accessors that still truly need to exist

The goal is to make Sprint 16’s thinner shell materially easier to understand on sight.

### 3. Tighten the remaining turn-repair contract

This is where `audit.txt` still matters. The remaining question is no longer “did we extract the repair path?” but “are we still repairing things the runtime should simply reject or halt on?”

Implementation targets:

- inventory the remaining repair/completion heuristics across:
  - `src/loader/runtime/repair.py`
  - `src/loader/runtime/completion_policy.py`
  - `src/loader/runtime/turn_completion.py`
  - `src/loader/runtime/assistant_turns.py`
- identify the heuristics that still look like in-stream puppeting or speculative follow-through rather than explicit runtime contract
- prefer one of:
  - deletion
  - a stricter typed failure state
  - an explicit retry budget with honest surfaced failure
  over adding another wrapper or reroute
- add direct regression tests for every deleted/tightened behavior so we do not silently reintroduce it later

This is the sprint where we should convert more “runtime tries to rescue the model” behavior into “runtime enforces the contract and reports honestly.”

### 4. Give explore a small but real operator surface

Sprint 16 gave explore continuity. Sprint 17 should make that state inspectable and manageable.

Implementation targets:

- add a small operator-facing explore surface, likely around:
  - recent explore status/history inspection
  - resetting explore continuity without touching main workflow sessions
  - clearer visibility into whether an explore query ran fresh vs continued
- keep the explore lane intentionally lightweight:
  - no DoD
  - no workflow artifacts
  - no mutation
  - no conversion into a second main runtime
- prefer one or two clean CLI/inspection surfaces over a broad subcommand family

The goal is to make explore continuity usable and debuggable, not to build a whole second product mode.

### 5. Keep the audit line active as a contract check, not a competing roadmap

Implementation targets:

- use `audit.txt` only for the still-valid patterns it warns about:
  - additive wrapper cleanup
  - hidden ownership
  - in-stream rescue behavior that should become explicit contract
- prefer current code and current sprint audits over old audit counts or old branch-era file claims
- update `PARITY.md` and the sprint audit only after the bootstrap narrowing and turn-contract work are directly covered

## Testing strategy

- unit coverage for:
  - narrowed launcher/bootstrap source construction
  - prompt/session-shell helpers after re-homing
  - deleted or tightened repair/completion heuristics
  - explore operator surfaces and continuity reset behavior
- runtime coverage for:
  - main runtime launch through the narrowed bootstrap source
  - explore continuity through the new operator surface
  - existing launcher/chat/decomposition parity staying green after the shell contraction
- regression coverage for:
  - no new implicit dependence on full `Agent` shape at the runtime boundary
  - no silent return of deleted repair/continuation heuristics
  - no explore continuity leakage into main session/DoD state

## Definition of done

- the public runtime bootstrap source is narrower and less `Agent`-shaped than Sprint 16
- `agent/loop.py` shrinks further toward public facade and lifecycle glue only
- Loader deletes or tightens more remaining in-stream repair behavior instead of merely relocating it
- explore continuity gains a small operator surface while staying read-only and workflow-light
- the parity baseline remains green after the Sprint 17 contract tightening

## Explicitly out of scope

- full claw-code policy-engine parity
- multi-agent or team orchestration
- AST-aware or LSP-aware semantic artifact diffs
- a full visual explore workflow
- a broad rule editor or policy authoring UI
