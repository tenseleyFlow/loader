# Sprint 16: Entrypoint Shell, Launcher Contract, and Explore Continuity

## Prerequisites

Sprint 15

## Goals

Turn Sprint 15's ownership cleanup into a cleaner public runtime shape: reduce the remaining `Agent`-shaped bootstrap dependency, shrink `agent/loop.py` further toward a thin facade, and make explore mode feel like a maintained read-only runtime lane instead of a one-shot side path.

Sprint 15 closed an important loop. Loader now has a shared runtime bootstrap seam, runtime-owned safeguards and deliberation helpers, compatibility-only `agent/reasoning.py` and `agent/safeguards.py`, no hidden `Agent._build_runtime_context()` helper, and a much smaller `agent/loop.py`.

That leaves a tighter but more visible set of residual debts:

- `conversation.py` and `explore.py` still start from an `Agent`-shaped bootstrap source at the public entrypoint layer
- `agent/loop.py` still owns conversational fast-path behavior, decomposition orchestration, and too much session/UI-facing wiring for a runtime that otherwise lives under `src/loader/runtime/`
- `agent/reasoning.py` and `agent/safeguards.py` are now compatibility-only, but Loader has not yet decided how narrow that compatibility surface should become
- explore mode is structurally cleaner than it was, but it is still a one-shot lookup lane rather than a more intentional read-only workflow

Sprint 16 is about finishing that next contraction without drifting into a brand-new roadmap:

- the runtime gets a first-class launcher/entrypoint contract instead of depending on an `Agent`-shaped bootstrap source everywhere
- `agent/loop.py` shrinks further toward a public facade over runtime-owned launch and orchestration helpers
- compatibility exports remain explicit, tested, and intentionally narrow instead of becoming permanent soft ownership
- explore mode gains a better continuity/inspection contract so it feels like a real product lane, not just a single-turn utility

This sprint should feel like consolidating the public shell after Sprint 15, not reopening the old runtime ownership debates.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/runtime_context.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint15.md`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`

## Deliverables

### 1. Introduce a first-class runtime launcher contract

Sprint 15 made bootstrap shared. Sprint 16 should make the public launch path less `Agent`-shaped.

Implementation targets:

- introduce an explicit launcher/entrypoint seam under `src/loader/runtime/`, likely around:
  - creating a main-turn runtime
  - creating an explore runtime
  - carrying the minimum public bootstrap state needed by those runtimes
- reduce direct dependence on `ConversationRuntime(self)` / `ExploreRuntime(self)` from `agent/loop.py`
- make the launcher contract narrow and explicit:
  - project/session/prompt/capability state that truly belongs to launch-time setup
  - public callbacks that still genuinely need to stay at the `Agent` layer
- avoid replacing one implicit wrapper with another vague wrapper; the goal is a visibly smaller ownership surface

The goal is not to erase `Agent`. The goal is to make `Agent` a thin public shell over a runtime launcher contract instead of the default owner of launch-time state.

### 2. Shrink `agent/loop.py` into a thinner facade

Sprint 15 deleted dead code. Sprint 16 should move more of the still-live orchestration to clearer homes.

Implementation targets:

- move or re-home more of the still-live `agent/loop.py` behavior that has become runtime or launcher ownership in practice, especially:
  - conversational fast-path handling
  - decomposition orchestration
  - runtime/explore construction and launch wiring
  - session-facing setup that does not truly need to live on the agent object
- keep `agent/loop.py` focused on:
  - public entrypoints
  - top-level session lifecycle
  - UI/event integration
  - explicit compatibility shims that still need to exist
- continue deleting dead helper paths instead of letting the shell regrow around the new launcher seam

This is the sprint where the public shell should become visibly easier to understand.

### 3. Narrow the compatibility-export surface deliberately

`agent/reasoning.py` and `agent/safeguards.py` are now compatibility layers. Sprint 16 should make that state intentional rather than indefinite.

Implementation targets:

- inventory the remaining compatibility-only exports under:
  - `src/loader/agent/reasoning.py`
  - `src/loader/agent/safeguards.py`
- decide which exports still need to exist for test/import compatibility and which can be retired
- keep internal Loader code importing canonical runtime modules instead of their compatibility mirrors
- add direct tests that lock the intended compatibility contract:
  - which symbols are still re-exported
  - which internal hot paths must not import via compatibility modules anymore

The goal is not immediate deletion of every compatibility export. The goal is to stop compatibility from behaving like shadow ownership.

### 4. Give explore mode better continuity and inspection shape

Explore is cleaner now, but still underpowered as a product lane.

Implementation targets:

- deepen the explore runtime without turning it into full workflow mode, likely around:
  - lightweight transcript continuity for follow-up explore questions
  - clearer inspection or status surfaces for recent explore activity
  - a stronger read-only session/state story
- preserve the current constraints:
  - read-only registry
  - no DoD
  - no mutating workflow artifacts
  - explicit denial of write/destructive actions
- add direct coverage so explore remains a maintained runtime lane rather than a product afterthought

The goal is to make explore feel intentionally usable over multiple adjacent questions, not to turn it into a second main runtime.

### 5. Keep the audit line active as a deletion-first check

The useful part of `audit.txt` is still the bias toward deleting disguised ownership instead of endlessly wrapping it.

Implementation targets:

- use the audit's core complaint as a Sprint 16 check:
  - do not leave the launcher contract `Agent`-shaped by habit
  - do not regrow logic in `agent/loop.py` after moving it out
  - do not let compatibility exports become the internal default import path again
- update `PARITY.md` and the sprint audit only after the launcher/entrypoint work is directly covered

## Testing strategy

- unit coverage for:
  - runtime launcher creation and bootstrap narrowing
  - conversational/decomposition orchestration after re-homing
  - compatibility-export boundaries
  - explore continuity state and inspection helpers
- runtime coverage for:
  - main runtime launch through the new launcher contract
  - explore runtime launch and follow-up continuity
  - Sprint 00-15 parity scenarios staying green after the entrypoint-shell contraction
- regression coverage for:
  - no return of hidden bootstrap helpers
  - no internal fallback to compatibility imports for runtime-owned helpers
  - no mutation leakage into explore continuity/session behavior

## Definition of done

- Loader has a first-class runtime launcher/entrypoint seam instead of relying on an `Agent`-shaped bootstrap source everywhere
- `agent/loop.py` shrinks further toward a public facade over runtime-owned launch/orchestration helpers
- compatibility exports under `agent/reasoning.py` and `agent/safeguards.py` are narrower, explicitly tested, and no longer used by internal hot paths
- explore mode has a stronger continuity/inspection contract while staying read-only and workflow-light
- the parity baseline remains green after the launcher/entrypoint changes

## Explicitly out of scope

- full claw-code policy-engine parity
- multi-agent or team orchestration
- AST-aware or LSP-aware semantic artifact diffs
- a full visual explore/workflow UI
