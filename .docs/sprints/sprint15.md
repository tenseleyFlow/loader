# Sprint 15: Bootstrap Ownership, Service Burn-Down, and Explore Independence

## Prerequisites

Sprint 14

## Goals

Finish the last high-value runtime cleanup that Sprint 14 exposed: move bootstrap ownership and the remaining agent-owned runtime services onto explicit runtime seams, so Loader's hot path is not only context-driven after initialization, but also context-driven at initialization.

Sprint 14 was a real architectural win. `RuntimeContext` is now the primary seam across workflow state, turn phases, response repair, response routing, turn looping, workflow recovery, and finalization. The older `RuntimeLegacyServices` shim is gone, raw-text tool recovery no longer depends on hidden agent extractors, and the main runtime path is much less accidental than it was when the audit line first branched.

That said, the current residual debt is now very specific:

- `conversation.py` and `explore.py` still bootstrap from `agent._build_runtime_context()`
- `agent/loop.py` still owns too much prompt/session/bootstrap coordination for a runtime that is otherwise context-owned
- `agent/reasoning.py` and `agent/safeguards.py` still own meaningful runtime behavior behind typed protocols
- the audit's core warning still matters in a narrower form:
  Loader should keep deleting wrapper-only ownership, not just wrapping it in nicer files

Sprint 15 is about finishing that next contraction honestly:

- runtime bootstrapping becomes an explicit runtime contract instead of an agent-only helper
- explore mode stops being a special runtime that still depends on agent construction shape
- reasoning and safeguard ownership become more inventoryable and less agent-bound
- `agent/loop.py` shrinks further toward entrypoint/session orchestration instead of runtime-service ownership

This sprint should feel like closing the structural loop opened by Sprint 14, not starting a new product branch.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/rust/crates/runtime/src/runtime_context.rs`
- `refs/claw-code/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/audit_sprints/sprint13_closure.md`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`

## Deliverables

### 1. Runtime bootstrap becomes a first-class runtime seam

Sprint 14 made `RuntimeContext` load-bearing after construction. Sprint 15 should make construction itself less agent-special.

Implementation targets:

- introduce an explicit runtime bootstrap/factory seam under `src/loader/runtime/`, likely around:
  - building `RuntimeContext`
  - initializing project/session/prompt/capability state needed by runtimes
  - synchronizing prompt/capability metadata when the backend or prompt contract changes
- reduce direct runtime dependence on `agent._build_runtime_context()` so `ConversationRuntime` and `ExploreRuntime` do not depend on a hidden agent helper as their primary construction mechanism
- keep the contract pragmatic:
  - it is acceptable for `Agent` to call the factory
  - it is not acceptable for runtime correctness to depend on ad hoc agent-only bootstrap behavior

The goal is to make runtime ownership explicit from the first line of construction, not only once the turn is already running.

### 2. Burn down more agent-owned runtime services

The remaining runtime service ownership now lives mostly in `agent/reasoning.py` and `agent/safeguards.py`.

Implementation targets:

- inventory the still-runtime-relevant behavior in:
  - `src/loader/agent/reasoning.py`
  - `src/loader/agent/safeguards.py`
- move or re-home the behavior that is still genuinely runtime-owned, especially around:
  - confidence / verification service boundaries
  - stream filtering / steering / duplicate detection
  - action validation hooks
- prefer one real implementation plus compatibility exports over keeping a runtime wrapper around an agent-owned implementation indefinitely
- explicitly delete or retire dead wrapper layers where the runtime already has a better home

This is the sprint where we should be suspicious of “adapter forever” solutions. If a behavior is still part of the runtime contract, it should increasingly live under `runtime/`.

### 3. Explore runtime should share the same bootstrap discipline

Explore is intentionally narrower than the main runtime, but it should not be structurally special in the wrong way.

Implementation targets:

- remove or narrow the `ExploreRuntime(agent)` construction shape so explore can be built from the same runtime bootstrap contract as the main runtime
- keep the read-only registry, read-only permission mode, and capability refresh behavior intact
- add direct tests for explore bootstrap and state ownership so explore remains a maintained runtime lane rather than a side path

The goal is not to make explore bigger. The goal is to make it less magical and more aligned with the primary runtime contract.

### 4. Shrink `agent/loop.py` toward entrypoint orchestration

Sprint 14 made the runtime path smaller and more explicit. Sprint 15 should let `agent/loop.py` benefit from that work.

Implementation targets:

- move more bootstrap/session/prompt/runtime wiring out of `agent/loop.py` where it has become runtime ownership in practice
- keep `agent/loop.py` focused on:
  - public entrypoints
  - session-facing orchestration
  - UI/event integration
  - compatibility wrappers that still truly need to exist
- avoid letting new runtime helpers bounce back into `agent/loop.py` just to preserve old ownership lines

This is the step that turns Sprint 14's seam cleanup into a visibly smaller agent shell.

### 5. Keep the audit line active as a regression check, not a second roadmap

`audit.txt` is old on specifics but still sharp on the pattern to avoid: additive cleanup that never deletes ownership.

Implementation targets:

- use the audit's core complaint as a check against Sprint 15 implementation:
  - do not add a new wrapper if we can adopt or delete
  - do not leave bootstrap ownership ambiguous
  - do not grow a “temporary” compatibility seam without direct tests and an exit story
- update `PARITY.md` and the sprint audit only after the bootstrap/service changes are actually covered

## Testing strategy

- unit coverage for:
  - runtime bootstrap/factory behavior
  - explore bootstrap behavior
  - runtime-owned reasoning/safeguard services after migration
  - prompt/capability synchronization at the new bootstrap seam
- runtime coverage for:
  - main turn execution through the new bootstrap path
  - explore mode through the shared bootstrap/runtime contract
  - Sprint 00-14 parity scenarios staying green after the bootstrap/service migration
- regression coverage for:
  - no reintroduction of hidden raw-text extractors
  - no reintroduction of legacy callback shims equivalent to `RuntimeLegacyServices`
  - no ownership drift where runtime modules silently depend on agent-only helpers again

## Definition of done

- runtime bootstrapping is a first-class runtime seam, not primarily an agent helper
- explore mode shares the same bootstrap discipline as the main runtime
- more runtime-relevant behavior is moved or retired out of `agent/reasoning.py` and `agent/safeguards.py`
- `agent/loop.py` shrinks further toward entrypoint/session orchestration
- the parity baseline remains green after the bootstrap/service migration

## Explicitly out of scope

- full claw-code policy-engine parity
- AST-aware or LSP-aware semantic artifact diffs
- a richer permission rule editor
- visual workflow tooling
- multi-agent or team orchestration
