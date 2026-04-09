# Sprint 22: Runtime Entry API, Verification Observations, and Compatibility Narrowing

## Prerequisites

Sprint 21

## Goals

Take the next honest step after Sprint 21: stop treating the new runtime-first owner as only a testing seam, move verification/accountability closer to the moment verification actually happens, and keep narrowing the public compatibility shell without pretending Loader is ready to delete `Agent`.

Sprint 21 changed the remaining debt in a useful way:

- Loader now has a runtime-owned internal handle and no longer needs `Agent` for some runtime-oriented tests
- policy/accountability surfaces can now show grouped supporting vs missing evidence instead of flattening everything into one summary string
- the workflow timeline read model is more canonical and less duplicative
- but `Agent` still remains the default construction seam for most real integrations
- verification evidence is still largely reconstructed from DoD/session state after the fact rather than captured as a first-class observation at execution time
- evidence provenance is richer, but it is still bounded and runtime-authored; Loader still cannot always answer which observed verification attempt or artifact actually justified the final stop/continue decision

Sprint 22 should keep using the references as architectural guardrails, not as a feature-copy list.

The standard remains:

- use claw-code to sharpen runtime-first entry seams, session/bootstrap ownership, and event accountability
- use OMX to sharpen verifier-backed evidence capture and auditability around successful vs missing proof
- do not add work just because the refs have it
- do add work when the refs show that Loader is still too shell-bound, too post-hoc, or too hard to audit honestly

`audit.txt` remains a guardrail against wrapper-heavy drift and fake rescue behavior. It is not the factual roadmap.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/bootstrap.rs`
- `refs/claw-code/rust/crates/runtime/src/session_control.rs`
- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/lane_events.rs`
- `refs/claw-code/rust/crates/runtime/src/green_contract.rs`
- `refs/claw-code/PARITY.md`
- `refs/oh-my-codex/src/verification/verifier.ts`
- `refs/oh-my-codex/src/autoresearch/contracts.ts`
- `refs/oh-my-codex/src/autoresearch/runtime.ts`
- `refs/oh-my-codex/src/hooks/session.ts`
- `.docs/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint21.md`

## Deliverables

### 1. Promote the runtime-first entry contract beyond test-only use

Sprint 21 proved that `RuntimeHandle` is a valid internal owner. Sprint 22 should use that seam in more real integration paths where `Agent` is only serving as an unnecessary wrapper.

Implementation targets:

- inventory internal call sites that still instantiate `Agent` by default even though they are really consuming runtime-owned behavior, especially around:
  - launcher/bootstrap helpers
  - interactive CLI/TUI integration seams
  - harnesses and utilities that are not actually testing public compatibility
- define a small runtime-first entry contract for internal integrations that need:
  - runtime bootstrap/session ownership
  - launcher/public-shell execution
  - inspection or session continuity hooks
- migrate a bounded but real set of internal callers to that runtime-first seam
- explicitly document what remains intentionally public-shell-only versus what is now runtime-first by default

The goal is not to delete `Agent`. The goal is to make `Agent` more clearly public/compatibility-facing while real internal integrations stop depending on it by habit.

### 2. Introduce typed verification observations closer to execution time

Loader is better at explaining missing evidence now, but it still often reconstructs that explanation after the fact from session and DoD state.

Implementation targets:

- inventory where verification evidence is currently inferred or flattened after execution across:
  - `src/loader/runtime/dod.py`
  - `src/loader/runtime/finalization.py`
  - `src/loader/runtime/task_completion.py`
  - `src/loader/runtime/completion_policy.py`
  - `src/loader/runtime/session.py`
  - `src/loader/runtime/workflow_policy.py`
- define a typed verification-observation contract that can represent things like:
  - verification command requested
  - verification command ran
  - verification command passed or failed
  - verification command output backed or contradicted a claimed result
  - verification attempt was skipped, stale, or still missing
  - observed artifact/touchpoint evidence that materially supported or blocked completion
- decide whether those observations should live directly in the canonical workflow timeline, as a derived session sub-artifact, or as a canonical companion model with a single clear ownership story
- avoid inventing a second peer truth beside the canonical policy story

The goal is to move Loader’s accountability closer to “this is what was actually observed” rather than “this is what the runtime later summarized.”

### 3. Strengthen stop/continue proof using observed verification and artifact evidence

Sprint 21 made evidence more structured, but it still leaves some stop/continue decisions too dependent on runtime-authored summaries.

Implementation targets:

- connect completion/finalization decisions more directly to:
  - typed verification observations
  - active DoD acceptance state
  - tracked pending items
  - observed artifact/touchpoint evidence
  - contradictions already captured in the workflow ledger or drift evidence
- prefer explicit proof stories like:
  - “verification command X passed and covers acceptance boundary Y”
  - “artifact Z was updated and verified against the planned touchpoint set”
  - “completion still failed because verification command X never ran / failed / contradicted the claim”
  over broader fallback summaries when the runtime has stronger evidence available
- where Loader still cannot prove completion, preserve the exact missing or contradictory observation set in the policy/accountability story

The goal is not to build a deep theorem prover. The goal is to keep pushing Loader from heuristic completion toward evidence-backed completion without bluffing.

### 4. Sharpen operator inspection around observed verification state

Sprint 21 made policy evidence easier to read. Sprint 22 should make observed verification attempts easier to audit from the same existing product surfaces.

Implementation targets:

- improve the existing operator views so users can answer:
  - which verification command last ran?
  - did it pass, fail, or never run?
  - what output or observed artifact actually backed the stop/continue decision?
  - what evidence is still missing versus already satisfied?
- prefer improving:
  - `loader status`
  - `loader session show`
  - `loader workflow show`
  over inventing a new command unless a new surface is clearly cleaner
- keep concise rollups first, and expose deeper verification/observation detail only where it materially improves post-mortem debugging

The goal is to make Loader easier to audit after the fact, not simply more verbose.

## Testing strategy

- unit coverage for:
  - runtime-first entry helpers adopted below `Agent`
  - typed verification-observation normalization and persistence
  - derived policy/accountability projections from observed verification state
- runtime coverage for:
  - successful completion with explicit observed verification backing
  - failed or missing verification that now produces a more concrete stop/continue story
  - internal integration paths that now use the runtime-first entry contract
- regression coverage for:
  - no drift back toward `Agent` as the default internal seam when a runtime-first contract exists
  - no duplicate truth beside the canonical policy/accountability story
  - no regression in Sprint 21’s evidence provenance, policy rollups, or runtime-handle contract

## Definition of done

- Loader has at least one more real internal integration path using a runtime-first entry contract below `Agent`
- verification/accountability captures more first-class observed state instead of only post-hoc summaries
- stop/continue decisions can point to clearer observed proof or missing proof
- existing status/session/workflow surfaces expose that stronger verification/accountability story without multiplying commands
- Sprint 21’s runtime-handle, provenance, and grouped policy-evidence gains remain green

## Explicitly out of scope

- deleting `Agent` as the public compatibility surface
- full claw-code policy-engine parity
- model-authored verifier narratives as a required runtime dependency
- AST-aware semantic diffs
- a broad visual workflow UI
- multi-agent or team orchestration

## Audit

### Status

- Sprint 22 is complete on the verification-observation and accountability lane, and the audit is green. Loader now captures typed verification observations closer to execution, carries them through the canonical policy story, and exposes that observed verification state directly in the existing operator surfaces.
- The planned runtime-first entry promotion beyond test-only use did not land in this sprint. That debt is now an explicit Sprint 23 carry-forward item, not an implied cleanup tail.

### Landed

- verification observations are now a first-class runtime contract instead of a reconstructed afterthought: `src/loader/runtime/verification_observations.py`, `src/loader/runtime/finalization.py`, `src/loader/runtime/workflow_policy.py`, `src/loader/runtime/policy_timeline.py`, `src/loader/runtime/completion_trace.py`, and `src/loader/runtime/turn_completion.py` now preserve typed observed verification state through the DoD gate, canonical policy events, and projected completion traces
- stop/continue policy is more explicit about why Loader stopped: `src/loader/runtime/task_completion.py`, `src/loader/runtime/completion_policy.py`, and `src/loader/runtime/turn_completion.py` now use observed verification facts when they exist and preserve those facts on exhausted continuation failures instead of only falling back to generic missing-evidence language
- operator inspection is sharper without multiplying surfaces: `src/loader/runtime/workflow_timeline_read_model.py`, `src/loader/runtime/inspection.py`, and `src/loader/cli/main.py` now surface observed verification in `loader status`, `loader session show`, and `loader workflow show`, including a unified `Recent Verification` view sourced from canonical policy observations first and DoD evidence second

### Verification

- `uv run pytest -q` is green: `388 passed`
- `tests/test_verification_observations.py`, `tests/test_finalization.py`, `tests/test_completion_policy.py`, and `tests/test_turn_completion.py` now pin the verification-observation contract through finalization and completion stop policy
- `tests/test_workflow_timeline_read_model.py` and `tests/test_inspection.py` now cover observed-verification rollups, the unified recent-verification view, and the operator-facing explanation strings sourced from canonical policy state

### Residual debt

- Sprint 22 intentionally did not complete Deliverable 1. Loader still has a runtime-first internal owner from Sprint 21, but this sprint did not promote additional real integration paths away from `Agent`
- verification/accountability is more observed and audit-friendly now, but it is still bounded and runtime-authored; Loader still stops short of deeper OMX-style verifier reasoning, richer artifact-derived proof, or model-assisted audit narratives
- the existing status/session/workflow surfaces are clearer now, but Loader still stops short of claw-code's fuller policy engine, narrower runtime-first public API, and richer rule/prompt accountability surfaces
