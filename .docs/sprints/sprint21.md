# Sprint 21: Evidence Provenance, Read-Model Cleanup, and Runtime-First API

## Prerequisites

Sprint 20

## Goals

Take the next honest step after Sprint 20: move Loader's completion and verification story from "better heuristics with canonical policy events" toward stronger evidence provenance, reduce compatibility read-model duplication where the canonical workflow timeline already carries the truth, and begin narrowing internal callers toward a runtime-first API instead of treating `Agent` as the only natural seam.

Sprint 20 changed the remaining debt in a useful way:

- the workflow timeline is now the canonical policy/accountability artifact, including live completion-trace projection
- follow-through checks now use stronger DoD/runtime evidence instead of only textual heuristics
- the remaining `Agent` shell is explicitly documented and guarded as a public facade
- but completion/verification evidence is still mostly flattened into human-readable strings rather than typed provenance
- compatibility/read-model surfaces still rely on a few projections that are honest but not yet minimal
- internal callers still treat `Agent` as the default runtime entry seam even though the shell is now explicitly a compatibility/public facade

Sprint 21 should keep using the references as architectural guardrails, not as a feature-copy list.

The standard remains:

- use claw-code to sharpen canonical event ownership, green-contract discipline, and runtime-first seams
- use OMX to sharpen verifier/accountability provenance and evidence-backed follow-through
- do not add work just because the refs have it
- do add work when the refs show that Loader is still too stringly-typed, too duplicative, or too dependent on a compatibility shell

`audit.txt` remains a guardrail against wrapper-heavy drift and soft rescue behavior. It is not the factual roadmap.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/green_contract.rs`
- `refs/claw-code/rust/crates/runtime/src/lane_events.rs`
- `refs/claw-code/rust/crates/runtime/src/session_control.rs`
- `refs/claw-code/rust/crates/runtime/src/bootstrap.rs`
- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/PARITY.md`
- `refs/oh-my-codex/src/verification/verifier.ts`
- `refs/oh-my-codex/src/autoresearch/contracts.ts`
- `refs/oh-my-codex/src/autoresearch/runtime.ts`
- `refs/oh-my-codex/src/hooks/session.ts`
- `refs/oh-my-codex/src/hooks/prompt-guidance-contract.ts`
- `.docs/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint20.md`

## Deliverables

### 1. Introduce typed evidence provenance for completion and verification

Sprint 20 strengthened follow-through, but most of the contract still collapses into free-text evidence summaries too early.

Implementation targets:

- inventory where completion/verification evidence is currently flattened into strings across:
  - `src/loader/runtime/task_completion.py`
  - `src/loader/runtime/completion_trace.py`
  - `src/loader/runtime/policy_timeline.py`
  - `src/loader/runtime/finalization.py`
  - `src/loader/runtime/dod.py`
  - `src/loader/runtime/workflow_policy.py`
- define a small typed provenance model that can represent things like:
  - verification command ran and passed/failed
  - verification command was still missing
  - tracked work item remained incomplete
  - artifact/touchpoint evidence existed or was contradicted
  - claimed runtime outcome was backed by observed output
- prefer structured provenance that can still be rendered into human-readable summaries, instead of making strings the primary contract
- thread that provenance through completion policy and canonical policy events where it materially improves honesty or inspectability

The goal is not to build a fake theorem prover. The goal is to stop throwing away runtime evidence structure too early.

### 2. Reduce read-model duplication around the canonical workflow timeline

Sprint 20 made the workflow timeline canonical, but a few read models still feel more coupled than they need to be.

Implementation targets:

- inventory where compatibility/read-model projections still depend on direct mutation or duplicated logic across:
  - `src/loader/runtime/completion_trace.py`
  - `src/loader/runtime/session.py`
  - `src/loader/runtime/inspection.py`
  - `src/loader/runtime/events.py`
  - any nearby status/session helper that reconstructs policy state manually
- make sure projections like completion traces and latest-policy summaries are clearly derivations from canonical policy events instead of semi-independent contracts
- remove any remaining direct writes or state bookkeeping that are only there to keep parallel policy read models in sync
- keep compact operator-facing read models where they help, but make their derived nature explicit in code and tests

The goal is one canonical truth plus honest projections, not a forest of near-duplicates.

### 3. Start the runtime-first internal API transition below the public `Agent` facade

Sprint 20 settled `Agent` as the public compatibility shell. Sprint 21 should stop using that shell as the default internal seam where it no longer needs to be.

Implementation targets:

- inventory current internal call sites that still instantiate or consume `Agent` when a runtime-first seam would be cleaner, especially in:
  - launcher/bootstrap helpers
  - CLI/TUI integration code
  - tests that are really exercising runtime behavior rather than public compatibility
- define a small runtime-first entry contract for internal consumers where it clearly reduces shell coupling
- keep `Agent` as the public compatibility surface, but begin migrating internal runtime-oriented callers away from assuming that `Agent` is the only valid execution owner
- document what remains intentionally public-shell-only versus what is now runtime-first

The goal is not to delete `Agent`. The goal is to make `Agent` clearly public/compatibility-facing while runtime internals use runtime-first seams by default.

### 4. Sharpen operator visibility for evidence-backed stop/continue decisions

Sprint 20 improved policy summaries, but the evidence itself is still only partially visible.

Implementation targets:

- improve the existing operator views so users can answer:
  - what exact evidence was missing when Loader stopped?
  - what exact evidence satisfied the completion contract?
  - which policy event carried that evidence?
- prefer improving:
  - `loader workflow show`
  - `loader session show`
  - `loader status`
  over inventing a new command unless a new surface is clearly cleaner
- add concise rollups first, and expose deeper provenance only where it materially helps post-mortem inspection

The goal is to make Loader easier to audit after the fact, not simply more verbose.

## Testing strategy

- unit coverage for:
  - typed evidence-provenance normalization/rendering
  - derived read-model projections from the canonical workflow timeline
  - any new runtime-first internal entry contract below `Agent`
- runtime coverage for:
  - honest finalization with explicit evidence provenance when completion still fails
  - successful completion paths that now surface structured proof instead of only summary strings
  - status/session/workflow inspection of the evidence-backed policy story
- regression coverage for:
  - no drift back toward peer policy artifacts beside the canonical workflow timeline
  - no drift back toward `Agent` as the default internal seam when a runtime-first contract exists
  - no loss of the current compact operator read models while provenance becomes richer

## Definition of done

- Loader preserves one canonical policy/accountability artifact while making evidence provenance more structured
- completion/verification evidence is less stringly-typed and more inspectable without weakening honesty
- internal runtime-oriented code has at least one cleaner runtime-first seam below the public `Agent` facade
- existing status/session/workflow surfaces answer stop/continue questions with clearer evidence context
- Sprint 20's canonical-policy and facade-settlement gains remain green

## Explicitly out of scope

- full claw-code policy-engine parity
- model-authored verifier narratives as a mandatory dependency
- multi-agent or team orchestration
- AST-aware semantic diffs
- a broad visual workflow UI
- rich permission-rule editing UX
