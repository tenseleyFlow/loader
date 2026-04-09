# Sprint 25: Public Runtime API, Verification Attempts, and Boundary Narrowing

## Prerequisites

Sprint 24

## Goals

Take the next honest step after Sprint 24: stop treating `Agent` plus `runtime.public_shell` as the only meaningful outer boundary by default, deepen verification from coarse lifecycle labels into explicit attempt semantics, and keep pushing Loader toward a runtime-first shape without pretending it is ready to delete the public compatibility surface.

Sprint 24 changed the remaining debt in a useful way:

- CLI, explore, the scripted harness, and the TUI now all use the runtime-first owner seam below `Agent`
- verification lifecycle now distinguishes `planned`, `pending`, `stale`, `skipped`, and observed states inside the canonical policy timeline
- operator surfaces can now explain that lifecycle directly in `status`, `session show`, and `workflow show`
- but `Agent` plus `runtime.public_shell` still remain the outer compatibility shell instead of a narrower runtime-first external API
- verification lifecycle is still a bounded runtime-authored label model, not an explicit record of verification attempts, queueing, start/completion moments, or freshness across retries
- Loader can now say that verification is planned or pending, but it still says less than it should about which verification attempt is active, what superseded it, and what evidence belongs to which attempt

Sprint 25 should keep using the references as architectural guardrails, not as a feature-copy list.

The standard remains:

- use claw-code to sharpen outer runtime boundaries, bootstrap/session ownership, and event accountability
- use OMX to sharpen verifier-attempt visibility, freshness semantics, and auditability around incomplete versus superseded proof
- do not add work just because the refs have it
- do add work when the refs show that Loader is still too compatibility-shell-bound or too coarse in its verification state model

`audit.txt` remains a guardrail against wrapper-heavy drift and compatibility-by-habit. It is not the factual roadmap.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/bootstrap.rs`
- `refs/claw-code/rust/crates/runtime/src/session_control.rs`
- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/lane_events.rs`
- `refs/claw-code/rust/crates/runtime/src/green_contract.rs`
- `refs/claw-code/PARITY.md`
- `refs/oh-my-codex/src/verification/verifier.ts`
- `refs/oh-my-codex/src/autoresearch/runtime.ts`
- `refs/oh-my-codex/src/autoresearch/contracts.ts`
- `refs/oh-my-codex/src/hooks/session.ts`
- `.docs/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint24.md`

## Deliverables

### 1. Define a narrower public runtime API below `Agent`

Sprint 24 made runtime-first real across the major product paths. Sprint 25 should make the remaining outer boundary more explicit and less compatibility-shaped.

Implementation targets:

- inventory what external callers actually need from:
  - `src/loader/agent/loop.py`
  - `src/loader/runtime/public_shell.py`
  - `src/loader/runtime/runtime_handle.py`
  - `src/loader/runtime/bootstrap.py`
  - `src/loader/cli/main.py`
  - `src/loader/ui/app.py`
- define a smaller runtime-owned API for external integrations that need:
  - runtime bootstrap/session ownership
  - shell execution entrypoints
  - inspection/continuity hooks
  - steering/question handling
- explicitly settle what stays public-compat-only under `Agent` and what should become runtime-first by default
- migrate at least one more real caller or integration seam onto that narrower runtime-first API if the seam is still compatibility-driven by habit

The goal is not to delete `Agent`. The goal is to make Loader's outer boundary answerable instead of half public facade and half runtime shell by historical accident.

### 2. Promote verification lifecycle labels into typed attempt semantics

Sprint 24 gave Loader lifecycle states. Sprint 25 should give those states explicit attempt structure.

Implementation targets:

- inventory where Loader already knows more than a plain status label across:
  - `src/loader/runtime/dod.py`
  - `src/loader/runtime/finalization.py`
  - `src/loader/runtime/task_completion.py`
  - `src/loader/runtime/tool_batches.py`
  - `src/loader/runtime/workflow_policy.py`
  - `src/loader/runtime/policy_timeline.py`
  - `src/loader/runtime/verification_observations.py`
- define a typed verification-attempt model that can represent things like:
  - verification planned as a future attempt
  - verification queued/pending as the current active attempt
  - verification started versus completed
  - verification superseded or made stale by later mutating work
  - verification intentionally skipped
  - verification attempt results tied to the right command/evidence bundle
- keep that attempt model inside the canonical policy/accountability story instead of creating a second peer verification log with a separate ownership story

The goal is to make Loader answer not just "is verification pending?" but "which verification attempt is pending, what superseded the last one, and what evidence belongs to this attempt?"

### 3. Tighten completion and freshness policy around verification attempts

Once attempt semantics exist, completion policy should stop flattening them back into generic lifecycle summaries.

Implementation targets:

- connect completion and reentry decisions more directly to:
  - active verification attempt identity
  - attempt freshness relative to later mutations
  - attempt result timestamps/order
  - superseded/stale attempt reasoning
  - explicit missing proof versus not-yet-finished proof
- preserve a clear distinction between:
  - proof that is planned but not started
  - proof currently in flight
  - proof that finished and passed
  - proof that finished and failed
  - proof that was once green but is no longer fresh
- ensure the canonical policy story explains why a stop/continue/retry decision was tied to one verification attempt rather than another

The goal is to make Loader's completion honesty stronger when verification gets interrupted, superseded, retried, or resumed.

### 4. Improve operator visibility for runtime boundary and verification attempts

Once Loader has a narrower runtime-first boundary and richer attempt semantics, the existing operator surfaces should make that audit story easier to follow.

Implementation targets:

- improve the current surfaces so users can answer:
  - which runtime/public boundary handled this session?
  - what is the current verification attempt?
  - what earlier attempt became stale or was superseded?
  - what evidence is attached to the active versus superseded attempt?
- prefer improving:
  - `loader status`
  - `loader session show`
  - `loader workflow show`
  - the TUI status surface
  over inventing a new command unless a new surface is clearly cleaner
- keep concise rollups first and expose deeper attempt detail only where it materially improves debugging

The goal is to make Loader's runtime ownership and verification story easier to audit after the fact, not simply more verbose.

## Testing strategy

- unit coverage for:
  - the narrower runtime-first public API contract
  - verification-attempt normalization, persistence, and supersession
  - completion freshness rules that now depend on attempt semantics
- runtime coverage for:
  - a verify handoff that records a planned attempt before active verification starts
  - a pending attempt that later completes or becomes stale after fresh mutating work
  - runtime-first callers that no longer need `Agent` by default
- regression coverage for:
  - no drift back toward `Agent` as the assumed external owner when a runtime-first API exists
  - no duplicate verification-attempt truth beside the canonical policy timeline
  - no regression in Sprint 24's lifecycle visibility and runtime-first TUI ownership

## Definition of done

- Loader has a narrower and more explicit runtime-first external API below `Agent`, or any remaining `Agent` ownership is clearly justified as public compatibility
- verification lifecycle is represented with explicit attempt semantics, not only coarse status labels
- completion and freshness policy can explain which verification attempt is active, stale, superseded, or satisfied
- existing status/session/workflow/TUI surfaces expose the stronger boundary and attempt story without multiplying product commands
- Sprint 24's runtime-first ownership and verification lifecycle gains remain green

## Explicitly out of scope

- deleting `Agent` as the public compatibility surface
- full claw-code policy-engine parity
- model-authored verifier narratives as a required runtime dependency
- AST-aware semantic diffs
- a broad visual workflow UI redesign
- multi-agent or team orchestration
