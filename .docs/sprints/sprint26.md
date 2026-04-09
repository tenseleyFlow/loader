# Sprint 26: Verification Attempt Timelines and Public Facade Delamination

## Prerequisites

Sprint 25

## Goals

Take the next honest step after Sprint 25: preserve more than attempt labels in the verifier lifecycle, make the runtime-owned API more authoritative as Loader's practical outer boundary, and keep shrinking the gap between "runtime-first internally" and "runtime-first by default" without pretending the public compatibility surface is ready to disappear.

Sprint 25 changed the remaining debt in a useful way:

- Loader now has a runtime-owned shell API boundary in `src/loader/runtime/runtime_api.py`
- verification lifecycle now preserves explicit attempt identity across planned, pending, stale, skipped, and observed states
- completion policy can explain active versus superseded proof in attempt-aware terms
- operator surfaces can now show runtime-boundary summaries and attempt-aware verification state in CLI and TUI
- but Loader still says more about attempt labels than about attempt timeline facts such as when an attempt was first planned, when it actually started, when it completed, and what exactly superseded it
- and `Agent` plus `runtime.public_shell` still remain the documented public compatibility shell even though the runtime-owned boundary is much stronger now

Sprint 26 should keep using the references as architectural guardrails, not as a feature-copy list.

The standard remains:

- use claw-code to sharpen runtime/bootstrap/session ownership, policy/accountability event structure, and explicit outer runtime boundaries
- use OMX to sharpen verifier attempt visibility, freshness reasoning, and the audit trail around incomplete, stale, superseded, or resumed proof
- do not add work just because the refs have it
- do add work when the refs show that Loader is still too compatibility-shell-bound or too coarse in its verification-attempt history

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
- `.docs/sprints/sprint25.md`

## Deliverables

### 1. Promote verification attempts into richer timeline records

Sprint 25 gave Loader attempt identity. Sprint 26 should make those attempts feel like real runtime history, not only labeled states.

Implementation targets:

- inventory where Loader already knows attempt-order or lifecycle facts across:
  - `src/loader/runtime/dod.py`
  - `src/loader/runtime/finalization.py`
  - `src/loader/runtime/tool_batches.py`
  - `src/loader/runtime/verification_observations.py`
  - `src/loader/runtime/workflow_policy.py`
  - `src/loader/runtime/policy_timeline.py`
  - `src/loader/runtime/workflow_timeline_read_model.py`
- define a typed verification-attempt timeline model that can preserve things like:
  - when an attempt was first planned
  - when it actually became active
  - when it completed or was skipped
  - when and why it became stale or was superseded
  - which commands/evidence bundle belonged to that attempt
- keep that richer attempt model inside the canonical policy/accountability story instead of creating a side log

The goal is to make Loader answer not only "which attempt is active?" but "what happened to attempt 2, when did attempt 3 actually start, and what proof belongs to each?"

### 2. Tighten completion/freshness policy around attempt history

Once attempt history is richer, completion policy should stop flattening that history into one summary string too early.

Implementation targets:

- connect completion and reentry decisions more directly to:
  - attempt planning versus active-start moments
  - completed versus superseded attempt ordering
  - freshness relative to later mutating work
  - explicit gaps between planned proof, running proof, and finished proof
- preserve a clear distinction between:
  - proof that is scheduled but has not started
  - proof that started but has not completed
  - proof that completed and passed
  - proof that completed and failed
  - proof that was once green but is now stale because a later attempt superseded it
- ensure the canonical policy story explains why Loader trusted, rejected, or waited on one attempt instead of another

The goal is to make Loader's stop/continue/retry logic more auditable when verification spans multiple retries or resumes.

### 3. Narrow the public facade below `Agent` again

Sprint 25 added a runtime-owned shell API. Sprint 26 should make that boundary more authoritative in practice.

Implementation targets:

- inventory what still materially depends on:
  - `src/loader/agent/loop.py`
  - `src/loader/runtime/public_shell.py`
  - `src/loader/runtime/runtime_api.py`
  - `src/loader/runtime/runtime_handle.py`
  - `src/loader/cli/main.py`
  - `src/loader/ui/app.py`
- migrate at least one more real caller or ownership seam onto the runtime-owned API if it is still compatibility-shaped by habit
- make remaining `Agent`-owned behavior explicitly compatibility-facing rather than ambiguous runtime glue
- add or extend boundary tests so future work does not drift back toward `Agent` as the assumed outer runtime owner

The goal is not to delete `Agent`. The goal is to make the runtime-owned API the default answer more often, and the compatibility facade the deliberate exception.

### 4. Improve operator visibility for attempt history and public boundary

Once attempt records get richer and the outer boundary gets cleaner, the existing surfaces should tell that story with less reconstruction.

Implementation targets:

- improve the current surfaces so users can answer:
  - which runtime/public boundary handled this session?
  - which verification attempt is active right now?
  - when did it become planned, active, stale, or superseded?
  - what evidence belongs to the active attempt versus an older one?
- prefer improving:
  - `loader status`
  - `loader session show`
  - `loader workflow show`
  - the TUI status surface
  over inventing a new command unless one is clearly cleaner
- keep concise rollups first, and expose deeper attempt history only where it materially improves debugging

The goal is to make Loader's runtime boundary and verifier history easier to audit after the fact, not simply more verbose.

## Testing strategy

- unit coverage for:
  - richer verification-attempt timeline normalization and persistence
  - completion/freshness decisions that now depend on attempt history
  - the narrower runtime-owned API boundary and any new caller migration
- runtime coverage for:
  - a planned attempt that later becomes active and then completes
  - a completed attempt that becomes stale after new mutating work
  - a resumed session whose active attempt history and runtime-owner boundary remain coherent
- regression coverage for:
  - no duplicate verification-attempt truth beside the canonical policy timeline
  - no drift back toward `Agent` as the assumed outer runtime owner when a runtime-owned API exists
  - no regression in Sprint 25's attempt-aware completion/freshness and operator surfaces

## Definition of done

- Loader preserves richer verification-attempt history than plain attempt labels inside the canonical policy/accountability story
- completion and freshness policy can explain which attempt was planned, active, completed, stale, or superseded and why
- the runtime-owned API below `Agent` is more authoritative in at least one more real integration seam, or remaining `Agent` ownership is explicitly justified as compatibility-only
- existing status/session/workflow/TUI surfaces expose the stronger boundary and attempt-history story without multiplying product commands
- Sprint 25's runtime-boundary and attempt-aware verification gains remain green

## Explicitly out of scope

- deleting `Agent` as the public compatibility surface
- full claw-code policy-engine parity
- model-authored verifier narratives as a required runtime dependency
- AST-aware semantic diffs
- a broad visual workflow UI redesign
- multi-agent or team orchestration
