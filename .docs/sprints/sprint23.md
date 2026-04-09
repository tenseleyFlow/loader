# Sprint 23: Runtime-First Integrations, Verification Producers, and Facade Narrowing

## Prerequisites

Sprint 22

## Goals

Take the next honest step after Sprint 22: stop treating the runtime-first owner as mostly a testing seam, move more real integrations onto that seam where `Agent` is only habit, and widen the new verification-observation contract from a finalization/result story into a more direct execution-time producer story.

Sprint 22 improved the remaining debt in a useful way:

- Loader now carries typed verification observations through canonical policy events and completion-stop decisions
- the operator surfaces can now explain recent verification from canonical policy observations first instead of stitching together post-hoc summaries
- the canonical workflow timeline is stronger as the accountability story
- but the planned runtime-first entry promotion beyond tests did not land
- `Agent` still remains the default construction seam for many real internal paths even though Loader now has a runtime-owned internal handle and explicit public facade boundaries
- verification observations are still strongest at the DoD/finalization edge; Loader still captures less from the earlier verification lifecycle than it now knows how to represent

Sprint 23 should keep using the references as architectural guardrails, not as a feature-copy list.

The standard remains:

- use claw-code to sharpen runtime-first bootstrap/session ownership, event capture, and narrower public boundaries
- use OMX to sharpen direct verifier-observation capture and evidence-backed accountability
- do not add work just because the refs have it
- do add work when the refs show that Loader is still too shell-bound, too late in its evidence capture, or too fuzzy about which boundary owns what

`audit.txt` remains a guardrail against wrapper-heavy drift and soft compatibility habits. It is not the factual roadmap.

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
- `.docs/sprints/sprint22.md`

## Deliverables

### 1. Promote the runtime-first entry contract into real internal integrations

Sprint 22 left this as explicit debt. Sprint 23 should land it for real.

Implementation targets:

- inventory internal call sites that still instantiate or route through `Agent` by default even though they are consuming runtime-owned behavior, especially around:
  - launcher/bootstrap helpers
  - CLI/TUI integration seams that are not actually testing the public compatibility contract
  - harnesses, utilities, and inspection/session helpers that only need runtime ownership
- define or refine a small runtime-first entry contract for internal consumers that need:
  - runtime bootstrap/session ownership
  - launcher/public-shell execution
  - inspection, continuity, or workflow/accountability hooks
- migrate a bounded but real set of internal integrations onto that seam
- explicitly document what remains intentionally public-shell-only versus what is now runtime-first by default

The goal is not to delete `Agent`. The goal is to stop using it internally by reflex where a runtime-owned seam is cleaner and already exists.

### 2. Expand verification observations to earlier execution-time producers

Sprint 22 made verification observations real, but they still enter the story relatively late.

Implementation targets:

- inventory where verification-related facts are currently available earlier than finalization across:
  - `src/loader/runtime/dod.py`
  - `src/loader/runtime/finalization.py`
  - `src/loader/runtime/tool_batches.py`
  - `src/loader/runtime/executor.py`
  - `src/loader/runtime/workflow_lanes.py`
  - `src/loader/runtime/workflow_policy.py`
- identify which observation kinds should be emitted closer to execution, such as:
  - verification command planned/requested
  - verification command actually executed
  - verification output observed and classified as passed/failed/contradictory
  - verification was intentionally skipped, stale, or still pending
  - observed artifact/touchpoint evidence materially backed or blocked completion before finalization
- thread those earlier observations into the canonical policy story without creating a peer truth beside the workflow timeline

The goal is to move Loader’s accountability closer to “this is what we observed while verification happened” instead of only “this is what the runtime concluded later.”

### 3. Narrow the remaining public facade boundary on purpose

Sprint 20 settled `Agent` as the public shell. Sprint 23 should keep making that shell smaller and more explicit.

Implementation targets:

- inventory what still lives in `src/loader/agent/loop.py` and nearby public-shell glue
- identify which pieces are:
  - true public compatibility API
  - UI integration seam
  - leftover runtime ownership that can move below the shell now
- move the still-obviously-runtime pieces below the public shell where that reduces ambiguity
- add or extend direct boundary tests so internal code does not drift back toward `Agent` ownership once a runtime seam exists

The goal is not “make the file smaller” for its own sake. The goal is that future work has a clearer answer to what the public shell is for.

### 4. Sharpen operator visibility for runtime-first ownership and observed verification

Once more of the real integration paths go runtime-first and more observations are captured earlier, the existing product surfaces should make that easier to audit.

Implementation targets:

- improve the existing surfaces so users can answer:
  - which runtime-owned path produced the current session/accountability state?
  - what verification was actually observed earlier in the turn versus only concluded at finalization?
  - what evidence is pending, contradicted, or already satisfied?
- prefer improving:
  - `loader status`
  - `loader session show`
  - `loader workflow show`
  over inventing a new command unless a new surface is clearly cleaner
- keep concise rollups first, and expose deeper ownership/observation detail only where it materially improves post-mortem debugging

The goal is to make Loader easier to audit after the fact, not simply more verbose.

## Testing strategy

- unit coverage for:
  - runtime-first entry helpers adopted in real internal integration paths
  - earlier verification-observation producer normalization and persistence
  - public-shell boundary helpers and import/boundary guards
- runtime coverage for:
  - successful completion with observed verification facts emitted before finalization
  - failed or pending verification that now leaves a clearer producer-backed policy trail
  - internal integration paths that no longer need `Agent` by default
- regression coverage for:
  - no drift back toward `Agent` as the default internal seam when a runtime-owned seam exists
  - no duplicate truth beside the canonical policy/accountability story
  - no regression in Sprint 22’s observed-verification inspection and stop/continue honesty

## Definition of done

- Loader has at least one more real internal integration path using a runtime-first entry seam below `Agent`
- verification observations are emitted from at least one earlier execution-time producer instead of only the finalization edge
- the remaining public-shell boundary is smaller or more explicitly defended on purpose
- existing status/session/workflow surfaces expose the stronger runtime-first and verification-observation story without multiplying commands
- Sprint 22’s observed-verification and accountability gains remain green

## Explicitly out of scope

- deleting `Agent` as the public compatibility surface
- full claw-code policy-engine parity
- model-authored verifier narratives as a required runtime dependency
- AST-aware semantic diffs
- a broad visual workflow UI
- multi-agent or team orchestration
