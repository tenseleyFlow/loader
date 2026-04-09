# Sprint 20: Canonical Policy Events, Verifier-Backed Follow-Through, and Facade Settlement

## Prerequisites

Sprint 19

## Goals

Take the next honest step after Sprint 19: stop treating policy accountability as a set of coordinated side channels, strengthen follow-through requirements with better runtime evidence, and decide what the public shell should actually remain responsible for now that most of the runtime owns itself.

Sprint 19 improved the shape of the remaining debt again:

- `src/loader/agent/loop.py` is now much smaller and more facade-like
- continuation behavior is more honest because missing follow-through can now end in explicit failure instead of silent acceptance
- operators can inspect policy accountability more coherently through `loader workflow show --policy` and the `Policy Timeline` preview in `loader session show`
- but completion traces and workflow timeline entries still coexist as separate persisted artifacts
- follow-through evidence is typed now, but it is still heuristic and runtime-authored rather than grounded in a stronger verifier/evidence contract
- Loader is much closer to an intentional public boundary, but it still has not explicitly settled whether the remaining `Agent` shell is the desired long-term seam or simply the next temporary contraction point

Sprint 20 should keep using the references as architectural guardrails, not as a porting checklist.

The standard remains:

- use claw-code to sharpen canonical runtime ownership, policy/event contracts, and session/runtime seams
- use OMX to sharpen verifier-backed follow-through requirements, evidence accounting, and operator-facing accountability
- do not add work just because the refs have it
- do add work when the refs show that Loader is still too heuristic, too fragmented, or too hard to audit

`audit.txt` remains a guardrail against backsliding into wrappers and soft rescue behavior. It is not the factual roadmap.

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
- `.docs/sprints/sprint19.md`

## Deliverables

### 1. Define one canonical persisted policy-event contract

Sprint 19 made the operator story clearer, but Loader still persists policy accountability in more than one shape.

Implementation targets:

- inventory the current persisted/debug policy surfaces across:
  - `src/loader/runtime/completion_trace.py`
  - `src/loader/runtime/policy_timeline.py`
  - `src/loader/runtime/workflow_policy.py`
  - `src/loader/runtime/session.py`
  - `src/loader/runtime/events.py`
- decide what the canonical persisted policy/accountability artifact should be:
  - extend the workflow timeline into the sole source of truth
  - or introduce a first-class policy-event model and derive the old surfaces from it
- make sure the canonical event contract can express:
  - workflow routing/handoff/reentry
  - repair retries/failures
  - completion accept/continue/finalize outcomes
  - verify skips and explicit stop reasons
  - evidence summaries and prompt/runtime context
- avoid preserving multiple peer artifacts unless one is clearly a compatibility/read-model projection of the other

The goal is that Loader has one canonical answer to “what policy decisions happened during this turn/session?” instead of two overlapping persistence models that merely render similarly.

### 2. Move follow-through requirements closer to a verifier-backed contract

Sprint 19 added typed required/missing evidence. Sprint 20 should make that contract less purely heuristic.

Implementation targets:

- inventory where follow-through requirements are still inferred from weak textual cues across:
  - `src/loader/runtime/task_completion.py`
  - `src/loader/runtime/completion_policy.py`
  - `src/loader/runtime/turn_completion.py`
  - any nearby DoD/verification/session helpers that already know more than the completion heuristic does
- define a stronger runtime completion-requirements contract using available structured evidence such as:
  - task class and workflow mode
  - DoD state and acceptance criteria
  - verification plans and prior verification results
  - actual tool history, artifact paths, and session/runtime evidence
- prefer explicit requirements like:
  - “verification command ran and passed”
  - “mutating touchpoint is accounted for in the active artifact set”
  - “claimed result is backed by observed output”
  over generic “probably incomplete” heuristics
- where the runtime still cannot prove completion, stop honestly and preserve the exact missing requirement set

The goal is to move closer to claw-code’s green-contract thinking and OMX’s verifier/accountability discipline without forcing Loader into a fake model-assisted verifier that it cannot honestly support yet.

### 3. Settle the intended long-term public shell boundary

By Sprint 19, `Agent` is much smaller. Sprint 20 should decide what remains on purpose.

Implementation targets:

- inventory the current responsibilities still living in `src/loader/agent/loop.py`
- identify which of those are:
  - true public API / compatibility surface
  - UI integration seam
  - leftover runtime or launcher ownership
- move remaining runtime-ish behavior below the public shell where that is still obviously correct
- explicitly document what stays in `Agent` and why, so future sprints stop treating the shell as a vague cleanup target

The goal is not “delete the public shell.” The goal is to stop having an ambiguous shell.

### 4. Make the operator-facing accountability story sharper without multiplying product surfaces

Sprint 19 improved inspection, but there is still room to reduce cognitive stitching.

Implementation targets:

- improve the existing policy-facing views so operators can answer:
  - why did Loader continue?
  - why did Loader stop?
  - what evidence was still missing?
  - what policy stage made the final decision?
- prefer improving:
  - `loader workflow show`
  - `loader session show`
  - `loader status`
  over inventing a brand-new inspection command unless a new surface is clearly cleaner
- where useful, add concise rollup/highlight views that summarize the last important policy event rather than requiring the user to parse the whole timeline manually

The goal is to make Loader easier to audit after the fact, not simply more verbose.

## Testing strategy

- unit coverage for:
  - the canonical policy-event serialization/restoration contract
  - verifier-backed completion-requirement derivation
  - any reduced/settled public-shell boundary helpers
- runtime coverage for:
  - honest finalization when runtime evidence still fails the completion contract
  - no regression in successful follow-through on normal non-mutating and mutating tasks
  - session/workflow/status inspection of the canonical policy story
- regression coverage for:
  - no drift back toward duplicate persisted policy artifacts without a canonical source of truth
  - no reintroduction of soft continuation acceptance after missing-evidence finalization
  - no drift back toward `agent/loop.py` accumulating runtime ownership because the shell boundary is still ambiguous

## Definition of done

- Loader has one clearly canonical persisted policy/accountability contract
- follow-through requirements are more explicitly grounded in runtime/verifier evidence instead of mostly textual heuristics
- the remaining public-shell responsibilities are materially smaller or explicitly settled on purpose
- operators can answer the main stop/continue/retry questions from one clearer set of existing inspection surfaces
- Sprint 19’s honesty and policy-inspection gains remain green

## Explicitly out of scope

- full claw-code policy-engine parity
- model-authored verifier narratives as a new mandatory runtime dependency
- multi-agent or team orchestration
- AST-aware semantic diffs
- a broad visual workflow UI
- rich permission-rule editing UX
