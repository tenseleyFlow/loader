# Sprint 19: Facade Finalization, Continuation Hardening, and Unified Policy Timeline

## Prerequisites

Sprint 18

## Goals

Take the next honest contraction after Sprint 18: finish thinning the public shell where it still reads like runtime glue, harden the last continuation heuristics that can still feel like soft rescue behavior, and unify the scattered policy/debug surfaces into one clearer operator story.

Sprint 18 changed the shape of the remaining debt in a useful way:

- completion policy is now explicit, persisted, and inspectable
- public-shell helpers now own steering, session install/load, event-emitter normalization, and capability-refresh decisions
- `src/loader/agent/loop.py` is materially thinner
- but `Agent` still owns the public entrypoints and launcher glue
- completion traces and workflow traces now both exist, but they are still separate operator surfaces
- Loader still keeps bounded continuation nudges for some non-mutating tasks, and those nudges are explicit now but not yet deeply justified

Sprint 19 should stay reference-guided, not reference-submissive.

The standard remains:

- use claw-code to sharpen runtime seams, policy ownership, and explicit lifecycle contracts
- use OMX to sharpen follow-through, verifier pressure, and operator-facing runtime accountability
- do not add a feature just because the refs have it
- do pursue changes when the refs reveal that Loader is still too soft, too implicit, or too hard to audit

`audit.txt` is still not the roadmap. It is useful only as a guardrail against sliding back into wrapper-heavy cleanup and soft model rescue behavior.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/bootstrap.rs`
- `refs/claw-code/rust/crates/runtime/src/session_control.rs`
- `refs/claw-code/rust/crates/runtime/src/lane_events.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/green_contract.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/PARITY.md`
- `.docs/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint18.md`
- `refs/oh-my-codex/src/autoresearch/contracts.ts`
- `refs/oh-my-codex/src/autoresearch/runtime.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`
- `refs/oh-my-codex/src/hooks/session.ts`
- `refs/oh-my-codex/src/hooks/prompt-guidance-contract.ts`

## Deliverables

### 1. Finish the next public-shell contraction below `Agent`

Sprint 18 moved more shell behavior into `src/loader/runtime/public_shell.py`, but `Agent` still owns the public entrypoint wrappers and some launch-time glue.

Implementation targets:

- inventory what remains in `src/loader/agent/loop.py` that still feels like runtime/public-shell plumbing instead of true public API ownership, especially:
  - run / run_streaming / run_explore event-wrapper glue
  - resume / clear lifecycle orchestration
  - launcher construction and runtime-source preparation
  - capability-refresh and prompt invalidation wiring
- move what is reusable into runtime-owned helpers or a tighter launcher/public-shell seam
- keep `Agent` focused on:
  - public API shape
  - compatibility-facing attributes
  - minimal UI-facing integration points

The goal is not “delete `Agent`.” The goal is for `Agent` to read like an intentionally tiny facade instead of a convenient place for leftover runtime glue.

### 2. Harden the remaining continuation contract

Sprint 18 made continuation behavior visible. Sprint 19 should decide which of that behavior is still too soft.

Implementation targets:

- inventory the remaining continuation behavior across:
  - `src/loader/runtime/completion_policy.py`
  - `src/loader/runtime/turn_completion.py`
  - `src/loader/runtime/assistant_turns.py`
  - any nearby repair/finalization controller that can still nudge rather than stop
- identify which continuation cases are still justified by explicit runtime evidence versus merely tolerated by textual heuristics
- prefer:
  - deletion
  - a stricter typed stop/fail state
  - explicit follow-through requirements derived from runtime artifacts or session state
  over keeping broad “continue once more” behavior
- where a continuation path remains, make the required evidence explicit and persisted

The goal is to keep following the Sprint 13 / Sprint 17 / Sprint 18 line: the runtime should proceed for a clear typed reason or stop honestly, not continue because the model “probably meant well.”

### 3. Unify completion, workflow, and repair accountability into one operator-facing timeline

Loader now has workflow timeline entries and a separate completion trace. That is better than hidden state, but still fragmented.

Implementation targets:

- define a compact unified policy timeline or policy event model that can carry:
  - workflow routing/handoff decisions
  - completion-policy outcomes
  - repair / retry / recovery decisions
  - terminal stop reasons
- decide whether the existing workflow timeline should absorb completion/repair events or whether a sibling policy timeline is the cleaner contract
- persist enough of that state to survive resume and make post-mortem inspection more useful
- surface it through existing product seams, likely one of:
  - `loader workflow show`
  - `loader session show`
  - a narrowly-scoped new policy-focused surface if and only if it is cleaner than overloading the workflow view

The goal is that operators can answer “why did Loader keep going, stop, retry, or accept this result?” from one coherent surface instead of stitching together multiple tables by hand.

### 4. Keep the ref relationship explicit and healthy

Implementation targets:

- use claw-code for:
  - lane-event shape
  - session/runtime control seams
  - explicit policy and green-contract ownership
- use OMX for:
  - verifier/follow-through accountability
  - session/runtime operator clarity
  - stronger prompt/runtime contract thinking
- do not add work just because the refs have it
- do add work when the refs reveal a real Loader weakness in:
  - honesty
  - inspectability
  - shell minimalism
  - follow-through

The goal is to keep Loader reference-guided and self-aware, not to drift into either blind feature copying or isolated local optimization.

## Testing strategy

- unit coverage for:
  - any new public-shell or launcher helper that further reduces `Agent` ownership
  - tightened continuation/terminal-stop decisions
  - unified policy timeline serialization and restoration
- runtime coverage for:
  - no regression in normal follow-through on non-mutating and mutating tasks
  - honest terminal behavior where continuation heuristics were deleted or narrowed
  - session/workflow inspection of the unified policy/debug story
- regression coverage for:
  - no drift back toward `agent/loop.py` owning extracted shell glue
  - no silent reintroduction of soft continuation phrasing after harder stop conditions
  - no loss of the current workflow/completion/explore inspection surfaces while timelines are unified

## Definition of done

- `agent/loop.py` shrinks again or becomes materially more facade-like even if line count only drops modestly
- Loader deletes or hardens more of the remaining continuation heuristics instead of merely explaining them better
- operators can inspect one more coherent runtime policy story after the fact
- Sprint 18’s completion-trace and public-shell gains remain green
- the parity baseline remains green after the Sprint 19 shell and policy tightening

## Explicitly out of scope

- full claw-code policy-engine parity
- multi-agent or team orchestration
- AST-aware semantic diffs
- a broad visual workflow UI
- rich permission-rule editing UX
