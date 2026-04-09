# Sprint 18: Shell Minimalism, Completion Contract, and Runtime Policy Trace

## Prerequisites

Sprint 17

## Goals

Take the next honest contraction after Sprint 17: finish shrinking the public shell where it is still load-bearing, tighten the remaining completion-policy heuristics that still act like soft rescue behavior, and expose more of the runtime’s stop/continue policy as inspectable state instead of hidden control flow.

Sprint 17 closed several important seams:

- the public runtime boundary is now explicitly runtime-shaped instead of raw-`Agent` by convention
- prompt/session shell behavior moved into `src/loader/runtime/public_shell.py`
- raw-text tool recovery now fails honestly once its budget is exhausted
- explore continuity now has a small but real operator surface

That changes the remaining debt in a useful way:

- `src/loader/agent/loop.py` is now much thinner, but it still owns resume/clear lifecycle, steering plumbing, capability refresh, and public event-wrapper glue
- `src/loader/runtime/completion_policy.py` and `src/loader/runtime/turn_completion.py` still contain heuristics that may be useful, but they are the clearest remaining place where Loader can still look like it is nudging the model instead of enforcing a typed contract
- the runtime now makes more honest decisions, but operators still cannot inspect completion-policy decisions with the same clarity they can inspect permissions, prompts, workflow, or explore continuity

Sprint 18 should keep using `refs/claw-code` and `refs/oh-my-codex` as architectural references, not as a literal feature-copy checklist. Loader is no longer in the “blindly match every upstream surface” phase. The right standard now is:

- honor claw-code where it provides proven runtime seams, policy discipline, and explicit state ownership
- honor OMX where it sharpens workflow/verifier thinking
- keep Loader-specific product surfaces when they improve inspectability or fit the current Python runtime better

That means Sprint 18 should still be guided by the refs, but it should use them to make better Loader decisions rather than to perform a mechanical port.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/PARITY.md`
- `.docs/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint17.md`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`

## Deliverables

### 1. Shrink the remaining public shell to an intentionally small facade

Sprint 17 moved prompt/session helpers out of `agent/loop.py`. Sprint 18 should finish the next obvious shell contraction.

Implementation targets:

- inventory the remaining `Agent` responsibilities that still feel like runtime/public-shell plumbing rather than true product entrypoint behavior, especially:
  - resume/clear lifecycle helpers
  - capability refresh wiring
  - steering queue plumbing
  - event-wrapper helpers for run/explore/streaming entrypoints
- move what is reusable/runtime-owned into dedicated runtime/public-shell helpers
- keep `agent/loop.py` focused on:
  - public API entrypoints
  - the minimal state that truly belongs to the agent shell
  - explicit compatibility or UI-facing integration points

The goal is not “delete `Agent`.” The goal is for `agent/loop.py` to read like an intentionally tiny facade rather than “the last place leftover things live.”

### 2. Tighten the remaining completion/continuation contract

This is the strongest remaining audit line after Sprint 17.

Implementation targets:

- inventory the remaining heuristic behavior across:
  - `src/loader/runtime/completion_policy.py`
  - `src/loader/runtime/turn_completion.py`
  - `src/loader/runtime/assistant_turns.py`
  - any nearby turn-loop controllers that still re-enter based on soft textual cues
- identify which of those behaviors still look like:
  - speculative completion rescue
  - soft continuation nudges that should instead become explicit failure or explicit follow-through state
  - hidden policy that operators cannot inspect afterward
- prefer:
  - deletion
  - stricter typed completion decisions
  - explicit stop/continue reason codes
  - surfaced runtime evidence
  over adding more wrapper logic

The goal is to continue the Sprint 13 and Sprint 17 line: the runtime should either proceed for a clear reason or stop honestly, not softly coax the model onward without making that policy visible.

### 3. Expose completion-policy and stop/continue decisions as inspectable runtime state

Loader’s policy decisions should become easier to inspect, not just easier to reason about in code.

Implementation targets:

- define a small typed trace or summary for completion-policy decisions, likely covering:
  - why a text response was accepted
  - why a continuation was requested
  - why a turn was stopped or finalized
  - whether the decision came from text-loop detection, DoD gating, raw-tool recovery exhaustion, or completion-policy evaluation
- persist enough of that state in the current session/turn summary to support operator inspection
- surface it through existing product seams, likely:
  - `loader status`
  - `loader session show`
  - possibly `loader workflow show` if that is the cleanest fit

The goal is to make completion policy inspectable the same way permissions, prompts, workflow, and explore continuity are now inspectable.

### 4. Keep the ref relationship explicit and healthy

By Sprint 18, Loader should be clearly past the “copy the refs feature-for-feature” stage without losing the discipline the refs gave us.

Implementation targets:

- use claw-code for:
  - runtime seam quality
  - explicit policy/state ownership
  - prompt/runtime separation
- use OMX for:
  - verifier/workflow pressure and follow-through ideas
- do not add work just because the refs have it
- do add work when a ref reveals a real Loader weakness in:
  - honesty
  - inspectability
  - runtime ownership
  - follow-through

The goal is to make Sprint 18 explicitly “reference-guided Loader optimization,” not “shadow-porting another codebase.”

## Testing strategy

- unit coverage for:
  - new public-shell helpers or reduced `Agent` shell behavior
  - tightened completion-policy decisions
  - persisted completion/stop decision summaries
- runtime coverage for:
  - no regression in normal follow-through
  - honest stop behavior where rescue logic was deleted or tightened
  - session/status inspection of completion-policy decisions
- regression coverage for:
  - no reintroduction of soft rescue phrasing after stop/failure conditions
  - no drift back toward `agent/loop.py` owning runtime plumbing
  - no loss of current explore/session/workflow inspection behavior while shell state shifts again

## Definition of done

- `agent/loop.py` shrinks again or becomes materially more facade-like even if line count does not fall dramatically
- Loader deletes or hardens more remaining completion/continuation heuristics instead of preserving them through softer wrappers
- operators can inspect more of the runtime’s completion-policy reasoning after the fact
- Sprint 17’s explicit-bootstrap and explore-operator gains remain green
- the parity baseline remains green after the Sprint 18 shell/policy tightening

## Explicitly out of scope

- full claw-code policy-engine parity
- multi-agent or team orchestration
- AST-aware semantic diffs
- a broad explore workflow UI
- broad permission-rule editing UX

## Audit

### Status

- Sprint 18 is complete, and the audit is green. Loader now persists explicit completion-policy decisions and step traces, exposes them through the existing operator surfaces, and moves more shell glue out of `src/loader/agent/loop.py` into runtime-owned public-shell helpers.

### Landed

- completion-policy outcomes are now explicit runtime contract instead of hidden side effects: `src/loader/runtime/completion_policy.py`, `src/loader/runtime/turn_completion.py`, and `src/loader/runtime/finalization.py` now preserve reason-coded stop/continue outcomes such as `premature_completion_nudge`, `non_mutating_response_accepted`, `verification_passed`, and `verification_failed_reentry`
- session/runtime state now persists both the latest completion decision and a bounded per-turn completion trace through `src/loader/runtime/session.py`, `src/loader/runtime/events.py`, and the new `src/loader/runtime/completion_trace.py`
- completion-policy state is now inspectable from the product surface: `src/loader/runtime/inspection.py` and `src/loader/cli/main.py` show the latest completion decision in `loader status`, `loader session list`, and `loader session show`, while `loader session show` also prints the richer step-by-step completion trace
- resumed session state now restores completion decisions and traces through `src/loader/runtime/public_shell.py`, so inspection survives restart instead of only describing the active process
- the public shell is thinner again: `src/loader/runtime/public_shell.py` now owns a real `SteeringMailbox`, fresh/load session-install helpers, sync/async event-emitter normalization, and capability-refresh decision helpers
- `src/loader/agent/loop.py` now adopts those runtime-owned helpers instead of open-coding them; the file is down to 432 lines, steering no longer leaks across resume/clear lifecycle, and replacing a session now invalidates cached prompt state explicitly

### Verification

- `uv run pytest -q` is green: `342 passed`
- `tests/test_completion_policy.py`, `tests/test_turn_completion.py`, `tests/test_finalization.py`, `tests/test_session_state.py`, and `tests/test_inspection.py` now pin persisted completion decisions plus step-trace behavior directly
- `tests/test_runtime_public_shell.py` now covers steering mailbox behavior, fresh/load session-install helpers, sync/async event emitters, and capability-refresh helper behavior directly
- `tests/test_turn_preparation.py` now proves a new turn clears stale completion trace state before execution starts
- `tests/test_runtime_context.py`, `tests/test_runtime_bootstrap.py`, and `tests/test_runtime_launcher.py` stayed green while `Agent` adopted the slimmer public-shell helper set

### Residual debt

- `src/loader/agent/loop.py` is materially more facade-like than Sprint 17, but it still owns the public entrypoints themselves plus the remaining launcher/UI glue; Sprint 18 shrank the shell again, it did not eliminate it
- completion policy is now explicit and inspectable, but Loader still keeps bounded continuation heuristics for some non-mutating tasks; those nudges are no longer hidden, but they are still policy choices rather than hard-stop failures
- the new completion trace is intentionally compact and per-turn; Loader still does not have a richer long-horizon policy/debug timeline that merges completion, workflow, and repair evidence into one operator view
- Sprint 18 kept following claw-code and OMX as architectural guardrails, but Loader still stops short of claw-code's tighter policy-engine seams and OMX's deeper verifier/interview rigor
