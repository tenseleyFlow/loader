# Sprint 11: Recovery Deletion and Tool Parsing Unification

## Status on `cleanup-audit-plan`

- repo verification is currently `212 passed`
- `src/loader/agent/loop.py` is down to `926` lines from the Sprint 09 baseline of `1111`
- the sprint has already deleted:
  - the post-action follow-up suffix
  - the first-turn `[` prefill trick
  - fake-tool narration repair prompts
  - deflection repair prompts
- empty-response handling has been tightened to one honest retry plus explicit failure
- raw-text parsing has been unified onto `src/loader/agent/parsing.py`
  - `src/loader/agent/loop.py` no longer carries `_extract_raw_json_tool_calls(...)`
  - `src/loader/runtime/repair.py` and `src/loader/runtime/explore.py` use the shared parser
  - `src/loader/llm/ollama.py` now routes both complete-mode and streaming final text parsing through the shared parser
- the sprint is not complete yet
  - the hard subtraction target is still missed by `115` lines
  - self-critique rerouting, text-loop bailout, non-mutating completion nudges, and action-loop bailout still need an explicit keep/delete decision

## Prerequisites

Sprint 10

## Goals

Delete or tightly gate the recovery layers that still make Loader puppet the assistant in-stream.

This is the central contract sprint. After Sprint 10 creates a clean runtime boundary, this sprint should remove the behaviors the audit called out instead of simply naming them more cleanly.

## Deliverables

### 1. Unify raw-text tool parsing

Resolve the duplicated parsing split between `agent/parsing.py` and `agent/loop.py`.

Current state:

- complete enough to count as landed for the core runtime path
- follow-on parser work should only target residual streaming UX shims or backend-specific cleanup, not reintroduce a second extraction path

Implementation targets:

- delete `_extract_raw_json_tool_calls(...)` from `agent/loop.py`, or reduce it to a thin compatibility shim over a shared parser
- make raw-text parsing aware of the real registry surface instead of a hardcoded tool list
- keep native-tool and raw-text paths converging on the same normalized `ToolCall` contract before execution
- gate raw-text fallback by capability profile rather than assuming every model should get it

The outcome should be one parsing strategy, not two diverging regex stacks.

### 2. Remove fake assistant continuation behavior

Delete the assistant-puppeteering paths unless Sprint 09 interactive evidence proves one must survive behind an explicit gate.

Current state:

- largely in progress with real deletions already landed
- the biggest surviving behavior in this area is the bounded empty-response retry, which is now explicit and much narrower than the original puppet prompts

Primary deletion targets:

- the `[` prefill trick in `runtime/conversation.py`
- the five hardcoded empty-output continuation prompts
- fake-tool narration scolding that fabricates assistant/user turns to steer the model back on track
- the unconditional "Would you like me to make any changes or additions?" suffix

Replace these with a simpler contract:

- bounded retries where truly necessary
- honest failure/escalation when the assistant does not act
- DoD/verification evidence for mutating tasks
- user-visible stop conditions instead of hidden assistant puppeteering

### 3. Re-scope critique, loop, and completion nudges

For each remaining heuristic, decide whether it should be:

- deleted
- moved to a session-level safeguard
- gated behind a capability/profile condition

This includes:

- self-critique rerouting
- text-loop bailout
- non-mutating completion nudges
- deflection handling

No heuristic survives this sprint without a written reason tied back to Sprint 09 evidence.

### 4. Shrink the legacy loop by subtraction

This sprint should materially reduce legacy surface area instead of moving it again.

Set a hard subtraction target:

- `src/loader/agent/loop.py` must shrink by at least 300 lines from the Sprint 09 baseline, or the sprint is not complete

Current score:

- baseline: `1111`
- current: `926`
- net: `-185`
- remaining to target: `115`

If a target is missed, document exactly which remaining behaviors blocked deletion and move them into the next sprint explicitly instead of silently carrying them forward.

## Commit slicing

- one commit for raw-parser unification or shim removal
- one commit per deleted or newly gated recovery behavior
- one commit for any capability-profile gating additions
- one commit for the final legacy-loop cleanup after the behavior changes are already green

## Testing strategy

- `uv run pytest -q`
- targeted parser tests for raw-text recovery across legacy and newer tools
- deterministic runtime coverage for empty-response handling, fake narration, and completion behavior after deletion
- parity-harness confirmation that the retained contract still works end-to-end
- interactive reruns of the Sprint 09 matrix for every capability profile whose behavior changed

## Definition of done

- Loader no longer fabricates assistant turns to keep the model moving in the common case
- raw-text tool recovery uses one normalized parser path and no stale tool allowlist
- every surviving recovery heuristic has an explicit owner, gate, and reason
- `agent/loop.py` shrinks materially by subtraction, not merely by forwarding calls elsewhere

## Explicitly out of scope

- clarify/plan workflow redesign
- broad safety-hook refactors unrelated to deleted recovery behavior
- multi-agent or planner/critic expansion
