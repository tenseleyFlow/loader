# Loader Audit Cleanup Sprint Index

These sprints translate the 2026-04-07 audit in `.docs/audit.txt` into a post-Sprint-08 cleanup plan that is explicitly about deleting contract debt, not wrapping it in more helpers.

The repo has moved since the audit snapshot. On this planning branch:

- `uv run pytest -q` is green with `211 passed`
- Sprint 08's prompt builder, turn-phase tracking, and permission inspection surfaces are already present on `HEAD`
- Sprint 09 interactive validation has started; `loader doctor` now distinguishes metadata reachability from live chat readiness, and both native-capable and `json_tag` Ollama lanes currently fail the live chat probe on `/api/chat` with HTTP 500
- Sprint 10's runtime-ownership inversion is now materially in place: `src/loader/runtime/` no longer reaches into `Agent` directly, and the remaining legacy dependencies are explicit `RuntimeLegacyServices` seams
- Sprint 11 has already deleted several puppet behaviors and collapsed the raw-text fallback stack onto the shared parser used by the runtime and Ollama text fallback paths
- the central debt still remains:
  - the runtime still carries some recovery and safety heuristics around the main turn contract, even though the inline completion/critique rescue layers have now been deleted
  - workflow modes are now honestly scoped as lightweight single-question and single-pass flows, but the refs' deeper protocol and routing discipline are still absent
  - `agent/loop.py`, `agent/reasoning.py`, `agent/safeguards.py`, and `agent/recovery.py` are still the load-bearing legacy tree

## Sprint 09 Ownership Baseline

- `src/loader/runtime/conversation.py`: 881 lines, `49` `self.agent.` reach-ins
- `src/loader/runtime/assistant_turns.py`: `23` `self.agent.` reach-ins
- `src/loader/runtime/tool_batches.py`: `26` `self.agent.` reach-ins
- `src/loader/runtime/completion_policy.py`: `9` `self.agent.` reach-ins
- `src/loader/runtime/repair.py`: `4` `self.agent.` reach-ins
- `src/loader/runtime/explore.py`: `12` `self.agent.` reach-ins
- `src/loader/agent/loop.py`: 1108 lines
- `src/loader/agent/reasoning.py`: 1235 lines
- `src/loader/agent/safeguards.py`: 1142 lines
- `src/loader/agent/recovery.py`: 648 lines

## Current runtime ownership status

- `src/loader/runtime/` direct `self.agent.` reach-ins: `0`
- runtime ownership now flows through `RuntimeContext` plus explicit `RuntimeLegacyServices` adapters
- the next contract work is deletion, not more ownership reshuffling

## Current Sprint 11 status

- `src/loader/agent/loop.py`: `815` lines, down `296` lines from the Sprint 09 baseline of `1111`
- raw-text parsing now runs through `src/loader/agent/parsing.py`; the stale `_extract_raw_json_tool_calls(...)` fallback is gone from `src/loader/agent/loop.py`
- deleted assistant-puppeting behaviors so far:
  - post-action follow-up suffix
  - first-turn `[` prefill trick
  - fake-tool narration scolding
  - deflection repair prompts
  - self-critique reroute
  - non-mutating completion nudge
  - text-loop bailout
  - action-loop bailout on successful repeated tool patterns
- empty-output handling is now one honest retry followed by explicit failure instead of five fake assistant continuation prompts
- Sprint 11's hard subtraction target is now met, and the remaining debt is no longer parser fragmentation or inline bailout decisions but the broader legacy tree in `agent/reasoning.py`, `agent/safeguards.py`, and `agent/recovery.py`

## Phase 1: Validate Before Deleting

- [Sprint 09](sprint09.md) — Interactive Validation, Baselines, and Guardrails

## Phase 2: Tighten the Runtime Contract

- [Sprint 10](sprint10.md) — Runtime Context and Ownership Inversion
- [Sprint 11](sprint11.md) — Recovery Deletion and Tool Parsing Unification

## Phase 3: Finish the Behavioral Cleanup

- [Sprint 12](sprint12.md) — Workflow Protocol Hardening and Decomposition Decision
- [Sprint 13](sprint13.md) — Legacy Runtime Retirement and Safeguard Refactor

## Working principles

- Prefer deletion over relocation. A sprint that only moves code around is not done.
- Each discrete fix gets its own commit. Do not batch unrelated parser, workflow, safety, and runtime-contract changes into one "cleanup" commit.
- Pair each behavior change with deterministic coverage or a committed interactive validation artifact.
- Keep `.docs/PARITY.md` and any residual-debt notes honest as claims change.
- Stop at the sprint boundary. If a sprint uncovers a larger follow-on job, create the next sprint or artifact instead of expanding the current one mid-flight.
- Treat interactive testing as first-class evidence. The parity harness is necessary, but it is not enough for deciding which recovery layers are truly load-bearing.
