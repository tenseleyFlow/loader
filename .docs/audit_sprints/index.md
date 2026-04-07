# Loader Audit Cleanup Sprint Index

These sprints translate the 2026-04-07 audit in `.docs/audit.txt` into a post-Sprint-08 cleanup plan that is explicitly about deleting contract debt, not wrapping it in more helpers.

The repo has moved since the audit snapshot. On this planning branch:

- `uv run pytest -q` is green with `226 passed`
- Sprint 08's prompt builder, turn-phase tracking, and permission inspection surfaces are already present on `HEAD`
- Sprint 09 interactive validation has started; `loader doctor` now distinguishes metadata reachability from live chat readiness, and both native-capable and `json_tag` Ollama lanes currently fail the live chat probe on `/api/chat` with HTTP 500
- Sprint 10's runtime-ownership inversion is now materially in place: `src/loader/runtime/` no longer reaches into `Agent` directly, and the remaining legacy dependencies are explicit `RuntimeLegacyServices` seams
- Sprint 11 has already deleted several puppet behaviors and collapsed the raw-text fallback stack onto the shared parser used by the runtime and Ollama text fallback paths
- Sprint 13 has now finished the runtime-side retirement work:
  - `src/loader/runtime/` has `0` direct imports from `agent/*`
  - shared safeguard, rollback, recovery, parsing, reasoning-type, and task-classification surfaces now live under `src/loader/runtime/`
  - the remaining debt is narrower:
    - workflow modes are honestly scoped, but the refs' deeper protocol and routing discipline are still absent
    - `agent/reasoning.py` and `agent/safeguards.py` still hold some real legacy behavior behind explicit seams
    - interactive validation against a healthy live backend is still blocked on Ollama `/api/chat` failures

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
- `src/loader/runtime/` direct imports from `agent/*`: `0`
- the runtime-side ownership migration phase is complete; the remaining work is closure reporting and any future follow-on deletion beyond the runtime package

## Current legacy-tree status

- `src/loader/agent/loop.py`: `721` lines, down `390` lines from the Sprint 09 baseline of `1111`
- `src/loader/agent/reasoning.py`: `649` lines, down `586` lines from the Sprint 09 baseline of `1235`
- `src/loader/agent/safeguards.py`: `595` lines, down `547` lines from the Sprint 09 baseline of `1142`
- `src/loader/agent/recovery.py`: `12` lines, down `636` lines from the Sprint 09 baseline of `648`
- shared raw-text parsing now runs through `src/loader/runtime/parsing.py`; `src/loader/agent/parsing.py` is compatibility-only and the stale `_extract_raw_json_tool_calls(...)` fallback is gone from `src/loader/agent/loop.py`
- deleted assistant-puppeting behaviors:
  - post-action follow-up suffix
  - first-turn `[` prefill trick
  - fake-tool narration scolding
  - deflection repair prompts
  - self-critique reroute
  - non-mutating completion nudge
  - text-loop bailout
  - action-loop bailout on successful repeated tool patterns
- empty-output handling is now one honest retry followed by explicit failure instead of five fake assistant continuation prompts
- the remaining debt is no longer parser fragmentation or hidden runtime ownership; it is the narrower set of explicit legacy callbacks, streamed safeguards, workflow-depth gaps, and still-blocked live backend validation

## Phase 1: Validate Before Deleting

- [Sprint 09](sprint09.md) — Interactive Validation, Baselines, and Guardrails

## Phase 2: Tighten the Runtime Contract

- [Sprint 10](sprint10.md) — Runtime Context and Ownership Inversion
- [Sprint 11](sprint11.md) — Recovery Deletion and Tool Parsing Unification

## Phase 3: Finish the Behavioral Cleanup

- [Sprint 12](sprint12.md) — Workflow Protocol Hardening and Decomposition Decision
- [Sprint 13](sprint13.md) — Legacy Runtime Retirement and Safeguard Refactor
- [Sprint 13 Closure](sprint13_closure.md) — Final scoreboard, remaining debt, and audit-closure honesty pass

## Working principles

- Prefer deletion over relocation. A sprint that only moves code around is not done.
- Each discrete fix gets its own commit. Do not batch unrelated parser, workflow, safety, and runtime-contract changes into one "cleanup" commit.
- Pair each behavior change with deterministic coverage or a committed interactive validation artifact.
- Keep `.docs/PARITY.md` and any residual-debt notes honest as claims change.
- Stop at the sprint boundary. If a sprint uncovers a larger follow-on job, create the next sprint or artifact instead of expanding the current one mid-flight.
- Treat interactive testing as first-class evidence. The parity harness is necessary, but it is not enough for deciding which recovery layers are truly load-bearing.
