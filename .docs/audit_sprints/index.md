# Loader Audit Cleanup Sprint Index

These sprints translate the 2026-04-07 audit in `.docs/audit.txt` into a post-Sprint-08 cleanup plan that is explicitly about deleting contract debt, not wrapping it in more helpers.

The repo has moved since the audit snapshot. On this planning branch:

- `uv run pytest -q` is green with `185 passed`
- Sprint 08's prompt builder, turn-phase tracking, and permission inspection surfaces are already present on `HEAD`
- the central debt still remains:
  - `runtime/*` still reaches into `Agent` directly instead of working through a typed runtime context
  - the runtime still repairs model misbehavior in-stream with retries, prefills, nudges, and fake-assistant continuations
  - raw-text tool extraction is still duplicated and still hardcodes a stale six-tool allowlist in `agent/loop.py`
  - clarify/plan workflows still persist artifacts without enforcing the deeper protocol the refs rely on
  - `agent/loop.py`, `agent/reasoning.py`, `agent/safeguards.py`, and `agent/recovery.py` are still the load-bearing legacy tree

## Current baseline

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
