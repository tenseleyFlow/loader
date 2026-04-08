# Sprint 13 Closure Report

## Outcome

Sprint 13 met its runtime-retirement target.

- `src/loader/runtime/` now has `0` direct imports from `agent/*`
- the runtime still has `0` direct `self.agent.` reach-ins after Sprint 10
- repo-wide verification is green at `226 passed`
- the remaining debt is no longer hidden runtime ownership; it is the narrower set of legacy prompt/filter helpers and the still-blocked live backend validation matrix

## Commit trail

Sprint 13 landed as small, behavior-scoped commits:

- `34effb0` `Move safeguard services into runtime`
- `098f467` `Move rollback planning into runtime`
- `3ebef1c` `Move recovery services into runtime`
- `50ab16b` `Move parsing helpers into runtime`
- `e36e64f` `Move reasoning types into runtime`
- `bb48c80` `Move task classification into runtime`

## Legacy tree delta vs Sprint 09 baseline

Baseline source: [sprint09_baseline.md](sprint09_baseline.md)

| File | Sprint 09 baseline | Current | Delta |
| --- | ---: | ---: | ---: |
| `src/loader/agent/loop.py` | 1111 | 721 | -390 |
| `src/loader/agent/reasoning.py` | 1235 | 649 | -586 |
| `src/loader/agent/safeguards.py` | 1142 | 595 | -547 |
| `src/loader/agent/recovery.py` | 648 | 12 | -636 |
| total | 4136 | 1977 | -2159 |

Additional shared-parser shrink not tracked in the original baseline table:

| File | Earlier shared-parser size | Current | Delta |
| --- | ---: | ---: | ---: |
| `src/loader/agent/parsing.py` | 182 | 7 | -175 |

## Runtime ownership scoreboard

Runtime-owned modules added during Sprint 13:

- `src/loader/runtime/safeguard_services.py`
- `src/loader/runtime/rollback.py`
- `src/loader/runtime/recovery.py`
- `src/loader/runtime/parsing.py`
- `src/loader/runtime/reasoning_types.py`
- `src/loader/runtime/task_classification.py`

Current runtime dependency state:

- direct `runtime -> agent/*` imports: `0`
- direct `runtime -> agent.safeguards` imports for hook behavior: `0`
- direct `runtime -> agent.recovery` imports: `0`
- direct `runtime -> agent.parsing` imports: `0`
- direct `runtime -> agent.reasoning` imports: `0`

This closes the audit's runtime-ownership complaint at the import boundary. The remaining legacy behavior is now reached either through explicit `RuntimeLegacyServices` callbacks or outside the runtime package entirely.

## Audit finding status

| Audit theme | Status | Notes |
| --- | --- | --- |
| runtime ownership depended on `Agent` reach-ins | closed | Sprint 10 removed `self.agent.` reach-ins; Sprint 13 removed direct `runtime -> agent/*` imports |
| runtime hooks were wrappers over `agent.safeguards` | closed | duplicate detection, validation, rollback tracking, and recovery ownership now live under `runtime/*` |
| raw-text parsing was split and stale | closed | shared parser now lives in `runtime/parsing.py`; legacy agent wrapper is compatibility-only |
| legacy recovery helpers remained load-bearing | closed | `agent/recovery.py` is now a thin compatibility re-export |
| reasoning types were still runtime-owned by `agent.reasoning` | closed | runtime event/context typing now comes from `runtime/reasoning_types.py` |
| workflow modes overstated protocol depth | partial | Sprint 12 made the scope honest, but Loader still does not implement the refs' deeper clarify/plan discipline |
| interactive validation against real backends | deferred | `loader doctor` is now honest, but the documented Ollama `/api/chat` HTTP 500 failures still block the live validation matrix |

## Remaining residual debt

- `src/loader/agent/safeguards.py` still owns streamed-output filtering and pattern steering. That file is no longer the hidden implementation behind runtime hooks, but it is still a real legacy surface.
- `src/loader/agent/reasoning.py` still owns prompt text and parsing helpers for self-critique, confidence scoring, verification, and completion checks. Those behaviors are no longer runtime-owned, but they still sit in the legacy tree behind explicit callbacks.
- `src/loader/ui/app.py` and `src/loader/ui/adapter.py` still depend on `agent.loop.AgentEvent`. That is outside the runtime package, but it is still part of the broader legacy boundary.
- the Sprint 09 interactive validation matrix remains blocked by live backend chat failures and should be rerun once `/api/chat` is healthy.

## Verification snapshot

- `uv run pytest -q` -> `226 passed`
- targeted Sprint 13 migration checks stayed green after each service move

## Honest end state

Loader is in a meaningfully different state than the audit described:

- the runtime contract is no longer hidden behind imports from `agent/safeguards.py`, `agent/recovery.py`, `agent/parsing.py`, or `agent/reasoning.py`
- the legacy tree is materially smaller by more than two thousand lines versus the Sprint 09 baseline
- the remaining debt is now narrow enough to discuss directly instead of being scattered through implicit runtime ownership

What Sprint 13 did **not** prove is live model behavior against a healthy real backend. That remains the next evidence gap, and the closure should stay honest about it.
