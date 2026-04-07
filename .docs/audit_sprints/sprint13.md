# Sprint 13: Legacy Runtime Retirement and Safeguard Refactor

## Prerequisites

Sprint 12

## Goals

Finish the cleanup cycle by shrinking the legacy `agent/` tree and moving the remaining load-bearing behavior into runtime-owned services, hooks, or clearly scoped helpers.

This sprint closes the residual debt left open by Sprint 03 and by the additive refactors that followed it.

## Deliverables

### 1. Refactor `agent/safeguards.py` by subtraction

Complete the refactor Sprint 03 claimed but never really achieved.

Implementation targets:

- move remaining hook-worthy validation and action-tracking logic into runtime-owned implementations or service modules
- delete wrapper-only layers where runtime hooks simply forward into `agent.safeguards`
- remove legacy filtering or duplicate-checking logic that is no longer needed after Sprint 11's contract tightening

The success condition is a smaller, less central safeguards file, not a prettier import graph around the same code.

### 2. Retire legacy reasoning/recovery helpers that no longer own behavior

After Sprint 11 and Sprint 12, some functions in `agent/reasoning.py` and `agent/recovery.py` should either move behind explicit service seams or disappear entirely.

Targets include helpers that only existed to support:

- premature-completion nudges that were deleted or gated
- raw-text parsing behavior that was unified elsewhere
- decomposition flows that were deleted or promoted
- old recovery prompts that the runtime no longer injects

### 3. Reduce runtime imports from `agent/*`

By the end of this sprint, runtime modules should depend on narrow typed services, not broad legacy modules.

Drive toward:

- no direct runtime import of `agent.safeguards` for primary hook behavior
- no direct runtime import of legacy parsing/recovery code where a runtime-owned equivalent now exists
- a smaller, clearer adapter boundary inside `Agent`

### 4. Legacy debt scoreboard and closure report

Commit a final audit-closure artifact under `.docs/audit_sprints/` that records:

- line-count deltas versus the Sprint 09 baseline
- remaining `agent/*` dependencies from `runtime/*`
- which audit findings are now closed, partially closed, or intentionally deferred

This is the final honesty pass for the cleanup cycle.

## Commit slicing

- one commit per safeguards/service migration
- one commit per deleted reasoning/recovery slice
- one commit for runtime import cleanup
- one commit for the closure report and PARITY/residual-debt updates

## Testing strategy

- `uv run pytest -q`
- safety and hook lifecycle regressions for the moved/deleted safeguards logic
- coverage for any remaining runtime-owned validation services
- smoke coverage showing the runtime still enforces safety, policy, and duplicate detection after the legacy shrink

## Definition of done

- `agent/safeguards.py` is materially smaller and no longer the hidden primary implementation behind runtime hooks
- legacy reasoning/recovery helpers only remain where they still own real behavior
- `runtime/*` depends on typed runtime services instead of broad `agent/*` modules wherever practical
- the repo has a committed closure report against the audit baseline

## Explicitly out of scope

- starting a new feature sprint before the closure report is written
- speculative architectural rewrites not tied to an audit finding
- expanding Loader beyond the current single-agent product scope
