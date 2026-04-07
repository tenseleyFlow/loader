# Sprint 09: Interactive Validation, Baselines, and Guardrails

## Prerequisites

Sprint 08

## Goals

Turn the audit from a strong paper read into an executable cleanup baseline.

Before deleting recovery layers, Loader needs two things the repo does not have yet:

- direct evidence from real interactive runs across at least one native-tool path and one raw-text/smaller-model path
- guardrail coverage for the specific parser/runtime blind spots the audit surfaced

This sprint is intentionally narrow. It should produce evidence, close the known stale allowlist bug, and leave the larger contract surgery to later sprints.

## Deliverables

### 1. Interactive validation matrix

Run a fixed task matrix against at least:

- one native-tool-capable backend/profile
- one smaller or raw-text-prone backend/profile

Capture for each run:

- task prompt
- active capability profile
- whether native tools or raw-text fallback fired
- phase trace
- final response quality
- verification outcome
- which recovery layers fired and whether they helped or harmed

Persist the report under `.docs/audit_sprints/` so later deletion sprints can cite real evidence instead of memory.

### 2. Parser blind-spot guardrails

Close the audit's most concrete regression before deeper parser work:

- add deterministic coverage for raw-text recovery of `TodoWrite`
- add deterministic coverage for raw-text recovery of `patch`
- add deterministic coverage for at least one newer workflow/operator tool such as `AskUserQuestion`
- stop hardcoding the six-tool allowlist in `agent/loop.py`

This sprint does not need the final parser architecture yet. It does need to stop widening the gap every time the registry grows.

### 3. Contract baseline artifacts

Record the current cleanup baseline in a committed artifact:

- `self.agent.` reach-in counts across `src/loader/runtime/`
- line counts for the major legacy files under `src/loader/agent/`
- inventory of every remaining recovery behavior, including:
  - owner
  - trigger
  - dependency
  - current test coverage
  - proposed disposition: `delete`, `gate`, or `keep`

This inventory becomes the checklist for Sprint 11 and Sprint 13.

### 4. Deletion criteria

Write down the rules for what survives:

- default to delete unless the interactive matrix shows the recovery path is materially load-bearing
- if a behavior stays, tie it to an explicit capability-profile condition or session-level guardrail
- forbid new fake-assistant continuation text unless a future sprint explicitly re-approves it with evidence

## Commit slicing

- one commit for the interactive validation artifact and baseline metrics
- one commit for each new raw-parser regression test cluster
- one commit for the allowlist fix or registry-derived fallback guard
- one commit for the recovery inventory / disposition table

## Testing strategy

- `uv run pytest -q`
- targeted runtime/parity coverage for raw-text recovery of newer tools
- at least one committed interactive validation report from a native-tool profile
- at least one committed interactive validation report from a smaller/raw-text-prone profile

## Definition of done

- Loader has committed interactive evidence for the contract discussion instead of only paper analysis
- the stale six-tool raw-extraction regression is closed and covered
- every remaining recovery layer has an owner, a disposition, and a later sprint target
- the repo has a stable baseline for reach-ins, line counts, and runtime heuristics before larger deletion work starts

## Explicitly out of scope

- introducing `RuntimeContext`
- deleting multiple recovery layers at once
- redesigning clarify/plan workflows
- refactoring `agent/safeguards.py` in bulk
