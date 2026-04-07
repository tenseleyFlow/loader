# Sprint 12: Workflow Protocol Hardening and Decomposition Decision

## Status on `cleanup-audit-plan`

- repo verification is currently `211 passed`
- clarify mode is now explicitly a single-question brief flow in prompts, runtime behavior, and persisted clarify artifacts
- plan mode is now explicitly single-pass implementation and verification artifact generation in prompts, runtime behavior, and persisted plans
- execute now records workflow-artifact status and artifact sources in session state when it activates or reuses the workflow bridge
- the legacy decomposition CLI flag and `agent/loop.py` decomposition orchestration have been deleted
- the sprint's explicit workflow-contract goals are now met
  - the remaining gap is not hidden workflow depth; it is the absence of the refs' deeper routing discipline and the broader legacy tree still living under `agent/`

## Prerequisites

Sprint 11

## Goals

Either make Loader's workflow modes real protocols or narrow their claims until they are honest.

Sprint 04 landed artifacts and routing, but not the discipline the refs rely on. After the contract cleanup work, Loader should stop pretending that clarify/plan are deeper than they are.

## Deliverables

### 1. Clarify mode becomes a real loop or is explicitly downscoped

Choose one honest outcome and implement it fully:

- either add a real clarify protocol with one-question-per-round discipline, ambiguity scoring, exit criteria, and persisted open questions
- or explicitly downscope clarify mode to a lightweight single-question artifact flow in code, prompts, docs, and parity notes

The repo should not keep OMX-style claims if the runtime is not actually enforcing OMX-style behavior.

### 2. Plan mode becomes iterative or is explicitly redefined

Choose one honest outcome and implement it fully:

- either add a minimal real iteration loop such as planner then critic with persisted revisions
- or redefine plan mode as "single-pass planning artifact generation" everywhere and stop implying consensus/planning discipline that does not exist

The important part is that the runtime behavior, docs, and persisted artifacts all agree.

### 3. Execution honors workflow artifacts explicitly

Tighten the boundary between workflow modes and execution:

- execution should consume plan/clarify artifacts intentionally
- skips or overrides should be recorded explicitly in runtime/session state
- mode routing should be based on clearer gates than prompt text alone where practical

This does not need OMX's full architecture. It does need a firmer contract than "label plus prompt."

### 4. Make a final decomposition decision

The decomposition path in `agent/loop.py` is currently opt-in legacy code that is not integrated with the workflow system cleanly.

This sprint must either:

- promote decomposition into an explicit workflow/runtime feature with ownership, tests, and surfaced state
- or delete the legacy decomposition orchestration from `agent/loop.py`

Dead-ish code is not an acceptable steady state.

## Commit slicing

- one commit for clarify-mode contract changes
- one commit for plan-mode contract changes
- one commit for execution/artifact integration
- one commit for decomposition promotion or deletion
- one commit for doc/PARITY updates that make the resulting behavior explicit

## Testing strategy

- `uv run pytest -q`
- workflow-mode tests for clarify and plan round behavior
- artifact persistence tests for briefs and plans under the revised contract
- runtime tests showing execute mode either consumes or explicitly bypasses workflow artifacts
- coverage for whichever decomposition outcome is chosen

## Definition of done

- clarify and plan modes are described honestly and enforced consistently
- execution has a clearer contract with the artifacts those modes produce
- decomposition is either a first-class workflow/runtime feature or gone
- `.docs/PARITY.md` and sprint docs no longer overclaim workflow depth

## Explicitly out of scope

- OMX-style multi-role planning beyond the chosen minimum honest protocol
- multi-agent delegation
- broad prompt-builder redesign unrelated to workflow enforcement
