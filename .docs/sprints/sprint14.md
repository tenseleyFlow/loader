# Sprint 14: Runtime Context Adoption, Legacy Burn-Down, and Policy Narrowing

## Prerequisites

Sprint 13

## Goals

Turn the newly merged audit-line cleanup into a first-class runtime contract by promoting `RuntimeContext` from a compatibility seam into the primary execution boundary, burning down more agent-owned policy/services, and narrowing the still-heavy response/tool policy helpers.

Sprint 13 closed a real loop. Loader now has a semantic workflow ledger, prompt/artifact diff surfaces, a more honest no-tool completion path, and a dedicated response-routing seam. The audit branch is now merged too, which changes the next leverage point in an important way:

- Loader now has runtime-owned context, parsing, recovery, rollback, safeguard, and task-classification modules on `trunk`
- `src/loader/runtime/` no longer imports `agent/*` directly, and the merged tree is green at `286 passed`
- but several of those modules are still only partially adopted, with compatibility layers and legacy agent hooks still carrying too much behavior
- `response_routing.py` and `tool_batches.py` are better than the old inline loop, but they still carry more heuristic policy than the narrower seams in claw-code
- `agent/loop.py` is smaller in responsibility than it used to be, but it is still doing too much as a holder for runtime-owned decisions, helper methods, and policy services
- `agent/reasoning.py` and `agent/safeguards.py` are no longer hidden runtime implementations, but they still own meaningful legacy behavior behind explicit seams

The next leverage point is to stop treating the merged audit runtime pieces as optional support modules and start treating them as the default execution contract:

- `RuntimeContext` should become the normal runtime boundary, not only a bridge for selected helpers and tests
- runtime-owned parsing/recovery/rollback/safeguard/task-classification services should replace more legacy agent ownership
- response/tool policy should narrow further now that the runtime service surface is richer

This sprint is about consolidating the merge into a cleaner architecture rather than leaving the audit branch as a large historical merge with only partial adoption.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/PARITY.md`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/audit_sprints/sprint13_closure.md`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`

## Deliverables

### 1. Promote `RuntimeContext` from bridge to primary runtime contract

The merged audit branch introduced a typed runtime context. Sprint 14 should make that seam the default runtime boundary for more helpers.

Implementation targets:

- adopt `src/loader/runtime/context.py` across more runtime controllers and services so they consume typed runtime state instead of direct `Agent` access where practical
- reduce direct runtime dependencies on:
  - `Agent._extract_raw_json_tool_calls(...)`
  - `Agent._assess_confidence(...)`
  - `Agent._verify_action(...)`
  - ad hoc steering/recovery workflow hooks
- make the remaining `legacy` services in `RuntimeContext` explicit migration seams instead of long-term hidden dependencies
- shrink the number of callbacks exposed through `RuntimeLegacyServices`, or narrow them so their contracts are obviously transitional
- keep the contract pragmatic:
  - do not rewrite the whole runtime in one sweep
  - prefer controller/service boundaries that improve testability immediately

The goal is not purity for its own sake. The goal is to make runtime behavior easier to reason about and less likely to regress when we continue narrowing policy seams.

### 2. Burn down more agent-owned runtime services

The merged branch brought runtime-owned modules onto `trunk`. Sprint 14 should make more of them actually own behavior.

Implementation targets:

- move more effective ownership for the following out of `agent/loop.py` and into runtime modules:
  - task classification
  - raw-text parsing and tool-call recovery
  - rollback planning helpers
  - safeguard services
  - recovery prompts / retry guidance
- make the remaining `agent/reasoning.py` and `agent/safeguards.py` callbacks inventoryable and explicit, so we can say which ones are still required and which are just legacy inertia
- keep `agent/loop.py` focused on:
  - public entrypoints
  - user/session-facing orchestration
  - compatibility wrappers that truly still need to exist
- avoid duplicating logic across `agent/*` and `runtime/*`; prefer one real implementation plus compatibility exports only where needed

This is how the merged audit work becomes structural improvement instead of passive file accumulation.

### 3. Narrow `response_routing.py` and `tool_batches.py`

Sprint 13 moved response dispatch out of `turn_iteration.py`. Sprint 14 should keep pushing the same discipline into the next heavy seams.

Implementation targets:

- split `src/loader/runtime/response_routing.py` into narrower policy helpers where it pays off, likely around:
  - final-answer routing
  - raw-text tool routing
  - no-tool completion routing
  - halt/finalize decision shaping
- split `src/loader/runtime/tool_batches.py` more deliberately around:
  - confidence gate
  - recovery handling
  - DoD post-tool bookkeeping
  - post-tool verification
- keep behavior steady while making route ownership and failure handling easier to test directly

The goal is to keep moving away from broad “policy soup” modules and toward claw-code-style narrower execution seams.

### 4. Consolidate merged cleanup behavior into the test contract

The merge brought in new tests and new runtime compatibility seams. Sprint 14 should turn that into a clearer, intentional contract.

Implementation targets:

- keep the `RuntimeContext` tests green while reducing how much behavior still depends on compatibility shims
- add direct tests for any newly split response/tool policy controllers
- keep honest-repair, no synthetic prefill, and no-tool completion cleanup behavior covered after the deeper service migration
- extend parity and inspection coverage where the merged audit docs surfaced real operator-facing expectations
- treat the merged `.docs/audit_sprints/` artifacts as regression evidence when deciding whether a seam is actually load-bearing

This makes the merged audit branch a maintained baseline rather than a one-time reconciliation event.

### 5. Reconcile docs after the merged audit line

The branch is merged. The docs should stop behaving like the audit line is still “over there.”

Implementation targets:

- refresh `REPORT.md`, `PARITY.md`, and the sprint audit trail where the merge changed the architectural baseline in a meaningful way
- preserve `.docs/audit_sprints/` as historical evidence, not as a second active roadmap
- keep the parity checkpoint honest about which runtime-context/service seams are truly primary vs compatibility-bound

## Testing strategy

- unit coverage for:
  - `RuntimeContext`-owned service behavior
  - response-routing subcontrollers
  - tool-batch subcontrollers
  - raw-text fallback and capability refresh paths after the deeper context adoption
- runtime coverage for:
  - native-tool and raw-text tool parity
  - explore mode through the typed runtime context
  - verification/recovery after service migration
  - Sprint 00-13 parity scenarios staying green after the context/service ownership shift
- regression coverage:
  - no synthetic prefill
  - no repeated empty-response puppeting
  - no self-critique reroute regression
  - no accidental rollback/recovery ownership drift between `agent/*` and `runtime/*`

## Definition of done

- `RuntimeContext` is a real primary runtime seam for more controllers/services, not just a compatibility adapter
- more runtime-owned parsing/recovery/rollback/safeguard/task-classification behavior is actually moved off `agent/loop.py`
- `response_routing.py` and `tool_batches.py` are narrower and more directly tested
- the merged audit line is reflected as one baseline, not two parallel architectures
- the remaining legacy callbacks are enumerated, narrower, and clearly transitional
- the full parity baseline remains green after the deeper runtime-context adoption

## Explicitly out of scope

- full claw-code policy-engine parity
- AST-aware or LSP-aware semantic artifact diffs
- visual workflow or timeline UIs
- multi-agent or team orchestration

## Audit

### Status

- Sprint 14 is complete, and the audit is green. `RuntimeContext` is now the normal runtime seam for the main turn path rather than a selective bridge layered over legacy callbacks.

### Landed

- `src/loader/runtime/context.py` is now a real primary contract instead of a partial adapter: the old `RuntimeLegacyServices` shim is gone, workflow-mode mutation lives on the typed context, and the runtime can refresh capability state, steering state, and workflow state without routing back through a legacy wrapper
- runtime-owned service adoption is materially deeper than the merged-audit baseline: `src/loader/runtime/workflow_state.py`, `src/loader/runtime/phases.py`, `src/loader/runtime/repair.py`, `src/loader/runtime/completion_policy.py`, `src/loader/runtime/turn_completion.py`, `src/loader/runtime/response_route_handlers.py`, `src/loader/runtime/response_routing.py`, `src/loader/runtime/turn_loop.py`, `src/loader/runtime/turn_iteration.py`, `src/loader/runtime/finalization.py`, `src/loader/runtime/workflow_lanes.py`, and `src/loader/runtime/workflow_recovery.py` now consume typed runtime state instead of reaching into `Agent` for session, backend, registry, or workflow state
- raw-text tool recovery no longer depends on a hidden `Agent._extract_raw_json_tool_calls(...)` escape hatch: `src/loader/runtime/repair.py` now routes fallback parsing through the runtime parser plus the active registry, which closes an old audit concern around newer tools such as `TodoWrite`
- the response/tool path is narrower and more directly testable: the earlier extraction of `src/loader/runtime/tool_batch_checks.py`, `src/loader/runtime/tool_batch_recovery.py`, `src/loader/runtime/response_route_handlers.py`, and `src/loader/runtime/response_route_types.py` is now paired with typed-context adoption across the hot path, so response routing, tool-batch gating, no-tool completion, and finalization no longer behave like disguised `agent/loop.py` helpers
- the merged audit line is now reflected as one architectural baseline instead of a second hidden runtime: the active `trunk` runtime owns reasoning callbacks, raw-text recovery, response policy, workflow state, turn state, and finalization directly, and the remaining `agent/*` ownership is much smaller and easier to inventory
- the test contract around this migration is stronger and more intentional: `tests/test_runtime_context.py`, `tests/test_runtime_state_controllers.py`, `tests/test_completion_policy.py`, `tests/test_repair.py`, `tests/test_response_route_handlers.py`, `tests/test_turn_loop.py`, `tests/test_turn_iteration.py`, and `tests/test_explore_runtime.py` now pin the typed-context contract directly instead of relying only on large integration tests

### Verification

- `uv run pytest -q` is green: `303 passed`
- `tests/test_runtime_context.py` and `tests/test_runtime_state_controllers.py` cover typed context construction plus direct workflow-state and phase-tracker behavior without an `Agent` object on the other side
- `tests/test_repair.py` covers raw-text fallback through the runtime parser/registry, including modern workflow-tool recovery such as `TodoWrite`
- `tests/test_completion_policy.py`, `tests/test_turn_completion.py`, `tests/test_response_route_handlers.py`, `tests/test_response_routing.py`, `tests/test_turn_iteration.py`, and `tests/test_turn_loop.py` cover the main response-policy and assistant-cycle path after the deeper context adoption
- `tests/test_explore_runtime.py` still proves explore refreshes capabilities before the first request, so the typed runtime context is also now load-bearing outside the main task loop
- the larger workflow/runtime suites remained green after the migration, so Sprint 14 did not trade architectural cleanup for parity regressions

### Residual debt

- `src/loader/runtime/conversation.py` and `src/loader/runtime/explore.py` still bootstrap from `agent._build_runtime_context()`, and `conversation.py` still performs a post-prepare sync of capability/prompt state from the agent wrapper; the remaining runtime/agent coupling is now mostly bootstrap ownership rather than policy ownership
- `src/loader/agent/loop.py` still owns meaningful planning/prompt/session orchestration outside the hot runtime path, so Loader is much cleaner than the merged-audit baseline but has not fully collapsed to a minimal public entrypoint shell yet
- `src/loader/agent/reasoning.py` and `src/loader/agent/safeguards.py` are now behind typed runtime protocols, but they still own meaningful behavior and remain future burn-down candidates if Sprint 15 wants to keep reducing agent-owned runtime services
- `src/loader/runtime/tool_batches.py` and parts of `src/loader/runtime/workflow_lanes.py` are narrower than before, but they still carry more heuristic policy than the tighter claw-code reference seams
- the workflow policy is stronger and the runtime contract is cleaner, but Loader still stops short of claw-code's fuller policy engine, OMX's deeper planning/interview rigor, and a richer operator UX for editing or simulating policy/rule state
