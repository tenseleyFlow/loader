# Trunk Divergence Sitrep

Date: 2026-04-07

## Snapshot

- cleanup branch: `cleanup-audit-plan` at `97e5aa9`
- local trunk: `trunk` at `4effa19`
- merge-base: `319013422032eb0436cc10d214d08bdd071f8743`
- branch divergence since merge-base: `59` commits on `trunk`, `59` commits on `cleanup-audit-plan`
- local trunk status at inspection time:
  - ahead of `origin/trunk` by `20` commits
  - one untracked local file: `.docs/audit.txt`

This is no longer a small rebase. The two lines of work are now materially different evolutions of the same post-Sprint-08 code.

## What trunk has done since the split

Trunk has pushed deeper on workflow protocol, clarify rigor, and conversation-runtime decomposition.

Major trunk themes:

- prompt builder and phase surfaces landed on trunk and were then built on further
- clarify mode gained grounding, slot-awareness, pressure passes, and richer brief synthesis
- workflow policy/timeline/state/recovery machinery expanded significantly
- `runtime/conversation.py` was split into explicit turn-control modules:
  - `turn_preparation.py`
  - `turn_preamble.py`
  - `turn_iteration.py`
  - `turn_completion.py`
  - `turn_loop.py`
  - `workflow_state.py`
  - `workflow_policy.py`
  - `workflow_lanes.py`
  - `workflow_recovery.py`
- CLI and inspection surfaces expanded around workflow and permission visibility
- trunk added a large amount of direct targeted coverage for those new seams

Representative trunk-only commits:

- `455a0f5` `Extract main turn loop control from conversation runtime`
- `8c70869` `Extract workflow state control from conversation runtime`
- `753d5b5` `Extract turn preparation from conversation runtime`
- `7ca9a28` `Add semantic artifact invalidation and replan recovery`
- `60d5983` `Add intent-aware clarify slot strategy`
- `5a65371` `Extract repo facts for clarify grounding`
- `5fda6ed` `Audit Sprint 12 interview rigor rollout`
- `4effa19` `Plan Sprint 13 semantic diff work`

## What cleanup has done since the split

The cleanup branch has pushed harder on runtime contract tightening, heuristic deletion, and legacy runtime retirement.

Major cleanup themes:

- Sprint 09 baseline, interactive-validation artifacts, and raw-text fallback guardrails
- Sprint 10 runtime ownership inversion through `RuntimeContext`
- Sprint 11 deletion of puppet behaviors and parser unification
- Sprint 12 honest downscoping of clarify/plan plus legacy decomposition deletion
- Sprint 13 runtime-owned service extraction and closure reporting

Representative cleanup-only commits:

- `f6cc62e` `Unify raw-text parsing on shared parser`
- `7c1d6d8` `Tighten empty-response retry contract`
- `ce20b55` `Delete post-action follow-up suffix`
- `bfeecb2` `Delete first-turn prefill trick`
- `34effb0` `Move safeguard services into runtime`
- `3ebef1c` `Move recovery services into runtime`
- `e36e64f` `Move reasoning types into runtime`
- `bb48c80` `Move task classification into runtime`
- `97e5aa9` `Record sprint 13 closure report`

## Current relationship between the branches

The two branches are not duplicative. They are mostly complementary, but they collide in exactly the files that now matter most.

Trunk optimized for:

- richer clarify/workflow behavior
- explicit controller/state-machine decomposition
- better operator-facing workflow evidence

Cleanup optimized for:

- stronger runtime ownership boundaries
- deletion of assistant puppeting behavior
- shrinking the hidden legacy tree
- documenting closure against the audit

That means the branches agree on direction, but they disagree on shape.

## Main overlap and merge-pressure zones

Highest-risk overlap:

- `src/loader/runtime/conversation.py`
- `src/loader/runtime/workflow.py`
- `src/loader/runtime/inspection.py`
- `src/loader/cli/main.py`
- `src/loader/runtime/session.py`
- `src/loader/runtime/repair.py`
- `src/loader/runtime/completion_policy.py`
- `src/loader/runtime/phases.py`
- `src/loader/llm/ollama.py`
- `tests/test_runtime_harness.py`
- `tests/test_workflow_runtime.py`

Trunk-only structural modules that need to be preserved:

- `src/loader/runtime/clarify_grounding.py`
- `src/loader/runtime/clarify_strategy.py`
- `src/loader/runtime/artifact_invalidation.py`
- `src/loader/runtime/turn_preparation.py`
- `src/loader/runtime/turn_preamble.py`
- `src/loader/runtime/turn_iteration.py`
- `src/loader/runtime/turn_completion.py`
- `src/loader/runtime/turn_loop.py`
- `src/loader/runtime/workflow_state.py`
- `src/loader/runtime/workflow_policy.py`
- `src/loader/runtime/workflow_lanes.py`
- `src/loader/runtime/workflow_recovery.py`
- `src/loader/runtime/workflow_signals.py`

Cleanup-only runtime modules that are likely worth porting onto trunk:

- `src/loader/runtime/context.py`
- `src/loader/runtime/safeguard_services.py`
- `src/loader/runtime/rollback.py`
- `src/loader/runtime/recovery.py`
- `src/loader/runtime/parsing.py`
- `src/loader/runtime/reasoning_types.py`
- `src/loader/runtime/task_classification.py`

## Most important sitrep conclusion

Trunk has likely become the better landing base for future work.

Why:

- trunk already contains the newer conversation/workflow decomposition
- trunk is the branch pushing product behavior and operator-visible semantics forward
- cleanup's strongest value now is not its old file layout, but the contract-tightening outcomes it achieved

In other words: the cleanup branch should probably be treated as a source of transplantable outcomes, not as the branch to merge wholesale into trunk without a deliberate integration pass.

## Recommended integration approach

Do **not** do a blind merge and hope Git sorts it out.

Recommended path:

1. Create a fresh integration worktree from current `trunk`.
2. Replay cleanup outcomes onto that base as new small commits.
3. Start with the lowest-conflict service moves:
   - `runtime/safeguard_services.py`
   - `runtime/rollback.py`
   - `runtime/recovery.py`
   - `runtime/parsing.py`
   - `runtime/reasoning_types.py`
   - `runtime/task_classification.py`
4. Rewire trunk's extracted controllers to those runtime-owned services instead of trying to resurrect cleanup's older `conversation.py` shape.
5. Re-apply only the cleanup deletions that still make sense after trunk's newer workflow/clarify changes.
6. Re-run:
   - `uv run pytest -q`
   - targeted workflow/runtime parity checks
   - the blocked live interactive validation matrix once backend chat is healthy

## What should not be lost

From trunk:

- semantic clarify grounding and pressure-pass work
- workflow state/policy/lane decomposition
- workflow recovery and artifact invalidation surfaces

From cleanup:

- zero direct `runtime -> agent/*` imports
- runtime-owned service seams instead of hidden legacy ownership
- deleted assistant puppeting behaviors
- the audit closure scoreboard and residual-debt honesty

## Bottom line

The divergence is real, but it is not bad news.

Trunk appears to have pushed the product/runtime decomposition story further.
Cleanup pushed the contract/ownership story further.

The right next move is to integrate cleanup's runtime-contract wins onto trunk's newer controller and workflow architecture, then re-sitrep from that integration branch.
