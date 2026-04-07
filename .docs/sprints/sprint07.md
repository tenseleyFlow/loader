# Sprint 07: Rule-Based Permissions and Runtime Decomposition

## Prerequisites

Sprint 06

## Goals

Finish the permission model and keep the runtime from re-forming a monolith.

Sprint 03 gave Loader explicit permission modes and lifecycle hooks. Sprint 06 made Loader more inspectable and product-like. The next leverage point is to replace the remaining legacy confirmation behavior with a real policy layer, then carve authorization and finalization concerns out of `src/loader/runtime/conversation.py` so later work does not keep accumulating there.

This is not a stop-the-world cleanup sprint. It is a contract sprint:

- policy decides whether tool use is allowed, denied, or prompted
- prompts are driven by authorization outcomes instead of ad hoc tool confirmations
- `conversation.py` becomes orchestration, not a sink for every new behavior

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/permissions.rs`
- `refs/claw-code/rust/crates/runtime/src/permission_enforcer.rs`
- `refs/claw-code/rust/crates/runtime/src/conversation.rs:295-470`

## Deliverables

### 1. Full permission policy layer

Extend Loader's current mode-based policy to a real rule-based policy.

Implementation targets:

- add `prompt` and `allow` to `PermissionMode`, mirroring `claw-code`
- extend `PermissionPolicy` with `allow_rules`, `deny_rules`, and `ask_rules`
- introduce a typed permission-rule representation with conservative matching over:
  - tool name
  - normalized tool input summary
  - optional workspace/path context where relevant
- define deterministic policy precedence:
  - deny rules always win
  - hook-level deny/cancel/fail still deny
  - ask rules and `prompt` mode route to interactive approval
  - allow rules and `allow` mode can elevate within policy
  - otherwise the required-mode gate still applies
- keep rule syntax intentionally narrow; do not invent a complex DSL in this sprint

The goal is to get the runtime to a `PermissionPolicy` shape closer to claw-code, not to build a policy language for its own sake.

### 2. Policy-backed prompting instead of legacy confirmations

Today Loader still falls back to legacy confirmation flows after policy allows certain destructive actions. This sprint should invert that relationship.

Implementation targets:

- route write/bash/edit/patch approvals through policy outcomes (`allow`, `deny`, `ask`)
- make `ToolExecutor` the primary owner of interactive approval decisions
- keep `Tool.check_confirmation()` only as a compatibility shim while call sites are migrated
- ensure prompt payloads include:
  - tool name
  - normalized input summary
  - active mode
  - required mode
  - matched rule or hook reason when available
- preserve Sprint 06's explore guarantee: explore mode remains forced `read-only` even if the broader session policy is `allow`

This is the behavioral finish to Sprint 03. Safety and approval should come from one runtime contract, not a policy layer plus leftover tool-specific prompting.

### 3. Split `conversation.py` into smaller runtime components

`src/loader/runtime/conversation.py` is working, but it is still too responsibility-dense. The next sprint should reduce risk by moving major responsibilities into dedicated runtime modules.

Implementation targets:

- extract assistant-turn request/response handling into a smaller request/turn helper
- extract tool-batch execution and retry bookkeeping into a dedicated runtime component
- extract verify/fix completion gating and session-finalization concerns into a dedicated runtime component
- keep `ConversationRuntime.run_turn(...)` as the coordinator that wires those parts together
- avoid moving behavior back into `agent/loop.py`; decomposition should continue inside `src/loader/runtime/`

A good outcome is not just fewer lines. A good outcome is that future work on authorization, verification, and completion lands in focused modules instead of reopening the turn loop every time.

### 4. Observable policy state in product surfaces

If Loader gains rule-based permissions, operators need to be able to see that state without reading code.

Implementation targets:

- extend `loader status` to show:
  - active permission mode
  - whether policy prompting is enabled
  - allow/deny/ask rule counts
- extend `loader doctor` to validate policy configuration and warn on invalid or conflicting rules
- persist enough policy metadata in session/runtime state that inspection surfaces can explain the effective policy cleanly
- fail closed on invalid policy configuration and report the failure clearly

This keeps the policy layer inspectable and reduces surprise when a tool is prompted, denied, or silently allowed by rule.

## Testing strategy

- unit coverage for:
  - `PermissionMode` parsing including `prompt` and `allow`
  - rule parsing and normalization
  - precedence between deny/ask/allow rules and hook overrides
  - invalid policy configuration failing closed
- deterministic harness coverage for:
  - `prompt_mode_prompts_destructive_write`
  - `allow_mode_skips_prompt_for_destructive_write`
  - `deny_rule_blocks_allowed_mode`
  - `ask_rule_prompts_even_when_mode_would_allow`
  - `explore_mode_ignores_global_allow_policy`
- inspection coverage for:
  - doctor reporting invalid policy files
  - status/session surfaces showing rule summaries and active policy mode
- regression coverage:
  - Sprint 00-06 parity scenarios remain green after the runtime split
  - no new behavior path should bypass `ToolExecutor`

## Definition of done

- Loader can express `read-only`, `workspace-write`, `danger-full-access`, `prompt`, and `allow` modes
- permission approvals and denials come from policy evaluation, not mostly from leftover tool-specific confirmation logic
- allow/deny/ask rules are real, typed, tested, and visible in product surfaces
- `conversation.py` is slimmer and more coordinator-like, with authorization/finalization logic living in dedicated runtime modules
- the existing parity baseline stays green and the new policy scenarios are deterministic
- Loader is closer to claw-code's execution-policy contract without taking on claw-code's full complexity

## Explicitly out of scope

- interactive multi-step explore workflows
- AST-aware, LSP-aware, or symbol-aware editing
- multi-agent or team orchestration
- broad plugin or MCP expansion

## Audit

### Landed

- Loader now supports the full Sprint 07 permission-mode surface: `read-only`, `workspace-write`, `danger-full-access`, `prompt`, and `allow`
- workspace-local `.loader/permission-rules.json` files now load into typed `allow` / `deny` / `ask` rule sets with conservative matching over tool name, normalized input summaries, and optional path hints
- policy precedence is now deterministic: deny rules win first, hook-level overrides can still deny/ask/allow, ask rules drive interactive approval, allow rules and `allow` mode can elevate, and the required-mode gate still applies otherwise
- `ToolExecutor` is now the primary owner of interactive approval decisions, and destructive write/bash/edit/patch paths run through policy outcomes instead of relying mainly on tool-specific confirmation logic
- policy prompt payloads now include the active mode, required mode, normalized input summary, and matched rule or hook reason where available
- explore mode preserves the Sprint 06 read-only guarantee by copying deny/ask rules but intentionally ignoring broader allow rules
- `loader doctor` and `loader status` now expose prompting state, allow/deny/ask rule counts, and invalid rule configuration clearly, while `loader session list/show` now persist and surface the effective policy metadata that a session actually ran with
- invalid permission configuration now fails closed both when starting an `Agent` and when inspecting a workspace through doctor/status surfaces
- runtime decomposition continued inside `src/loader/runtime/`: assistant requests now live in `assistant_turns.py`, tool-batch execution/recovery/post-tool verification now live in `tool_batches.py`, and DoD/finalization logic now live in `finalization.py`
- `ConversationRuntime.run_turn(...)` is now more coordinator-like: it prepares the workflow, requests an assistant turn, delegates tool execution to the batch runner, delegates DoD gating/finalization, and keeps orchestration in one place instead of owning every behavior directly
- the deterministic parity harness now includes prompt/allow/rule-policy scenarios and remains green after the runtime split

### Verification

- `uv run pytest -q` is green: `167 passed`
- `tests/test_permissions.py` covers `prompt` / `allow` parsing, rule parsing, deny/ask/allow precedence, hook overrides, and policy-backed prompting behavior
- `tests/test_runtime_harness.py` keeps the full Sprint 00-06 baseline green and now covers prompt-mode prompting, allow-mode skipping, deny-rule blocking, ask-rule prompting, and explore-mode isolation from global allow policy
- `tests/test_inspection.py` covers invalid rule reporting plus rule-aware `doctor`, `status`, `session list`, and `session show` surfaces
- `tests/test_session_state.py` now covers persisted permission-policy metadata alongside the earlier session persistence/resume/compaction coverage
- targeted `ruff` checks are green for the new runtime modules and the touched inspection/session test files

### Residual debt

- rule syntax is intentionally narrow and workspace-local; Loader still does not have claw-code's richer rule model, preview UX, or temporary allow/deny override ergonomics
- policy-backed prompting is now primary, but the older tool-confirmation compatibility layer still exists and should continue shrinking rather than becoming a second policy path again
- `conversation.py` is materially slimmer than before Sprint 07, but it still owns workflow routing, prompt repair, self-critique/completion heuristics, and several other coordinator behaviors that remain more heuristic-heavy than the refs
- shell mutability classification and rule matching are still conservative string/command heuristics rather than a richer semantic sandbox or argument-aware policy model
- session inspection now preserves effective policy state, but Loader still does not offer a first-class product surface for authoring, validating, or dry-running permission rules
- explore mode remains intentionally one-shot and read-only; Sprint 07 does not add a richer interactive inspection workflow
