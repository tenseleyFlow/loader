# Loader Runtime Parity Checkpoint

Date: 2026-04-06

This file tracks the current deterministic runtime baseline for Loader. It stays intentionally narrow and operational: what the runtime can do today, what remains weak, and what scenarios we measure with repeatable tests.

## Supported today

- streamed text-only replies
- native-tool round trips for `read`, `write`, `edit`, `glob`, `grep`, and `bash`
- explicit permission modes: `read-only`, `workspace-write`, and `danger-full-access`
- tool lifecycle hooks in `pre_tool_use` → permission check → execute → `post_tool_use` / `post_tool_use_failure` order
- confirmation callbacks still exist for destructive `write` and `bash` actions after policy allows them
- raw JSON fallback when the model emits tool syntax in plain text
- persisted definition-of-done state under `.loader/dod/`
- explicit verify/fix loops for mutating tasks, with a bounded retry budget
- task-size-aware verification command derivation based on actual tool history
- heuristic completion nudges only for non-mutating tasks; mutating tasks now complete through the DoD gate
- typed `TurnSummary` output for completed turns, including trace events and tool-result messages
- unified tool execution for native and extracted tool calls through `runtime.executor.ToolExecutor`
- typed tool-result messages backed by `Message.tool_results`
- CLI and TUI status surfaces for DoD phase, pending items, and last verification result
- CLI and TUI permission-mode visibility with color-coded status
- workspace-bound file operations with canonicalized boundary checks, binary detection, size limits, and structured patch metadata
- shell mutability classification plus structured truncation and stderr/exit-code metadata

## Known weak spots

- the core turn loop moved into [`src/loader/runtime/conversation.py`](../src/loader/runtime/conversation.py), but it is still much larger and more heuristic-heavy than the reference runtime in `refs/claw-code`
- planning, decomposition, and several helper behaviors still live in [`src/loader/agent/loop.py`](../src/loader/agent/loop.py), so ownership is cleaner than Sprint 00 but not fully simplified yet
- DoD acceptance criteria and pending items are still runtime-derived and minimal, not model-authored task plans
- evidence summaries are deterministic runtime summaries of captured output, not model-written verification narratives
- policy rules (`allow` / `deny` / `ask`) are still deferred, and the current permission system is mode-based rather than rule-based
- destructive tool calls still pass through the legacy confirmation path after policy allows them, so Loader has not fully reached claw-code's prompt/allow model
- shell safety is still heuristic and command-based; Loader does not yet have a richer shell sandbox or argument-aware mutability model

## Out of scope in the current baseline

- persisted sessions / memory beyond DoD state
- mode router, clarify, or planning artifacts
- richer permission rules / prompt mode / per-command allowlists
- doctor / status / session product surfaces

## Deterministic parity scenarios

The auditable manifest lives at [`tests/fixtures/runtime_parity_manifest.json`](../tests/fixtures/runtime_parity_manifest.json) and is exercised by [`tests/test_runtime_harness.py`](../tests/test_runtime_harness.py).

- `streaming_text`: green
- `read_file_roundtrip`: green
- `multi_tool_turn_roundtrip`: green
- `write_file_allowed`: green
- `write_file_denied`: green
- `bash_stdout_roundtrip`: green
- `bash_confirmation_prompt_approved`: green
- `bash_confirmation_prompt_denied`: green
- `read_only_mode_denies_write`: green
- `read_only_mode_denies_mutating_bash`: green
- `read_only_mode_allows_safe_bash`: green
- `workspace_write_denies_write_outside_root`: green
- `danger_full_access_allows_dangerous_bash`: green
- `raw_json_tool_call_fallback`: green
- `completion_check_continuation`: green
- `tool_result_contract_regression`: green
- `turn_summary_smoke_for_multi_tool_turn`: green
- `native_and_raw_tool_paths_share_executor_trace`: green
- `backend_capability_probe_refreshes_native_tool_mode`: green
- `run_streaming_delegates_to_primary_runtime`: green
- `definition_of_done_verify_phase`: green
- `verify_failure_routes_to_fix_loop`: green
- `verify_retry_budget_exhaustion`: green
- `conversational_task_skips_verify_phase`: green

## Verification snapshot

As of 2026-04-06:

- `uv run pytest -q`: 106 passed
- `tests/test_runtime_harness.py` is fully green, including permission-mode parity, DoD verify/fix coverage, and the original contract regression
- `tests/test_dod.py` covers persistence, sizing boundaries, and verification command derivation
- `tests/test_permissions.py` covers permission policy overrides and hook lifecycle ordering
- `tests/test_tool_safety.py` covers workspace boundaries, binary/oversize guards, patch metadata, and shell truncation/classification
- `tests/test_status_surfaces.py` covers the CLI/TUI DoD and permission-mode formatting helpers
- native and extracted tool calls now record the same executor trace events, with source-specific metadata
- turn startup can refine backend capability profiles before the first request, `run_streaming()` delegates into the main runtime path, mutating tasks route through persisted evidence-backed completion, and tool execution now hangs off a stable hook-and-policy seam

## Definition of honesty

- If a scenario is green here, it should have deterministic automated coverage.
- If a scenario is flaky or broken, it should be called out here before we claim parity work is done.
- Sprint 01 turned the original `tool_call_id` regression green by fixing the message contract, not by weakening the test.
- Sprint 02 replaced "looks done" completion for mutating tasks with a real verify/fix gate, but it has not yet reached the richer workflow contracts described in the report and Sprint 04+.
- Sprint 03 established permission modes, hooks, and tool hardening, but it intentionally stops short of claw-code's fuller rule engine and prompt/allow permission variants.
