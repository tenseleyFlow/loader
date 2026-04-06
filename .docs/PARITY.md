# Loader Runtime Parity Checkpoint

Date: 2026-04-06

This file tracks the current deterministic runtime baseline for Loader. It stays intentionally narrow and operational: what the runtime can do today, what remains weak, and what scenarios we measure with repeatable tests.

## Supported today

- streamed text-only replies
- native-tool round trips for `read`, `write`, `edit`, `glob`, `grep`, and `bash`
- confirmation callbacks for destructive `write` and `bash` actions
- raw JSON fallback when the model emits tool syntax in plain text
- persisted definition-of-done state under `.loader/dod/`
- explicit verify/fix loops for mutating tasks, with a bounded retry budget
- task-size-aware verification command derivation based on actual tool history
- heuristic completion nudges only for non-mutating tasks; mutating tasks now complete through the DoD gate
- typed `TurnSummary` output for completed turns, including trace events and tool-result messages
- unified tool execution for native and extracted tool calls through `runtime.executor.ToolExecutor`
- typed tool-result messages backed by `Message.tool_results`
- CLI and TUI status surfaces for DoD phase, pending items, and last verification result

## Known weak spots

- the core turn loop moved into [`src/loader/runtime/conversation.py`](../src/loader/runtime/conversation.py), but it is still much larger and more heuristic-heavy than the reference runtime in `refs/claw-code`
- planning, decomposition, and several helper behaviors still live in [`src/loader/agent/loop.py`](../src/loader/agent/loop.py), so ownership is cleaner than Sprint 00 but not fully simplified yet
- DoD acceptance criteria and pending items are still runtime-derived and minimal, not model-authored task plans
- evidence summaries are deterministic runtime summaries of captured output, not model-written verification narratives
- permissions are confirmation-based, not policy-based

## Out of scope in the current baseline

- permission modes / policy engine
- persisted sessions / memory beyond DoD state
- mode router, clarify, or planning artifacts
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

- `uv run pytest -q`: 90 passed
- `tests/test_runtime_harness.py` is fully green, including DoD verify/fix coverage and the original contract regression
- `tests/test_dod.py` covers persistence, sizing boundaries, and verification command derivation
- `tests/test_status_surfaces.py` covers the CLI/TUI DoD status formatting helpers
- native and extracted tool calls now record the same executor trace events, with source-specific metadata
- turn startup can refine backend capability profiles before the first request, `run_streaming()` delegates into the main runtime path, and mutating tasks now route through persisted evidence-backed completion

## Definition of honesty

- If a scenario is green here, it should have deterministic automated coverage.
- If a scenario is flaky or broken, it should be called out here before we claim parity work is done.
- Sprint 01 turned the original `tool_call_id` regression green by fixing the message contract, not by weakening the test.
- Sprint 02 replaced "looks done" completion for mutating tasks with a real verify/fix gate, but it has not yet reached the richer workflow contracts described in the report and Sprint 04+.
