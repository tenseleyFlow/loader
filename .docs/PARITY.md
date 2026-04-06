# Loader Runtime Baseline

Date: 2026-04-06

This file is the Sprint 00 baseline for Loader's current runtime behavior. It is intentionally narrow and operational: what the loop can do today, what is flaky, what is out of scope, and what scenarios we now measure with deterministic tests.

## Supported today

- streamed text-only replies
- native-tool round trips for `read`, `write`, `edit`, `glob`, `grep`, and `bash`
- confirmation callbacks for destructive `write` and `bash` actions
- raw JSON fallback when the model emits tool syntax in plain text
- heuristic completion nudges when the model stops before finishing a simple actionable task

## Known weak spots

- the main runtime still lives in one large loop at [`src/loader/agent/loop.py`](../src/loader/agent/loop.py)
- duplicate suppression and pre-validation still try to construct `Message(..., tool_call_id=...)`, which is a known broken contract until Sprint 01 lands
- extracted raw-text tool execution duplicates the main tool execution path
- completion is still heuristic, not evidence-backed
- permissions are confirmation-based, not policy-based

## Out of scope in the current baseline

- typed turn engine / unified executor
- permission modes
- persisted sessions / memory / `.loader/` runtime state
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
- `tool_result_contract_regression`: intentionally red in Sprint 00

## Verification snapshot

As of 2026-04-06:

- `uv run pytest`: 70 passed, 1 failed
- the single failing test is `tests/test_runtime_harness.py::test_tool_result_contract_regression`
- that regression proves both broken branches currently raise `TypeError: Message.__init__() got an unexpected keyword argument 'tool_call_id'`

## Definition of honesty

- If a scenario is green here, it should have deterministic automated coverage.
- If a scenario is flaky or broken, it should be called out here before we claim parity work is done.
- Sprint 01 should turn the intentional red regression green by fixing the tool-result message contract, not by weakening the test.
