# Sprint 09 Interactive Validation — Native Lane

## Backend Summary

- Date: 2026-04-07
- Operator: Codex
- Branch: `cleanup-audit-plan`
- Capture base: `f60523c` (`Probe live chat health in doctor`)
- Backend: `ollama`
- Models exercised:
  - `qwen2.5:7b`
  - `qwen2.5:14b`
- Capability profile:
  - `qwen2.5:7b` → `native`
  - `qwen2.5:14b` → `native`
- Native tools enabled: yes
- Streaming enabled: yes
- Permission mode: `read-only`
- Workflow override: none (`execute`)

## Run 1

- Task: `Say hello in five words.`
- Expected productive path: one assistant turn, no tools, immediate final response
- Expected risky heuristics: none; this should not need repair or completion nudges
- Actual outcome: failed before the first assistant turn completed
- Verification outcome: not reached
- Final response quality: none; CLI exited with `httpx.HTTPStatusError`

### Evidence

- Doctor result: `uv run loader doctor -m qwen2.5:7b` reported `backend: pass`, `chat: fail`, and capabilities `pass`
- Runtime invocation: `uv run loader -m qwen2.5:7b --no-tui --permission-mode read-only "Say hello in five words."`
- Tool path used: none
- Phase trace: startup banner printed, then `Generating...`, then `/api/chat` failed with HTTP 500 before the first streamed chunk
- Recovery layers fired: none observed
- Session/runtime notes:
  - Loader selected `Mode: Native`
  - session id: `20260407T190318Z-2f8a2087`
  - failure site: `src/loader/llm/ollama.py:363` during `backend.stream(...)`

### Assessment

- Did the runtime help or get in the way? The runtime did not get a chance to help or interfere; the backend failed before the first assistant turn was materialized.
- Which heuristics were load-bearing? None in this run.
- Which heuristics felt like churn? None in this run.
- Proposed disposition updates:
  - Do not draw recovery-layer conclusions from this run.
  - Treat the native validation lane as blocked on live `/api/chat` reliability, not on Loader turn logic.

## Run 2

- Task: `Say hello in five words.`
- Expected productive path: one assistant turn, no tools, immediate final response
- Expected risky heuristics: none
- Actual outcome: failed before the first assistant turn completed
- Verification outcome: not reached
- Final response quality: none; CLI exited with `httpx.HTTPStatusError`

### Evidence

- Doctor result: `uv run loader doctor -m qwen2.5:14b` reported `backend: pass`, `chat: fail`, and capabilities `pass`
- Runtime invocation: `uv run loader -m qwen2.5:14b --no-tui --permission-mode read-only "Say hello in five words."`
- Tool path used: none
- Phase trace: startup banner printed, then `Generating...`, then `/api/chat` failed with HTTP 500 before the first streamed chunk
- Recovery layers fired: none observed
- Session/runtime notes:
  - Loader selected `Mode: Native`
  - session id: `20260407T190412Z-0eaaa107`
  - failure site: `src/loader/llm/ollama.py:363` during `backend.stream(...)`

### Assessment

- Did the runtime help or get in the way? Same result as Run 1; the runtime never got past the first backend chat request.
- Which heuristics were load-bearing? None in this run.
- Which heuristics felt like churn? None in this run.
- Proposed disposition updates:
  - The failure is not isolated to one native-capable model.
  - Keep the native Sprint 09 lane open, but mark it blocked until Ollama chat requests succeed again.

## Summary

- Most useful runtime behaviors: `loader doctor` now separates model availability from live chat readiness, which makes the blockage explicit instead of implying the lane is healthy.
- Least useful runtime behaviors: none assessed; runtime behavior was not exercised.
- Recovery layers that should likely be deleted: no update from this artifact
- Recovery layers that should likely be gated by capability profile: no update from this artifact
- Follow-up code or test tasks:
  - investigate why `doctor` passes on `/api/tags` and `/api/show` while live `/api/chat` requests fail with HTTP 500
  - rerun this lane with a file-read task once chat requests are healthy, so the runtime can actually exercise native-tool behavior
