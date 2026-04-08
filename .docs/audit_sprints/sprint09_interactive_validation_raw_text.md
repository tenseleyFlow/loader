# Sprint 09 Interactive Validation — Raw-Text-Prone Lane

## Backend Summary

- Date: 2026-04-07
- Operator: Codex
- Branch: `cleanup-audit-plan`
- Capture base: `f60523c` (`Probe live chat health in doctor`)
- Backend: `ollama`
- Models exercised:
  - `qwen3-coder:30b`
  - `gemma3:12b`
  - `llama2:latest` (doctor only)
- Capability profile:
  - `qwen3-coder:30b` → `json_tag`
  - `gemma3:12b` → `json_tag`
  - `llama2:latest` → `json_tag`
- Native tools enabled: no
- Streaming enabled: yes
- Permission mode: `read-only`
- Workflow override: none (`execute`)

## Run 1

- Task: `Say hello in five words.`
- Expected productive path: one assistant turn, no tools, immediate final response
- Expected risky heuristics: none; even the raw-text lane should handle this without fallback parsing
- Actual outcome: failed before the first assistant turn completed
- Verification outcome: not reached
- Final response quality: none; CLI exited with `httpx.HTTPStatusError`

### Evidence

- Doctor result: `uv run loader doctor -m qwen3-coder:30b` reported `backend: pass`, `chat: fail`, and capabilities `warn` (`json_tag`)
- Runtime invocation: `uv run loader -m qwen3-coder:30b --no-tui --permission-mode read-only --react "Say hello in five words."`
- Tool path used: none
- Phase trace: startup banner printed, then `Generating...`, then `/api/chat` failed with HTTP 500 before the first streamed chunk
- Recovery layers fired: none observed
- Session/runtime notes:
  - Loader selected `Mode: ReAct`
  - session id: `20260407T190318Z-a3a6385a`
  - failure site: `src/loader/llm/ollama.py:363` during `backend.stream(...)`

### Assessment

- Did the runtime help or get in the way? The runtime did not get a usable response back from the backend, so the raw-text tool fallback was never exercised.
- Which heuristics were load-bearing? None in this run.
- Which heuristics felt like churn? None in this run.
- Proposed disposition updates:
  - Do not use this run to justify keeping or deleting raw-text repair heuristics.
  - Treat the raw-text-prone lane as blocked on live `/api/chat` reliability.

## Run 2

- Task: `Say hello in five words.`
- Expected productive path: one assistant turn, no tools, immediate final response
- Expected risky heuristics: none
- Actual outcome: failed before the first assistant turn completed
- Verification outcome: not reached
- Final response quality: none; CLI exited with `httpx.HTTPStatusError`

### Evidence

- Doctor result: `uv run loader doctor -m gemma3:12b` reported `backend: pass`, `chat: fail`, and capabilities `warn` (`json_tag`)
- Runtime invocation: `uv run loader -m gemma3:12b --no-tui --permission-mode read-only --react "Say hello in five words."`
- Tool path used: none
- Phase trace: startup banner printed, then `Generating...`, then `/api/chat` failed with HTTP 500 before the first streamed chunk
- Recovery layers fired: none observed
- Session/runtime notes:
  - Loader selected `Mode: ReAct`
  - session id: `20260407T190412Z-7e745da3`
  - failure site: `src/loader/llm/ollama.py:363` during `backend.stream(...)`

### Assessment

- Did the runtime help or get in the way? Same result as Run 1; the runtime never reached the point where ReAct parsing or repair could matter.
- Which heuristics were load-bearing? None in this run.
- Which heuristics felt like churn? None in this run.
- Proposed disposition updates:
  - The block is not isolated to one ReAct-profile model.
  - Keep the raw-text Sprint 09 lane open, but mark it blocked until live chat requests succeed.

## Run 3

- Task: doctor-only preflight for a second fallback-profile model
- Expected productive path: confirm the lane is not blocked by model availability alone
- Expected risky heuristics: none
- Actual outcome: doctor passed for `llama2:latest`, but no live run was attempted after the matching `/api/chat` failures in Runs 1 and 2
- Verification outcome: not applicable
- Final response quality: not applicable

### Evidence

- Doctor result: `uv run loader doctor -m llama2:latest` reported `backend: pass`, `chat: fail`, and capabilities `warn` (`json_tag`)
- Tool path used: none
- Phase trace: not applicable
- Recovery layers fired: none observed
- Session/runtime notes:
  - This run was intentionally limited to preflight evidence because the lane was already blocked by repeated `/api/chat` failures.

### Assessment

- Did the runtime help or get in the way? Not assessed.
- Which heuristics were load-bearing? Not assessed.
- Which heuristics felt like churn? Not assessed.
- Proposed disposition updates:
  - Once `/api/chat` is stable again, prioritize this model family for a real raw-text fallback task that exercises `read`, `TodoWrite`, `patch`, and `AskUserQuestion`.

## Summary

- Most useful runtime behaviors: `loader doctor` now distinguishes capability classification from live chat readiness, so the raw-text lane blockage is explicit.
- Least useful runtime behaviors: none assessed; runtime behavior was not exercised.
- Recovery layers that should likely be deleted: no update from this artifact
- Recovery layers that should likely be gated by capability profile: no update from this artifact
- Follow-up code or test tasks:
  - investigate the `/api/chat` failure independently of Sprint 09 runtime deletion work
  - rerun this lane with a raw-text tool-use task as soon as live chat requests succeed so the fallback machinery can be evaluated on real evidence
