# Sprint 09 Baseline and Recovery Inventory

## Snapshot

- Branch: `cleanup-audit-plan`
- Worktree: `/tmp/loader-audit-cleanup`
- Source snapshot for this baseline: `5c10aab` (`Harden raw tool-call fallback coverage`)
- Targeted verification completed before writing this baseline:
  - `uv run pytest -q tests/test_parsing.py`
  - `uv run pytest -q tests/test_runtime_harness.py -k 'raw_json or native_and_raw_tool_paths_share_executor_trace or runtime_parity_manifest_matches_implemented_cases'`
- Repo-wide verification after the latest Sprint 09 characterization update:
  - `uv run pytest -q` → `191 passed`
- Doctor surface update:
  - `f60523c` adds a dedicated live chat probe so `loader doctor` now reports `backend` reachability separately from `/api/chat` readiness
- New Sprint 09 guardrails now cover raw-text recovery for:
  - `read`
  - `TodoWrite`
  - `patch`
  - `AskUserQuestion`

## Runtime Ownership Baseline

| File | Lines | `self.agent.` reach-ins |
| --- | ---: | ---: |
| `src/loader/runtime/conversation.py` | 881 | 49 |
| `src/loader/runtime/assistant_turns.py` | 155 | 23 |
| `src/loader/runtime/tool_batches.py` | 372 | 26 |
| `src/loader/runtime/finalization.py` | 339 | 8 |
| `src/loader/runtime/completion_policy.py` | 187 | 9 |
| `src/loader/runtime/repair.py` | 208 | 4 |
| `src/loader/runtime/explore.py` | 220 | 12 |

This is the migration scoreboard for Sprint 10. The goal is not only to move code around, but to remove `Agent` as the runtime's implicit data model.

## Legacy Tree Baseline

| File | Lines |
| --- | ---: |
| `src/loader/agent/loop.py` | 1111 |
| `src/loader/agent/reasoning.py` | 1235 |
| `src/loader/agent/safeguards.py` | 1142 |
| `src/loader/agent/recovery.py` | 648 |

This is the subtraction scoreboard for Sprint 11 and Sprint 13.

## Recovery Inventory

| Behavior | Current owner | Trigger | Dependency | Current coverage | Proposed disposition |
| --- | --- | --- | --- | --- | --- |
| Prefill trick | `src/loader/runtime/conversation.py:142-161` | First iteration, single user message, action-keyword heuristic | Direct session write of fake assistant `[` | `tests/test_runtime_repair_flows.py::test_fresh_agent_messages_are_disconnected_from_session_history` | Delete. Fresh sessions currently keep `agent.messages` disconnected from `session.messages`, so this gate is already stale in practice. |
| Empty-output retry prompts | `src/loader/runtime/repair.py:43-76` via `conversation.py:192-211` | Assistant content is empty up to `max_empty_retries=5` | Five fake assistant continuation prompts | `tests/test_runtime_repair_flows.py::test_empty_response_repair_injects_retry_prompt_and_recovers` | Delete or reduce to one bounded retry with honest failure |
| Raw-text tool fallback | `src/loader/runtime/repair.py:101-125` plus `src/loader/agent/parsing.py` and legacy `src/loader/agent/loop.py:862-1111` | Native tool call list is empty but response contains tool syntax | Parser stack, capability-profile behavior, legacy extractor | `tests/test_parsing.py`, `tests/test_runtime_harness.py` raw JSON scenarios | Keep short-term, gate by capability profile, unify in Sprint 11 |
| Fake-tool narration repair | `src/loader/runtime/repair.py:156-182` plus `src/loader/agent/loop.py:770-860` | `_contains_unexecuted_code(...)` matches narration or code-block heuristics | Legacy regex wall plus injected scolding prompt | `tests/test_runtime_repair_flows.py::test_fake_tool_narration_repair_injects_scolding_prompt` | Delete |
| Deflection repair | `src/loader/runtime/repair.py:184-201` | Non-ReAct response deflects with "you can/should/could/try running" and no actions taken | Phrase heuristic plus injected user repair turn | `tests/test_runtime_repair_flows.py::test_deflection_repair_injects_use_your_tools_prompt` | Delete unless interactive evidence shows it is load-bearing |
| Self-critique reroute | `src/loader/runtime/completion_policy.py:49-90` | Long response and `should_self_critique(...)` says revise | `agent._self_critique`, reasoning prompt, session reinjection | `tests/test_runtime_repair_flows.py::test_self_critique_reroutes_long_code_response_for_revision` | Gate tightly or delete |
| Text-loop bailout | `src/loader/runtime/completion_policy.py:92-125` | `self.agent.safeguards.detect_text_loop(...)` reports repetition | `agent/safeguards.py` action tracker | `tests/test_runtime_repair_flows.py::test_text_loop_bailout_stops_after_repeated_continuation_response` | Keep only if moved toward a session-level safeguard |
| Non-mutating completion nudge | `src/loader/runtime/completion_policy.py:127-170` | `completion_check` enabled, no mutating actions, `detect_premature_completion(...)` hits | `agent/reasoning.py` continuation heuristics plus session reinjection | `tests/test_runtime_harness.py::test_completion_check_continuation` | Likely narrow sharply or delete |
| Post-action follow-up suffix | `src/loader/runtime/completion_policy.py:173-186` | Actions were taken and final text does not already end in `?` | Pure string heuristic | `tests/test_runtime_repair_flows.py::test_post_action_follow_up_suffix_is_appended_to_final_response` | Delete |
| Action-loop bailout | `src/loader/runtime/tool_batches.py:228-250` | `self.agent.safeguards.detect_loop()` reports repeated tool behavior | `agent/safeguards.py` action tracker | `tests/test_runtime_repair_flows.py::test_action_loop_bailout_stops_repeating_tool_pattern` | Keep candidate, but move behind a clearer runtime/service seam |

## Interactive Validation Matrix

These runs are still pending. They require at least one configured real native-tool backend and one configured raw-text-prone backend.

| Lane | Task | Why it matters | Status |
| --- | --- | --- | --- |
| Native-tool lane | Read a file, then write a small file and let DoD verify it | Confirms the runtime can stay on the normal native path without repair machinery stepping in | Blocked: `loader doctor` now reports `backend: pass` but `chat: fail`, and live `/api/chat` still fails with HTTP 500 before turn execution. See `sprint09_interactive_validation_native.md`. |
| Native-tool lane | Ambiguous request that routes through clarify mode | Measures whether current clarify behavior is helpful or just extra prompt text | Blocked behind the same native-lane `/api/chat` failure. |
| Native-tool lane | Multi-step implementation that uses `TodoWrite` and verification | Measures if completion behavior stays disciplined without fake continuations | Blocked behind the same native-lane `/api/chat` failure. |
| Raw-text-prone lane | Recover `read`, `patch`, `TodoWrite`, and `AskUserQuestion` from raw JSON/text | Confirms which raw fallback paths are still load-bearing after the new guardrails | Blocked: `loader doctor` now reports `backend: pass` but `chat: fail`, and live `/api/chat` fails with HTTP 500 before any raw-text output is produced. See `sprint09_interactive_validation_raw_text.md`. |
| Raw-text-prone lane | Prompt that tends to elicit narrated fake tool use | Measures whether fake-tool repair actually saves the run or just churns the conversation | Blocked behind the same raw-text-lane `/api/chat` failure. |
| Raw-text-prone lane | Prompt that tends to return empty or deflective text | Measures whether empty-output and deflection repairs help enough to justify keeping them | Blocked behind the same raw-text-lane `/api/chat` failure. |

Use [sprint09_interactive_validation.md](sprint09_interactive_validation.md) as the capture format for each completed run set.

Completed captures:

- [Native lane](sprint09_interactive_validation_native.md)
- [Raw-text-prone lane](sprint09_interactive_validation_raw_text.md)

## Immediate Sprint 09 Follow-on

- Use the `191 passed` repo-wide baseline as the regression floor for the next Sprint 09 slices.
- Restore a working live chat backend, then rerun the interactive validation matrix against the documented native and raw-text-prone model lanes.
- Keep `191 passed` as the regression floor for any Sprint 10 runtime-seam work that begins before the backend is healthy again.
- Use this inventory as the checklist for Sprint 10 service seams and Sprint 11 deletions.
