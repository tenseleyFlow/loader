# Loader Runtime Parity Checkpoint

Date: 2026-04-06

This file tracks the current deterministic runtime baseline for Loader. It stays intentionally narrow and operational: what the runtime can do today, what remains weak, and what scenarios we measure with repeatable tests.

## Supported today

- streamed text-only replies
- native-tool round trips for `read`, `write`, `edit`, `patch`, `glob`, `grep`, `bash`, `git`, `TodoWrite`, `AskUserQuestion`, `project_memory_*`, and `notepad_*`
- explicit permission modes: `read-only`, `workspace-write`, and `danger-full-access`
- tool lifecycle hooks in `pre_tool_use` → permission check → execute → `post_tool_use` / `post_tool_use_failure` order
- confirmation callbacks still exist for destructive `write` and `bash` actions after policy allows them
- raw JSON fallback when the model emits tool syntax in plain text
- persisted definition-of-done state under `.loader/dod/`
- persisted clarify briefs under `.loader/briefs/`
- persisted implementation and verification plans under `.loader/plans/`
- persisted conversation sessions under `.loader/sessions/` plus active session state under `.loader/state/`
- `loader --resume` and `loader --resume <session-id>` restore persisted session state
- durable project memory in `.loader/project-memory.json` and working notes in `.loader/notepad.md`
- native memory tools for `project_memory_*` and `notepad_*`
- heuristic workflow routing across `clarify` → `plan` → `execute` → `verify`
- mode-specific system prompts for clarify, plan, execute, and verify
- explicit verify/fix loops for mutating tasks, with a bounded retry budget
- verify/fix retries return to execute mode without re-triggering clarify or plan
- task-size-aware verification command derivation based on actual tool history
- verification command loading from persisted `verification.md` artifacts when present
- heuristic completion nudges only for non-mutating tasks; mutating tasks now complete through the DoD gate
- typed `TurnSummary` output for completed turns, including trace events and tool-result messages
- normalized per-turn usage plus cumulative session usage in `TurnSummary`
- automatic transcript compaction with priority-aware line compression and continuation instructions
- unified tool execution for native and extracted tool calls through `runtime.executor.ToolExecutor`
- typed tool-result messages backed by `Message.tool_results`
- `loader doctor` for backend, capability, workspace, command, state, and permission health checks outside the main runtime loop
- `loader status` plus `loader session list/show/resume` for inspecting persisted runtime state without invoking the LLM
- `loader explore <prompt>` as a one-shot read-only lookup lane with its own prompt, constrained registry, and no DoD or workflow routing
- CLI and TUI status surfaces for model, capability profile, mode, workflow mode, permission mode, DoD phase, pending items, last verification result, and active session id
- CLI and TUI workflow-mode visibility plus artifact notifications
- CLI and TUI permission-mode visibility with color-coded status
- workspace-bound file operations with canonicalized boundary checks, binary detection, size limits, and structured patch metadata
- shell mutability classification plus structured truncation and stderr/exit-code metadata
- richer structured `AskUserQuestion` prompts with titles, context, options, and optional freeform responses

## Known weak spots

- the core turn loop moved into [`src/loader/runtime/conversation.py`](../src/loader/runtime/conversation.py), but it is still much larger and more heuristic-heavy than the reference runtime in `refs/claw-code`
- planning, decomposition, and several helper behaviors still live in [`src/loader/agent/loop.py`](../src/loader/agent/loop.py), so ownership is cleaner than Sprint 00 but not fully simplified yet
- the mode router is still heuristic-only; Loader does not yet implement OMX's deeper ambiguity scoring, pressure-pass discipline, or branch-specific routing policy
- clarify mode currently stops after one structured question and one brief artifact; it does not yet run a deeper Socratic loop
- plan mode is still a single-pass artifact generator, not a Planner/Architect/Critic consensus loop
- DoD acceptance criteria and pending items are stronger than Sprint 02, but todo progress is still lightly structured compared with claw-code's richer workflow state
- evidence summaries are deterministic runtime summaries of captured output, not model-written verification narratives
- session compaction summaries are heuristic runtime summaries, not model-assisted continuity artifacts
- project-memory capture on finalized DoD evidence is still lightweight and command-summary oriented, not semantically curated memory extraction
- policy rules (`allow` / `deny` / `ask`) are still deferred, and the current permission system is mode-based rather than rule-based
- destructive tool calls still pass through the legacy confirmation path after policy allows them, so Loader has not fully reached claw-code's prompt/allow model
- shell safety is still heuristic and command-based; Loader does not yet have a richer shell sandbox or argument-aware mutability model
- explore mode is a one-shot read-only lane, not yet a richer interactive inspection workflow with deeper repo navigation affordances
- the read-only `git` helper is intentionally narrow compared with claw-code and OMX's broader repo/product surfaces, and the `patch` tool still stops short of AST/LSP-aware editing

## Out of scope in the current baseline

- richer permission rules / prompt mode / per-command allowlists
- multi-agent / team orchestration

## Deterministic parity scenarios

The auditable manifest lives at [`tests/fixtures/runtime_parity_manifest.json`](../tests/fixtures/runtime_parity_manifest.json) and is exercised by [`tests/test_runtime_harness.py`](../tests/test_runtime_harness.py). Sprint 04 also adds focused workflow integration coverage in [`tests/test_workflow_runtime.py`](../tests/test_workflow_runtime.py) and artifact/router unit coverage in [`tests/test_workflow.py`](../tests/test_workflow.py). Sprint 06 adds inspection/explore coverage in [`tests/test_inspection.py`](../tests/test_inspection.py), [`tests/test_explore_runtime.py`](../tests/test_explore_runtime.py), and [`tests/test_expanded_tools.py`](../tests/test_expanded_tools.py).

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
- `ambiguous_prompt_routes_to_clarify`: green
- `complex_prompt_routes_to_plan`: green
- `verify_failure_fix_loop_does_not_reroute_workflow`: green
- `conversational_task_skips_verify_phase`: green
- `explore_mode_skips_dod_and_router`: green
- `explore_mode_denies_write`: green

## Verification snapshot

As of 2026-04-06:

- `uv run pytest -q`: 153 passed
- `tests/test_runtime_harness.py` is fully green, including permission-mode parity, DoD verify/fix coverage, workflow routing parity, and the original contract regression
- `tests/test_dod.py` covers persistence, sizing boundaries, and verification command derivation
- `tests/test_workflow.py` covers router heuristics, clarify/plan artifact round trips, DoD workflow links, and todo-to-DoD syncing
- `tests/test_workflow_runtime.py` covers clarify routing, plan routing, and verify-fix workflow handoff
- `tests/test_workflow_tools.py` and `tests/test_workflow_runtime_tools.py` cover `TodoWrite`, `AskUserQuestion`, and runtime callback plumbing
- `tests/test_session_state.py` covers session persistence, resume, rotation, compaction persistence, and cumulative usage rollups
- `tests/test_compaction.py` covers claw-style line compression and compacted continuation-message behavior
- `tests/test_memory_tools.py` covers project-memory writes, notepad writes, lifecycle-hook mirroring, and DoD-summary capture into project memory
- `tests/test_cli_resume.py` covers `--resume` argument rewriting for latest and named-session restore
- `tests/test_inspection.py` covers `loader doctor`, `loader status`, `loader session list/show`, and session-resume CLI dispatch
- `tests/test_explore_runtime.py` covers the direct explore lane contract and forced read-only behavior outside the parity harness
- `tests/test_expanded_tools.py` covers structured patch application, read-only git helpers, `notepad_append`, and richer structured user questions
- `tests/test_permissions.py` covers permission policy overrides and hook lifecycle ordering
- `tests/test_tool_safety.py` covers workspace boundaries, binary/oversize guards, patch metadata, and shell truncation/classification
- `tests/test_status_surfaces.py` covers the CLI/TUI DoD, workflow-mode, permission-mode, capability-profile, and session-id formatting helpers
- native and extracted tool calls now record the same executor trace events, with source-specific metadata
- turn startup can refine backend capability profiles before the first request, `run_streaming()` delegates into the main runtime path, mutating tasks route through persisted evidence-backed completion, workflow artifacts survive across turns, sessions compact safely, explore queries bypass DoD/router overhead safely, and tool execution hangs off a stable hook-and-policy seam

## Definition of honesty

- If a scenario is green here, it should have deterministic automated coverage.
- If a scenario is flaky or broken, it should be called out here before we claim parity work is done.
- Sprint 01 turned the original `tool_call_id` regression green by fixing the message contract, not by weakening the test.
- Sprint 02 replaced "looks done" completion for mutating tasks with a real verify/fix gate, but it has not yet reached the richer workflow contracts described in the report and Sprint 04+.
- Sprint 03 established permission modes, hooks, and tool hardening, but it intentionally stops short of claw-code's fuller rule engine and prompt/allow permission variants.
- Sprint 04 adds routing, artifacts, and structured user questions, but it is still a first-pass workflow layer rather than full OMX consensus planning or deep interview rigor.
- Sprint 05 adds durable sessions, resume, compaction, and native memory/notepad tools, but it stops short of Sprint 06's inspectable session/status product surfaces and still uses heuristic continuity summaries rather than richer semantic memory extraction.
- Sprint 06 adds inspectable product surfaces, a constrained explore lane, and a broader tool registry, but it still stops short of interactive explore workflows, richer git ergonomics, AST/LSP-aware editing, or any multi-agent/team runtime.
