# Loader Runtime Parity Checkpoint

Date: 2026-04-07

This file tracks the current deterministic runtime baseline for Loader. It stays intentionally narrow and operational: what the runtime can do today, what remains weak, and what scenarios we measure with repeatable tests.

## Supported today

- streamed text-only replies
- native-tool round trips for `read`, `write`, `edit`, `patch`, `glob`, `grep`, `bash`, `git`, `TodoWrite`, `AskUserQuestion`, `project_memory_*`, and `notepad_*`
- explicit permission modes: `read-only`, `workspace-write`, `danger-full-access`, `prompt`, and `allow`
- tool lifecycle hooks in `pre_tool_use` → permission check → execute → `post_tool_use` / `post_tool_use_failure` order
- rule-based permission policy with workspace-local `allow` / `deny` / `ask` rules from `.loader/permission-rules.json`
- policy-backed prompting for destructive tool use, with approval context that includes mode, requirement, and matched rule information
- raw JSON fallback when the model emits tool syntax in plain text
- persisted definition-of-done state under `.loader/dod/`
- persisted clarify briefs under `.loader/briefs/`
- persisted implementation and verification plans under `.loader/plans/`
- persisted conversation sessions under `.loader/sessions/` plus active session state under `.loader/state/`
- persisted permission policy metadata alongside session state, so `loader status` / `loader session list` / `loader session show` can explain the effective policy that ran
- `loader --resume` and `loader --resume <session-id>` restore persisted session state
- durable project memory in `.loader/project-memory.json` and working notes in `.loader/notepad.md`
- native memory tools for `project_memory_*` and `notepad_*`
- heuristic workflow routing across `clarify` → `plan` → `execute` → `verify`
- clarify mode as an explicit single-question brief flow that returns to execute mode
- plan mode as explicit single-pass implementation and verification artifact generation
- persisted workflow-artifact status and source metadata in session state when execute consumes or reuses workflow artifacts
- mode-specific system prompts for clarify, plan, execute, and verify
- explicit verify/fix loops for mutating tasks, with a bounded retry budget
- verify/fix retries return to execute mode without re-triggering clarify or plan
- task-size-aware verification command derivation based on actual tool history
- verification command loading from persisted `verification.md` artifacts when present
- mutating tasks complete through the DoD gate, and non-mutating tasks now return their answer directly instead of injecting continuation nudges
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
- assistant-turn request handling now lives in `runtime.assistant_turns`, tool-batch execution/recovery now lives in `runtime.tool_batches`, and DoD/finalization logic now lives in `runtime.finalization` instead of accumulating further inside `conversation.py`

## Known weak spots

- the core turn loop moved into [`src/loader/runtime/conversation.py`](../src/loader/runtime/conversation.py), but it still owns workflow routing, remaining loop safeguards, and other coordination logic that remains more heuristic-heavy than the reference runtime in `refs/claw-code`
- workflow routing is cleaner than Sprint 00, but the router and artifact bridge still live in [`src/loader/runtime/conversation.py`](../src/loader/runtime/conversation.py) and remain more heuristic than the reference runtimes
- the mode router is still heuristic-only; Loader does not yet implement OMX's deeper ambiguity scoring, pressure-pass discipline, or branch-specific routing policy
- clarify mode is now explicitly a single-question brief flow, not a deeper Socratic protocol
- plan mode is now explicitly a single-pass artifact generator, not a Planner/Architect/Critic consensus loop
- DoD acceptance criteria and pending items are stronger than Sprint 02, but todo progress is still lightly structured compared with claw-code's richer workflow state
- evidence summaries are deterministic runtime summaries of captured output, not model-written verification narratives
- session compaction summaries are heuristic runtime summaries, not model-assisted continuity artifacts
- project-memory capture on finalized DoD evidence is still lightweight and command-summary oriented, not semantically curated memory extraction
- rule syntax is intentionally narrow and workspace-local; Loader still does not have claw-code's richer rule model or broader prompt/allow operator surface
- policy state is inspectable in doctor/status/session surfaces, but there is not yet a richer UX for editing, previewing, or temporarily overriding rules from the product surface
- shell safety is still heuristic and command-based; Loader does not yet have a richer shell sandbox or argument-aware mutability model
- explore mode is a one-shot read-only lane, not yet a richer interactive inspection workflow with deeper repo navigation affordances
- the read-only `git` helper is intentionally narrow compared with claw-code and OMX's broader repo/product surfaces, and the `patch` tool still stops short of AST/LSP-aware editing

## Out of scope in the current baseline

- richer permission-rule UX / per-command allowlists
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
- `prompt_mode_prompts_destructive_write`: green
- `allow_mode_skips_prompt_for_destructive_write`: green
- `deny_rule_blocks_allowed_mode`: green
- `ask_rule_prompts_even_when_mode_would_allow`: green
- `raw_json_tool_call_fallback`: green
- `non_mutating_completion_no_longer_forces_continuation`: green
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
- `explore_mode_ignores_global_allow_policy`: green

## Verification snapshot

As of 2026-04-07:

- `uv run pytest -q`: 211 passed
- `tests/test_runtime_harness.py` is fully green, including permission-mode parity, DoD verify/fix coverage, workflow routing parity, and the original contract regression
- `tests/test_dod.py` covers persistence, sizing boundaries, and verification command derivation
- `tests/test_workflow.py` covers router heuristics, clarify/plan artifact round trips, DoD workflow links, and todo-to-DoD syncing
- `tests/test_workflow_runtime.py` covers clarify routing, plan routing, and verify-fix workflow handoff
- `tests/test_workflow_tools.py` and `tests/test_workflow_runtime_tools.py` cover `TodoWrite`, `AskUserQuestion`, and runtime callback plumbing
- `tests/test_session_state.py` covers session persistence, resume, rotation, compaction persistence, cumulative usage rollups, persisted permission-policy metadata, and persisted workflow-artifact state
- `tests/test_compaction.py` covers claw-style line compression and compacted continuation-message behavior
- `tests/test_memory_tools.py` covers project-memory writes, notepad writes, lifecycle-hook mirroring, and DoD-summary capture into project memory
- `tests/test_cli_resume.py` covers `--resume` argument rewriting for latest and named-session restore
- `tests/test_inspection.py` covers `loader doctor`, `loader status`, `loader session list/show`, and session-resume CLI dispatch
- `tests/test_explore_runtime.py` covers the direct explore lane contract and forced read-only behavior outside the parity harness
- `tests/test_expanded_tools.py` covers structured patch application, read-only git helpers, `notepad_append`, and richer structured user questions
- `tests/test_permissions.py` covers prompt/allow mode parsing, rule precedence, policy-backed prompting behavior, and hook lifecycle ordering
- `tests/test_tool_safety.py` covers workspace boundaries, binary/oversize guards, patch metadata, and shell truncation/classification
- `tests/test_status_surfaces.py` covers the CLI/TUI DoD, workflow-mode, permission-mode, capability-profile, and session-id formatting helpers
- native and extracted tool calls now record the same executor trace events, with source-specific metadata
- turn startup can refine backend capability profiles before the first request, `run_streaming()` delegates into the main runtime path, mutating tasks route through persisted evidence-backed completion, workflow artifacts survive across turns, sessions compact safely, explore queries bypass DoD/router overhead safely, policy rules are enforced deterministically, session inspection preserves effective policy state, and assistant-turn/tool-batch/finalization concerns now hang off smaller runtime modules instead of growing the conversation monolith further

## Definition of honesty

- If a scenario is green here, it should have deterministic automated coverage.
- If a scenario is flaky or broken, it should be called out here before we claim parity work is done.
- Sprint 01 turned the original `tool_call_id` regression green by fixing the message contract, not by weakening the test.
- Sprint 02 replaced "looks done" completion for mutating tasks with a real verify/fix gate, but it has not yet reached the richer workflow contracts described in the report and Sprint 04+.
- Sprint 03 established permission modes, hooks, and tool hardening, but it intentionally stops short of claw-code's fuller rule engine and prompt/allow permission variants.
- Sprint 04's workflow layer is now explicitly scoped as lightweight: single-question clarify, single-pass planning, explicit artifact bridging, and no legacy decomposition path.
- Sprint 05 adds durable sessions, resume, compaction, and native memory/notepad tools, but it stops short of Sprint 06's inspectable session/status product surfaces and still uses heuristic continuity summaries rather than richer semantic memory extraction.
- Sprint 06 adds inspectable product surfaces, a constrained explore lane, and a broader tool registry, but it still stops short of interactive explore workflows, richer git ergonomics, AST/LSP-aware editing, or any multi-agent/team runtime.
- Sprint 07 is complete: Loader now has prompt/allow modes, rule-based permission policy, policy-backed prompting, persisted policy inspection state, and smaller assistant-turn/tool-batch/finalization runtime seams, but it still stops short of a richer rule UX, deeper policy sandboxing, and the more opinionated workflow/runtime contracts in the refs.
