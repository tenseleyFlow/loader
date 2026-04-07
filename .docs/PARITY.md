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
- `loader permissions show` for normalized rule inspection, source-path visibility, and prompt-state inspection without opening JSON files by hand
- `loader permissions check` for dry-running one hypothetical tool request against the active policy, including required mode, normalized input summary, and matched-rule reasoning
- raw JSON fallback when the model emits tool syntax in plain text
- persisted definition-of-done state under `.loader/dod/`
- persisted clarify briefs under `.loader/briefs/`
- persisted implementation and verification plans under `.loader/plans/`
- persisted conversation sessions under `.loader/sessions/` plus active session state under `.loader/state/`
- persisted permission policy metadata alongside session state, so `loader status` / `loader session list` / `loader session show` can explain the effective policy that ran
- `loader --resume` and `loader --resume <session-id>` restore persisted session state
- durable project memory in `.loader/project-memory.json` and working notes in `.loader/notepad.md`
- native memory tools for `project_memory_*` and `notepad_*`
- scored workflow routing across `clarify` → `plan` → `execute` → `verify`, with route scores, runner-up pressure, unresolved-question carry-forward, and scheduled-next-mode hints
- typed workflow-signal extraction with persisted `signal_summary` context for route pressure, recent workflow history, and unresolved questions
- mode-specific system prompts for clarify, plan, execute, and verify
- intent-aware bounded clarify follow-through with explicit focus slots and persisted unresolved-question carry-forward
- pressure-pass clarify reviews with explicit readiness gates, challenged-assumption/tradeoff/example pressure kinds, and persisted clarify-pressure metadata
- codebase-backed clarify grounding with workspace evidence, repo facts, slot-aware evidence selection, pressure-aware evidence selection, and grounded brief hints for persisted clarify artifacts
- semantic artifact invalidation that can choose targeted plan refresh, clarify reentry, or full re-plan before execution continues
- structured workflow drift evidence covering confirmed touchpoints, inferred touchpoints, acceptance anchors, contradicted assumptions, verification contradictions, and task-boundary drift
- persisted workflow timeline entries for routes, handoffs, reentries, clarify outcomes, plan refreshes, and verify skips
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
- typed prompt construction in `runtime.prompting`, with explicit dynamic sections, a static/dynamic boundary marker, and persisted prompt-format / prompt-section metadata in session state
- validated turn-state transitions (`prepare`, `assistant`, `repair`, `tools`, `critique`, `completion`, `finalize`) with typed transition metadata, persisted session state, and emitted runtime events
- typed workflow-decision metadata persisted in session/runtime state, including reason codes, summaries, decision kind, workflow scores, and scheduled-next-mode hints
- `loader doctor` for backend, capability, workspace, command, state, and permission health checks outside the main runtime loop
- `loader status` plus `loader session list/show/resume` for inspecting persisted runtime state without invoking the LLM
- `loader prompt show [task]` for previewing the current prompt contract, workflow mode, permission mode, dynamic sections, and prompt body without a live model request
- `loader workflow show [session-id]` with `--mode`, `--kind`, and `--limit` filters plus operator-focused workflow highlights, and recent timeline snippets in `loader session show`
- `loader explore <prompt>` as a one-shot read-only lookup lane with its own prompt, constrained registry, and no DoD or workflow routing
- CLI and TUI status surfaces for model, capability profile, mode, workflow mode, workflow reason, last transition summary, permission mode, explicit turn phase, prompt format/sections, DoD phase, pending items, last verification result, and active session id
- CLI and TUI workflow-mode visibility plus artifact notifications
- CLI and TUI permission-mode visibility with color-coded status
- workspace-bound file operations with canonicalized boundary checks, binary detection, size limits, and structured patch metadata
- shell mutability classification plus structured truncation and stderr/exit-code metadata
- richer structured `AskUserQuestion` prompts with titles, context, options, and optional freeform responses
- assistant-turn request handling now lives in `runtime.assistant_turns`, clarify/plan lane execution now lives in `runtime.workflow_lanes`, tool-batch execution/recovery now lives in `runtime.tool_batches`, DoD/finalization logic now lives in `runtime.finalization`, workflow-state/session mutation lives in `runtime.workflow_state`, and the main loop now runs through `runtime.turn_preparation`, `runtime.turn_preamble`, `runtime.turn_iteration`, and `runtime.turn_loop` instead of accumulating further inside `conversation.py`
- `src/loader/runtime/conversation.py` now acts as a compact coordinator over dedicated runtime controllers rather than owning a monolithic turn loop

## Known weak spots

- [`src/loader/runtime/conversation.py`](../src/loader/runtime/conversation.py) is now much smaller and controller-based, but [`src/loader/runtime/turn_iteration.py`](../src/loader/runtime/turn_iteration.py) remains a relatively heavy seam with repair, tool-routing, and completion policy that is still more heuristic-heavy than the reference runtime in `refs/claw-code`
- planning, decomposition, and several helper behaviors still live in [`src/loader/agent/loop.py`](../src/loader/agent/loop.py), so ownership is cleaner than Sprint 00 but not fully simplified yet
- the workflow policy now consumes typed signals, but signal extraction is still heuristic and hand-tuned; Loader does not yet implement OMX's deeper ambiguity analysis, richer pressure-pass discipline, or branch-specific policy depth
- clarify is now intent-aware, pressure-aware, and codebase-grounded, but it is still much shallower than OMX's deep-interview behavior and does not adapt its budget or questioning style by task class
- plan freshness now handles broader semantic invalidation with typed evidence, but it is still lightweight and runtime-authored; Loader does not yet reason deeply over richer artifact metadata, contradicting verification evidence, or larger task reframes
- plan mode is still a single-pass artifact generator, not a Planner/Architect/Critic consensus loop
- DoD acceptance criteria and pending items are stronger than Sprint 02, but todo progress is still lightly structured compared with claw-code's richer workflow state
- evidence summaries are deterministic runtime summaries of captured output, not model-written verification narratives
- session compaction summaries are heuristic runtime summaries, not model-assisted continuity artifacts
- project-memory capture on finalized DoD evidence is still lightweight and command-summary oriented, not semantically curated memory extraction
- rule syntax is intentionally narrow and workspace-local; Loader still does not have claw-code's richer rule model or broader prompt/allow operator surface
- policy state is inspectable in doctor/status/session surfaces and dry-runnable through `loader permissions show/check`, but there is not yet a richer UX for editing, previewing multiple candidate rule sets, or temporarily overriding rules from the product surface
- prompt assembly is now typed and previewable, but Loader still does not expose prompt diffs, prompt snapshots over multiple candidate configurations, or a richer prompt-contract parity harness beyond the current unit coverage
- workflow history is now filterable and more operator-friendly through `loader workflow show`, but the product still does not offer timeline diffing, richer artifact previews, or a visual workflow trace
- shell safety is still heuristic and command-based; Loader does not yet have a richer shell sandbox or argument-aware mutability model
- explore mode is a one-shot read-only lane, not yet a richer interactive inspection workflow with deeper repo navigation affordances
- the read-only `git` helper is intentionally narrow compared with claw-code and OMX's broader repo/product surfaces, and the `patch` tool still stops short of AST/LSP-aware editing

## Out of scope in the current baseline

- richer permission-rule UX / per-command allowlists
- multi-agent / team orchestration

## Deterministic parity scenarios

The auditable manifest lives at [`tests/fixtures/runtime_parity_manifest.json`](../tests/fixtures/runtime_parity_manifest.json) and is exercised by [`tests/test_runtime_harness.py`](../tests/test_runtime_harness.py). Sprint 04 adds focused workflow integration coverage in [`tests/test_workflow_runtime.py`](../tests/test_workflow_runtime.py) and artifact/router unit coverage in [`tests/test_workflow.py`](../tests/test_workflow.py). Sprint 06 adds inspection/explore coverage in [`tests/test_inspection.py`](../tests/test_inspection.py), [`tests/test_explore_runtime.py`](../tests/test_explore_runtime.py), and [`tests/test_expanded_tools.py`](../tests/test_expanded_tools.py). Sprint 10 extends that workflow coverage in [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py), [`tests/test_workflow_runtime.py`](../tests/test_workflow_runtime.py), and [`tests/test_inspection.py`](../tests/test_inspection.py) for scored routing, clarify-budget behavior, plan refresh, and workflow timeline inspection. Sprint 11 adds [`tests/test_workflow_signals.py`](../tests/test_workflow_signals.py), [`tests/test_clarify_strategy.py`](../tests/test_clarify_strategy.py), [`tests/test_artifact_invalidation.py`](../tests/test_artifact_invalidation.py), and expanded inspection/runtime coverage for signal summaries, intent-aware clarify, semantic replan recovery, and workflow timeline filtering/highlights. Sprint 12 adds [`tests/test_clarify_grounding.py`](../tests/test_clarify_grounding.py), [`tests/test_turn_preparation.py`](../tests/test_turn_preparation.py), [`tests/test_turn_completion.py`](../tests/test_turn_completion.py), [`tests/test_turn_iteration.py`](../tests/test_turn_iteration.py), [`tests/test_turn_preamble.py`](../tests/test_turn_preamble.py), [`tests/test_workflow_state.py`](../tests/test_workflow_state.py), and [`tests/test_turn_loop.py`](../tests/test_turn_loop.py) for grounded clarify, structured recovery evidence, and the controllerized turn runtime.

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
- `explore_mode_ignores_global_allow_policy`: green

## Verification snapshot

As of 2026-04-07:

- `uv run pytest -q`: 231 passed
- `tests/test_runtime_harness.py` is fully green, including permission-mode parity, DoD verify/fix coverage, workflow routing parity, and the original contract regression
- `tests/test_prompt_builder.py` covers section rendering, native-vs-ReAct formatting, and prompt metadata persistence
- `tests/test_turn_state_machine.py` covers allowed/disallowed turn transitions and terminal transition metadata
- `tests/test_runtime_phases.py` covers repair/completion phase transitions plus persisted transition metadata in runtime events and session state
- `tests/test_dod.py` covers persistence, sizing boundaries, and verification command derivation
- `tests/test_workflow.py` covers workflow artifact round trips, scored-router expectations, DoD workflow links, and todo-to-DoD syncing
- `tests/test_workflow_signals.py` covers typed signal extraction, recent timeline pressure, and persisted `signal_summary` context
- `tests/test_clarify_strategy.py` covers clarify-slot prioritization and targeted follow-up question selection
- `tests/test_clarify_grounding.py` covers workspace evidence extraction, slot-aware/pressure-aware clarify grounding, and grounded clarify-brief hints
- `tests/test_workflow_policy.py` covers score breakdowns, clarify follow-up reviews, signal-summary persistence, artifact-freshness metadata, and workflow timeline serialization
- `tests/test_artifact_invalidation.py` covers semantic invalidation triggers plus targeted recovery selection for plan refresh vs full re-plan
- `tests/test_workflow_runtime.py` covers clarify routing, intent-aware clarify continuation, plan routing, targeted plan refresh, full re-plan through clarify reentry, verify-fix workflow handoff, and persisted workflow-decision metadata
- `tests/test_turn_preparation.py`, `tests/test_turn_completion.py`, `tests/test_turn_iteration.py`, `tests/test_turn_preamble.py`, `tests/test_workflow_state.py`, and `tests/test_turn_loop.py` cover the controllerized turn-runtime seams directly instead of relying only on large end-to-end runtime tests
- `tests/test_workflow_tools.py` and `tests/test_workflow_runtime_tools.py` cover `TodoWrite`, `AskUserQuestion`, and runtime callback plumbing
- `tests/test_session_state.py` covers session persistence, resume, rotation, compaction persistence, cumulative usage rollups, and persisted permission-policy metadata
- `tests/test_compaction.py` covers claw-style line compression and compacted continuation-message behavior
- `tests/test_memory_tools.py` covers project-memory writes, notepad writes, lifecycle-hook mirroring, and DoD-summary capture into project memory
- `tests/test_cli_resume.py` covers `--resume` argument rewriting for latest and named-session restore
- `tests/test_inspection.py` covers `loader doctor`, `loader status`, `loader session list/show`, `loader permissions show/check`, `loader prompt show`, `loader workflow show`, workflow timeline filtering/highlights, and workflow inspection surfaces
- `tests/test_explore_runtime.py` covers the direct explore lane contract and forced read-only behavior outside the parity harness
- `tests/test_expanded_tools.py` covers structured patch application, read-only git helpers, `notepad_append`, and richer structured user questions
- `tests/test_permissions.py` covers prompt/allow mode parsing, rule precedence, policy-backed prompting behavior, and hook lifecycle ordering
- `tests/test_tool_safety.py` covers workspace boundaries, binary/oversize guards, patch metadata, and shell truncation/classification
- `tests/test_status_surfaces.py` covers the CLI/TUI DoD, workflow-mode, permission-mode, capability-profile, and session-id formatting helpers
- native and extracted tool calls now record the same executor trace events, with source-specific metadata
- turn startup can refine backend capability profiles before the first request, `run_streaming()` delegates into the main runtime path, mutating tasks route through persisted evidence-backed completion, workflow artifacts survive across turns, sessions compact safely, explore queries bypass DoD/router overhead safely, policy rules are enforced deterministically, operators can inspect/dry-run policy decisions without live turns, prompt construction is sectioned and persisted, explicit turn phases are visible while a turn runs, session inspection preserves effective policy state, typed workflow signals now feed routing directly, semantic invalidation can force targeted refresh vs full re-plan, brownfield clarify can ask evidence-backed questions from repo facts, and the turn runtime now hangs off dedicated controllers instead of a single conversation-loop monolith

## Definition of honesty

- If a scenario is green here, it should have deterministic automated coverage.
- If a scenario is flaky or broken, it should be called out here before we claim parity work is done.
- Sprint 01 turned the original `tool_call_id` regression green by fixing the message contract, not by weakening the test.
- Sprint 02 replaced "looks done" completion for mutating tasks with a real verify/fix gate, but it has not yet reached the richer workflow contracts described in the report and Sprint 04+.
- Sprint 03 established permission modes, hooks, and tool hardening, but it intentionally stops short of claw-code's fuller rule engine and prompt/allow permission variants.
- Sprint 04 adds routing, artifacts, and structured user questions, but it is still a first-pass workflow layer rather than full OMX consensus planning or deep interview rigor.
- Sprint 05 adds durable sessions, resume, compaction, and native memory/notepad tools, but it stops short of Sprint 06's inspectable session/status product surfaces and still uses heuristic continuity summaries rather than richer semantic memory extraction.
- Sprint 06 adds inspectable product surfaces, a constrained explore lane, and a broader tool registry, but it still stops short of interactive explore workflows, richer git ergonomics, AST/LSP-aware editing, or any multi-agent/team runtime.
- Sprint 07 is complete: Loader now has prompt/allow modes, rule-based permission policy, policy-backed prompting, persisted policy inspection state, and smaller assistant-turn/tool-batch/finalization runtime seams, but it still stops short of a richer rule UX, deeper policy sandboxing, and the more opinionated workflow/runtime contracts in the refs.
- Sprint 08 is complete: Loader now has a typed prompt builder, explicit runtime turn phases, first-class `loader permissions show/check` operator surfaces, and more coherent prompt/policy observability in doctor/status/session output, but it still stops short of a richer rule editor, formal state-machine routing, prompt-preview tooling, or the deeper workflow rigor in the refs.
- Sprint 09 is complete: Loader now has a validated turn state machine, typed persisted workflow decisions, `loader prompt show`, and richer workflow-reason/transition inspection surfaces, but it still stops short of deeper workflow routing policy, prompt diffing/versioning, and the more opinionated planning discipline used by the refs.
- Sprint 10 is complete: Loader now has a scored workflow policy, bounded clarify follow-through, targeted plan-refresh discipline, persisted workflow timeline inspection, and a greener workflow test contract, but it still stops short of deeper semantic routing/replanning, adaptive clarify depth, prompt-history diffing, or the fuller workflow rigor used by the refs.
- Sprint 11 is complete: Loader now has typed workflow signals, intent-aware clarify slots, semantic invalidation with targeted recovery vs full re-plan, filtered workflow inspection, and dedicated clarify/plan lane execution in `runtime.workflow_lanes`, but it still stops short of OMX-style pressure-pass interviewing, richer semantic artifact reasoning, and the deeper end-to-end orchestration discipline in the refs.
- Sprint 12 is complete: Loader now has pressure-pass clarify behavior, codebase-backed clarify grounding, structured recovery evidence, controllerized turn-runtime seams, and a genuinely coordinator-shaped `runtime.conversation`, but it still stops short of OMX-style deep interview depth, richer semantic artifact reasoning, artifact/prompt diff ergonomics, and the broader operator/runtime sophistication in the refs.
