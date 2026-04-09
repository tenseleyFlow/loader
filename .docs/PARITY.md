# Loader Runtime Parity Checkpoint

Date: 2026-04-08

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
- raw JSON fallback now routes through the runtime parser plus the active registry, including modern workflow tools such as `TodoWrite` and `AskUserQuestion`
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
- persisted workflow ledger state for assumptions, contradicted assumptions, acceptance anchors, and open/closed decision boundaries, threaded through clarify, plan, recovery, and inspection
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
- persisted prompt snapshot history in session state so prompt-contract changes survive resume and later inspection
- validated turn-state transitions (`prepare`, `assistant`, `repair`, `tools`, `critique`, `completion`, `finalize`) with typed transition metadata, persisted session state, and emitted runtime events
- typed workflow-decision metadata persisted in session/runtime state, including reason codes, summaries, decision kind, workflow scores, and scheduled-next-mode hints
- `loader doctor` for backend, capability, workspace, command, state, and permission health checks outside the main runtime loop
- `loader status` plus `loader session list/show/resume` for inspecting persisted runtime state without invoking the LLM
- `loader prompt show [task]` for previewing the current prompt contract, workflow mode, permission mode, dynamic sections, and prompt body without a live model request
- `loader prompt diff [session-id]` for comparing persisted prompt contracts, with concise summaries by default and unified diffs on demand
- `loader workflow show [session-id]` with `--mode`, `--kind`, and `--limit` filters plus operator-focused workflow highlights, recent timeline snippets in `loader session show`, and `--diff` / `--full-diff` artifact comparison for persisted workflow artifacts
- `loader explore <prompt>` as a read-only lookup lane with its own prompt, constrained registry, persisted bounded continuity under `.loader/state/explore.json`, `--fresh` to ignore prior explore history when needed, and `--status` / `--reset` for continuity inspection and reset
- `RuntimeContext` is now the primary runtime seam for workflow state, turn phases, response repair, no-tool completion, response routing, turn looping, finalization, workflow lanes, and workflow recovery; the older `RuntimeLegacyServices` shim has been removed
- shared runtime bootstrap through `runtime.bootstrap.build_runtime_context(...)` / `sync_runtime_context(...)`, with both conversation and explore runtimes now starting from an explicit `RuntimeBootstrapView` rather than a raw `Agent` object by default
- runtime-owned safeguard and reasoning helpers now have canonical homes under `src/loader/runtime/`; `src/loader/agent/safeguards.py` and `src/loader/agent/reasoning.py` are compatibility-export layers rather than the primary implementations
- the public launcher contract now owns conversational routing, decomposition entry routing, direct turn routing, and explore launch through `src/loader/runtime/launcher.py`, which leaves a smaller and more honest `src/loader/agent/loop.py`
- runtime-owned prompt/session shell helpers now live in `src/loader/runtime/public_shell.py`, including prompt construction, prompt snapshot persistence, session creation, session restore, and few-shot example selection
- compatibility exports are now explicitly bounded by direct tests, and internal runtime code is guarded against drifting back to `agent/reasoning.py` / `agent/safeguards.py` imports
- CLI and TUI status surfaces for model, capability profile, mode, workflow mode, workflow reason, last transition summary, permission mode, explicit turn phase, prompt format/sections, DoD phase, pending items, last verification result, and active session id
- CLI status now also surfaces recent explore activity, including bounded explore turn/message counts, the last explore history mode, and the last explore query
- CLI and TUI workflow-mode visibility plus artifact notifications
- CLI and TUI permission-mode visibility with color-coded status
- workspace-bound file operations with canonicalized boundary checks, binary detection, size limits, and structured patch metadata
- shell mutability classification plus structured truncation and stderr/exit-code metadata
- richer structured `AskUserQuestion` prompts with titles, context, options, and optional freeform responses
- honest repair/completion behavior for no-tool turns: empty assistant replies get a single explicit retry, and Loader no longer relies on synthetic prefill, fake-tool scolding reroutes, or self-critique puppeting for plain-text answers
- raw-text tool recovery now also fails honestly once its budget is exhausted instead of adding a synthetic follow-up invitation
- dedicated assistant-response routing in `runtime.response_routing`, so final-answer, tool-batch, and no-tool completion dispatch no longer live inline inside `turn_iteration.py`
- assistant-turn request handling now lives in `runtime.assistant_turns`, clarify/plan lane execution now lives in `runtime.workflow_lanes`, tool-batch execution/recovery now lives in `runtime.tool_batches`, DoD/finalization logic now lives in `runtime.finalization`, workflow-state/session mutation lives in `runtime.workflow_state`, and the main loop now runs through `runtime.turn_preparation`, `runtime.turn_preamble`, `runtime.turn_iteration`, and `runtime.turn_loop` instead of accumulating further inside `conversation.py`
- `src/loader/runtime/conversation.py` now acts as a compact coordinator over dedicated runtime controllers rather than owning a monolithic turn loop
- persisted completion-decision summaries plus bounded completion traces are now available in session/runtime state, status/session inspection, and resume flows
- typed follow-through evidence now backs non-mutating completion checks, with explicit required/missing evidence and honest terminal failure once the continuation budget is exhausted without enough proof of completion
- runtime-owned public-shell helpers now cover prompt/session factories, session install/load helpers, steering mailbox behavior, sync/async event wrapping, and capability-refresh decision helpers
- `loader workflow show --policy` now filters directly to unified repair / verify-skip / completion accountability events, and `loader session show` now includes a `Policy Timeline` preview so operators can inspect the stop/continue/retry story without stitching together separate surfaces by hand

## Known weak spots

- the public runtime boundary is now explicit and runtime-shaped, but `Agent` still constructs and supplies that boundary; Loader has not yet decided whether the current 292-line shell is the stable long-term seam or just the next contraction checkpoint
- [`src/loader/agent/loop.py`](../src/loader/agent/loop.py) is down to 292 lines and much closer to a public facade than the pre-Sprint-15 shell, but it still owns the public compatibility shell and remaining launcher/UI glue instead of collapsing fully to a minimal shell
- [`src/loader/agent/reasoning.py`](../src/loader/agent/reasoning.py) and [`src/loader/agent/safeguards.py`](../src/loader/agent/safeguards.py) are now compatibility shims rather than primary implementations, but they still remain as export layers until Loader narrows its external compatibility surface further
- [`src/loader/runtime/tool_batches.py`](../src/loader/runtime/tool_batches.py) and parts of [`src/loader/runtime/workflow_lanes.py`](../src/loader/runtime/workflow_lanes.py) are narrower and more directly tested than before, but they still carry more heuristic policy than the tightest reference seams in `refs/claw-code`
- the workflow policy now consumes typed signals, but signal extraction is still heuristic and hand-tuned; Loader does not yet implement OMX's deeper ambiguity analysis, richer pressure-pass discipline, or branch-specific policy depth
- clarify is now intent-aware, pressure-aware, and codebase-grounded, but it is still much shallower than OMX's deep-interview behavior and does not adapt its budget or questioning style by task class
- plan freshness now handles broader semantic invalidation with typed evidence, but it is still lightweight and runtime-authored; Loader does not yet reason deeply over richer artifact metadata, contradicting verification evidence, or larger task reframes
- the workflow ledger is now explicit and persisted, but it is still a pragmatic text-first contract rather than deeper symbolic task/state reasoning with stronger provenance
- plan mode is still a single-pass artifact generator, not a Planner/Architect/Critic consensus loop
- DoD acceptance criteria and pending items are stronger than Sprint 02, but todo progress is still lightly structured compared with claw-code's richer workflow state
- evidence summaries are deterministic runtime summaries of captured output, not model-written verification narratives
- session compaction summaries are heuristic runtime summaries, not model-assisted continuity artifacts
- project-memory capture on finalized DoD evidence is still lightweight and command-summary oriented, not semantically curated memory extraction
- rule syntax is intentionally narrow and workspace-local; Loader still does not have claw-code's richer rule model or broader prompt/allow operator surface
- policy state is inspectable in doctor/status/session surfaces and dry-runnable through `loader permissions show/check`, but there is not yet a richer UX for editing, previewing multiple candidate rule sets, or temporarily overriding rules from the product surface
- follow-through evidence is now explicit and persisted, but it is still heuristic and runtime-authored rather than backed by a deeper verifier/model contract or richer artifact-derived proof model
- prompt assembly is now typed, previewable, and diffable across persisted sessions, but Loader still does not compare multiple candidate prompt contracts before execution or enforce a richer prompt-contract parity harness beyond the current unit and inspection coverage
- workflow history is now filterable, ledger-backed, and diffable for persisted artifacts, but it is still text-first; Loader still does not offer semantic/AST-aware artifact diffs, richer artifact preview UX, or a visual workflow trace
- shell safety is still heuristic and command-based; Loader does not yet have a richer shell sandbox or argument-aware mutability model
- explore mode now has lightweight transcript continuity plus `loader explore --status` / `--reset`, but it is still a narrow read-only lookup lane rather than a richer interactive inspection workflow with deeper repo navigation affordances
- the read-only `git` helper is intentionally narrow compared with claw-code and OMX's broader repo/product surfaces, and the `patch` tool still stops short of AST/LSP-aware editing

## Out of scope in the current baseline

- richer permission-rule UX / per-command allowlists
- multi-agent / team orchestration

## Deterministic parity scenarios

The auditable manifest lives at [`tests/fixtures/runtime_parity_manifest.json`](../tests/fixtures/runtime_parity_manifest.json) and is exercised by [`tests/test_runtime_harness.py`](../tests/test_runtime_harness.py). Sprint 04 adds focused workflow integration coverage in [`tests/test_workflow_runtime.py`](../tests/test_workflow_runtime.py) and artifact/router unit coverage in [`tests/test_workflow.py`](../tests/test_workflow.py). Sprint 06 adds inspection/explore coverage in [`tests/test_inspection.py`](../tests/test_inspection.py), [`tests/test_explore_runtime.py`](../tests/test_explore_runtime.py), and [`tests/test_expanded_tools.py`](../tests/test_expanded_tools.py). Sprint 10 extends that workflow coverage in [`tests/test_workflow_policy.py`](../tests/test_workflow_policy.py), [`tests/test_workflow_runtime.py`](../tests/test_workflow_runtime.py), and [`tests/test_inspection.py`](../tests/test_inspection.py) for scored routing, clarify-budget behavior, plan refresh, and workflow timeline inspection. Sprint 11 adds [`tests/test_workflow_signals.py`](../tests/test_workflow_signals.py), [`tests/test_clarify_strategy.py`](../tests/test_clarify_strategy.py), [`tests/test_artifact_invalidation.py`](../tests/test_artifact_invalidation.py), and expanded inspection/runtime coverage for signal summaries, intent-aware clarify, semantic replan recovery, and workflow timeline filtering/highlights. Sprint 12 adds [`tests/test_clarify_grounding.py`](../tests/test_clarify_grounding.py), [`tests/test_turn_preparation.py`](../tests/test_turn_preparation.py), [`tests/test_turn_completion.py`](../tests/test_turn_completion.py), [`tests/test_turn_iteration.py`](../tests/test_turn_iteration.py), [`tests/test_turn_preamble.py`](../tests/test_turn_preamble.py), [`tests/test_workflow_state.py`](../tests/test_workflow_state.py), and [`tests/test_turn_loop.py`](../tests/test_turn_loop.py) for grounded clarify, structured recovery evidence, and the controllerized turn runtime. Sprint 13 adds [`tests/test_runtime_repair_flows.py`](../tests/test_runtime_repair_flows.py), [`tests/test_response_routing.py`](../tests/test_response_routing.py), [`tests/test_workflow_ledger.py`](../tests/test_workflow_ledger.py), and expanded [`tests/test_session_state.py`](../tests/test_session_state.py) / [`tests/test_inspection.py`](../tests/test_inspection.py) coverage for honest repair behavior, dedicated response routing, persisted semantic ledger state, prompt snapshot history, and prompt/artifact diff inspection. Sprint 15 adds [`tests/test_runtime_bootstrap.py`](../tests/test_runtime_bootstrap.py), [`tests/test_safeguard_services.py`](../tests/test_safeguard_services.py), [`tests/test_reasoning_compat.py`](../tests/test_reasoning_compat.py), and updated [`tests/test_runtime_context.py`](../tests/test_runtime_context.py) coverage for the shared bootstrap seam plus the runtime-owned safeguards/reasoning compatibility contract. Sprint 16 adds [`tests/test_runtime_launcher.py`](../tests/test_runtime_launcher.py), [`tests/test_chat_lane.py`](../tests/test_chat_lane.py), [`tests/test_decomposition_lane.py`](../tests/test_decomposition_lane.py), [`tests/test_compat_boundaries.py`](../tests/test_compat_boundaries.py), and expanded [`tests/test_explore_runtime.py`](../tests/test_explore_runtime.py) / [`tests/test_inspection.py`](../tests/test_inspection.py) coverage for the public launcher contract, compatibility boundaries, and persisted explore continuity. Sprint 17 adds [`tests/test_runtime_public_shell.py`](../tests/test_runtime_public_shell.py), expanded [`tests/test_runtime_bootstrap.py`](../tests/test_runtime_bootstrap.py) / [`tests/test_runtime_launcher.py`](../tests/test_runtime_launcher.py) / [`tests/test_runtime_context.py`](../tests/test_runtime_context.py) coverage for the explicit bootstrap view, expanded [`tests/test_repair.py`](../tests/test_repair.py) / [`tests/test_runtime_repair_flows.py`](../tests/test_runtime_repair_flows.py) coverage for honest raw-text recovery failure, and expanded [`tests/test_explore_runtime.py`](../tests/test_explore_runtime.py) / [`tests/test_inspection.py`](../tests/test_inspection.py) coverage for explore continuity inspection, reset, and persisted fresh-vs-continue visibility.

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

As of 2026-04-08:

- `uv run pytest -q`: 359 passed
- `tests/test_runtime_harness.py` is fully green, including permission-mode parity, DoD verify/fix coverage, workflow routing parity, and the original contract regression
- `tests/test_prompt_builder.py` covers section rendering, native-vs-ReAct formatting, and prompt metadata persistence
- `tests/test_turn_state_machine.py` covers allowed/disallowed turn transitions and terminal transition metadata
- `tests/test_runtime_phases.py` covers repair/completion phase transitions plus persisted transition metadata in runtime events and session state
- `tests/test_runtime_repair_flows.py` covers honest empty-response retries, no synthetic prefill on first turns, and the removal of the older no-tool puppeting/scolding reroutes
- `tests/test_runtime_context.py` and `tests/test_runtime_state_controllers.py` cover typed runtime-context construction plus direct workflow-state and phase-tracker behavior without relying on a full `Agent`
- `tests/test_runtime_bootstrap.py` covers the shared runtime bootstrap contract, prompt/capability synchronization, and direct conversation/explore construction through the runtime bootstrap seam
- `tests/test_runtime_public_shell.py` covers runtime-owned prompt/session shell helpers directly, including session creation metadata, prompt snapshot persistence, restored last-turn summary state, steering mailbox behavior, sync/async event emitter normalization, capability-refresh helper behavior, and fresh/load session-install helpers
- `tests/test_runtime_launcher.py`, `tests/test_chat_lane.py`, and `tests/test_decomposition_lane.py` cover the public launcher contract, conversational entry routing, decomposition entry routing, and direct runtime turn delegation
- `tests/test_safeguard_services.py` covers the canonical runtime safeguard implementation plus the compatibility-export path under `loader.agent.safeguards`
- `tests/test_reasoning_compat.py` covers runtime-owned deliberation/completion helpers plus the compatibility-export path under `loader.agent.reasoning`
- `tests/test_compat_boundaries.py` fails if internal Loader code drifts back to importing runtime-owned helpers through compatibility shims
- `tests/test_repair.py` covers raw-text fallback through the runtime parser and active registry, including `TodoWrite` recovery and honest failure once raw-tool recovery exceeds its budget
- `tests/test_completion_policy.py` covers direct text-loop bailout and continuation-prompt behavior on the typed runtime context
- `tests/test_completion_policy.py`, `tests/test_turn_completion.py`, and `tests/test_session_state.py` now also pin typed follow-through evidence, honest budget-exhausted completion finalization, and persisted completion-evidence summaries
- `tests/test_response_routing.py` covers direct final-answer routing and halted tool-batch routing at the new response-policy seam
- `tests/test_dod.py` covers persistence, sizing boundaries, and verification command derivation
- `tests/test_workflow.py` covers workflow artifact round trips, scored-router expectations, DoD workflow links, and todo-to-DoD syncing
- `tests/test_workflow_signals.py` covers typed signal extraction, recent timeline pressure, and persisted `signal_summary` context
- `tests/test_clarify_strategy.py` covers clarify-slot prioritization and targeted follow-up question selection
- `tests/test_clarify_grounding.py` covers workspace evidence extraction, slot-aware/pressure-aware clarify grounding, and grounded clarify-brief hints
- `tests/test_workflow_policy.py` covers score breakdowns, clarify follow-up reviews, signal-summary persistence, artifact-freshness metadata, and workflow timeline serialization
- `tests/test_artifact_invalidation.py` covers semantic invalidation triggers plus targeted recovery selection for plan refresh vs full re-plan
- `tests/test_workflow_ledger.py` covers ledger seeding, contradiction tracking, acceptance-anchor updates, and operator-facing highlight summaries
- `tests/test_workflow_runtime.py` covers clarify routing, intent-aware clarify continuation, plan routing, targeted plan refresh, full re-plan through clarify reentry, verify-fix workflow handoff, persisted workflow-decision metadata, and workflow-ledger updates through runtime recovery
- `tests/test_turn_preparation.py`, `tests/test_turn_completion.py`, `tests/test_turn_iteration.py`, `tests/test_turn_preamble.py`, `tests/test_workflow_state.py`, and `tests/test_turn_loop.py` cover the controllerized turn-runtime seams directly instead of relying only on large end-to-end runtime tests
- `tests/test_workflow_tools.py` and `tests/test_workflow_runtime_tools.py` cover `TodoWrite`, `AskUserQuestion`, and runtime callback plumbing
- `tests/test_session_state.py` covers session persistence, resume, rotation, compaction persistence, cumulative usage rollups, persisted permission-policy metadata, workflow-ledger state, and prompt snapshot history
- `tests/test_compaction.py` covers claw-style line compression and compacted continuation-message behavior
- `tests/test_memory_tools.py` covers project-memory writes, notepad writes, lifecycle-hook mirroring, and DoD-summary capture into project memory
- `tests/test_cli_resume.py` covers `--resume` argument rewriting for latest and named-session restore
- `tests/test_inspection.py` covers `loader doctor`, `loader status`, `loader session list/show`, `loader permissions show/check`, `loader prompt show`, `loader prompt diff`, `loader workflow show --diff`, workflow timeline filtering/highlights, the new `loader workflow show --policy` accountability filter, and workflow/session inspection surfaces, including recent explore activity plus `loader explore --status` / `--reset`
- `tests/test_explore_runtime.py` covers the direct explore lane contract, forced read-only behavior, persisted follow-up continuity, persisted `fresh` vs `continue` visibility, and `fresh` explore resets outside the parity harness
- `tests/test_expanded_tools.py` covers structured patch application, read-only git helpers, `notepad_append`, and richer structured user questions
- `tests/test_permissions.py` covers prompt/allow mode parsing, rule precedence, policy-backed prompting behavior, and hook lifecycle ordering
- `tests/test_tool_safety.py` covers workspace boundaries, binary/oversize guards, patch metadata, and shell truncation/classification
- `tests/test_status_surfaces.py` covers the CLI/TUI DoD, workflow-mode, permission-mode, capability-profile, and session-id formatting helpers
- native and extracted tool calls now record the same executor trace events, with source-specific metadata
- turn startup can refine backend capability profiles before the first request, `run_streaming()` delegates into the main runtime path, mutating tasks route through persisted evidence-backed completion, workflow artifacts and workflow-ledger state survive across turns, sessions compact safely, explore queries bypass DoD/router overhead safely, policy rules are enforced deterministically, operators can inspect/dry-run policy decisions without live turns, prompt construction is sectioned and persisted, prompt snapshots and artifact diffs are inspectable after the fact, explicit turn phases are visible while a turn runs, session inspection preserves effective policy state, typed workflow signals now feed routing directly, semantic invalidation can force targeted refresh vs full re-plan, brownfield clarify can ask evidence-backed questions from repo facts, and the turn runtime now avoids the older synthetic repair/no-tool puppeting while routing assistant outcomes through dedicated controllers instead of a single conversation-loop monolith

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
- Sprint 13 is complete: Loader now avoids synthetic prefill and no-tool puppeting, persists a semantic workflow ledger, exposes prompt/artifact diff surfaces, and routes assistant responses through a dedicated response-policy seam, but it still stops short of claw-code's narrower response/tool policy factoring, deeper planning rigor, and broader operator ergonomics.
- Sprint 14 is complete: Loader now treats `RuntimeContext` as the primary seam across workflow state, turn phases, response policy, turn looping, workflow recovery, and finalization; the old runtime legacy shim is gone, raw-text tool recovery no longer depends on hidden agent extractors, and the hot path is substantially more runtime-owned, but bootstrap ownership still begins at the agent wrapper and Loader still stops short of claw-code's fuller policy engine, OMX's deeper workflow rigor, and richer operator/runtime surfaces.
- Sprint 15 is complete: Loader now has a shared runtime bootstrap seam, runtime-owned safeguards and deliberation helpers, compatibility-only `agent/reasoning.py` and `agent/safeguards.py`, no remaining `Agent._build_runtime_context()` helper, and a much smaller `agent/loop.py` after dead planner/raw-extraction cleanup, but it still stops short of a minimal entrypoint shell, deeper explore ergonomics, claw-code's tighter policy engine, and OMX's richer planning/interview rigor.
- Sprint 16 is complete: Loader now has a first-class runtime launcher contract, a thinner `agent/loop.py`, explicit compatibility-boundary proof, and persisted explore continuity plus status visibility, but it still stops short of a fully minimal public shell, richer explore workflows, claw-code's tighter policy seams, and OMX's deeper planning/interview rigor.
- Sprint 17 is complete: Loader now starts public runtime launch from an explicit runtime-shaped bootstrap view, moves prompt/session shell helpers into `src/loader/runtime/public_shell.py`, fails raw-text tool recovery more honestly once its budget is exhausted, and exposes explore continuity through `loader explore --status` / `--reset`, but it still stops short of a fully minimal public shell, deeper completion-policy deletions, richer explore workflows, claw-code's tighter policy seams, and OMX's deeper planning/interview rigor.
- Sprint 18 is complete: Loader now persists explicit completion decisions and bounded completion traces, exposes them through status/session inspection, restores them across resume, and moves more event/steering/capability/session shell glue into `src/loader/runtime/public_shell.py`, but it still stops short of a fully minimal public shell, harder deletion of all continuation heuristics, claw-code's tighter policy seams, and OMX's deeper planning/interview rigor.
- Sprint 19 is complete: Loader now pushes more public entry glue under `src/loader/runtime/public_shell.py`, derives typed follow-through evidence for non-mutating completion checks, fails honestly when that evidence is still missing after the continuation budget is exhausted, and exposes a clearer unified policy story through `loader workflow show --policy` plus the `Policy Timeline` preview in `loader session show`, but it still stops short of a fully minimal public shell, claw-code's fuller policy engine, and OMX's deeper verifier/interview rigor.
