# Sprint 08: Prompt Builder, Runtime Phases, and Permission Operator UX

## Prerequisites

Sprint 07

## Goals

Turn the remaining "smart heuristics" into explicit runtime contracts and make Loader's permission system operable without reading JSON files by hand.

Sprint 07 gave Loader a real execution-policy layer and smaller runtime seams. The next leverage point is to stop letting prompt assembly and response repair live as ad hoc strings and inline heuristics. `claw-code` keeps the runtime tighter partly because prompt construction is its own subsystem, and OMX keeps workflows legible because phase/state is explicit instead of implied.

This sprint is the bridge from "better runtime internals" to "better operator control":

- prompt construction becomes a typed builder with explicit sections
- turn phases become named runtime states instead of inline branches
- permission policy becomes dry-runnable and explainable from the CLI

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/rust/crates/runtime/src/permissions.rs`
- `refs/claw-code/rust/crates/runtime/src/conversation.rs:295-470`
- `refs/oh-my-codex/src/modes/base.ts`

## Deliverables

### 1. Typed prompt builder instead of hand-built templates

Loader's prompt layer is still mostly string templates in `src/loader/agent/prompts.py`. Sprint 08 should turn that into a runtime prompt builder with explicit sections and clearer ownership.

Implementation targets:

- replace the current monolithic prompt strings with a typed builder under `src/loader/runtime/` or another clearly-owned prompt module
- separate static scaffolding from dynamic runtime context with an explicit boundary, following the shape of `claw-code`'s prompt builder
- render mode guidance, project context, runtime config, permission mode, and relevant workflow/artifact context as independent sections instead of one merged string blob
- keep native-tool vs ReAct prompt differences as a thin formatting concern, not two largely separate prompt bodies
- make it easy to add or remove one section without rewriting the whole prompt template
- preserve Loader's current local-first/product-specific guidance; this sprint is about structure first, not prompt verbosity for its own sake

The goal is not "more prompt text." The goal is a prompt contract that is inspectable, composable, and easier to evolve without reopening the turn loop.

### 2. Explicit runtime phases for response repair and completion behavior

`src/loader/runtime/conversation.py` is healthier after Sprint 07, but it still owns too many inline heuristics:

- empty-output retries
- raw-tool-call fallback
- fake-tool narration correction
- self-critique rerouting
- completion nudges for non-mutating tasks
- text-loop bailout behavior

Sprint 08 should move those into named runtime phases and focused helpers.

Implementation targets:

- define a typed turn-phase model for the remaining coordinator behaviors, for example:
  - `prepare`
  - `assistant`
  - `tools`
  - `repair`
  - `critique`
  - `completion`
- extract response-repair and completion-policy decisions into dedicated runtime components rather than leaving them inline in `ConversationRuntime.run_turn(...)`
- keep `conversation.py` as the turn coordinator that advances phase state and delegates to helpers
- persist enough phase metadata in trace/session state that status surfaces and debugging can explain where Loader spent time during a turn
- avoid moving these heuristics back into `agent/loop.py`; the runtime split should keep going in one direction

This is the next step toward a runtime that is easier to debug and less vulnerable to regressions when we tune follow-through behavior.

### 3. First-class permission operator surfaces

Sprint 07 made policy inspectable. Sprint 08 should make it operable.

Implementation targets:

- add `loader permissions show` to display:
  - active permission mode
  - prompting state
  - rule source path
  - normalized allow/deny/ask rules
  - rule counts and validity
- add `loader permissions check` to dry-run one hypothetical tool request and show:
  - tool name
  - normalized input summary
  - required mode
  - policy decision (`allow` / `deny` / `ask`)
  - matched rule or hook-style reason when present
- support both structured JSON-like tool arguments and simple string inputs where practical
- wire `loader doctor` remediation hints to these policy commands when rules are invalid or confusing
- keep the UX read-mostly in this sprint; authoring/edit flows can remain file-based for now

This gives operators a way to answer "why did Loader allow/prompt/deny this?" without reproducing the behavior in a live turn.

### 4. Prompt and policy state surfaced coherently in product surfaces

Once prompt construction and phase state are explicit, the product surfaces should expose enough context to make Loader's behavior legible.

Implementation targets:

- extend `loader status` and the TUI status line to surface the active turn phase when a run is in progress
- make doctor/status/session output consistent about permission terminology (`mode`, `prompting`, `rules`, `source`)
- expose prompt-builder metadata in a minimal, operator-friendly way:
  - current mode
  - whether ReAct/native formatting is active
  - which dynamic sections were included
- keep these surfaces concise; this sprint is about observability, not dumping the whole prompt to the screen by default

The goal is to make Loader easier to reason about in live use, not just in code review.

## Testing strategy

- unit coverage for:
  - prompt-builder section rendering
  - native vs ReAct prompt-format differences over the same section set
  - turn-phase transitions for repair/critique/completion flows
  - permission dry-run explanations and invalid-input handling
- CLI coverage for:
  - `loader permissions show`
  - `loader permissions check`
  - doctor remediation output that points users toward policy inspection
- deterministic/runtime coverage for:
  - empty assistant output triggering repair-phase retries without regressing the tool path
  - raw-tool fallback still sharing the same executor and phase bookkeeping
  - non-mutating completion nudges remaining deterministic after the phase split
  - Sprint 00-07 parity scenarios staying green
- status/TUI coverage for:
  - active phase rendering
  - coherent permission terminology across doctor/status/session output

## Definition of done

- prompt construction is builder-based, sectioned, and easier to inspect than the current template strings
- `conversation.py` is slimmer again, with response-repair and completion heuristics moved into focused runtime components
- Loader exposes first-class permission inspection and dry-run commands instead of requiring manual JSON reading
- prompt, phase, and policy state are more legible in status/inspection surfaces
- the full parity baseline remains green after the phase split
- Loader moves closer to claw-code's "tight runtime, explicit prompt contract" shape without overcommitting to a huge configuration system

## Explicitly out of scope

- a full interactive rule editor or TUI-based permission authoring flow
- AST-aware, LSP-aware, or symbol-aware editing
- a richer shell sandbox than the current command-based model
- interactive multi-step explore workflows
- multi-agent or team orchestration

## Audit

### Landed

- Loader's prompt construction now lives in `src/loader/runtime/prompting.py` as a typed builder with explicit sections, a static/dynamic boundary marker, and thin native-vs-ReAct formatting differences instead of one mostly hand-built string blob
- prompt metadata now persists in session state, so `loader status`, `loader session list/show`, and the live agent state can explain the active prompt format and which dynamic sections were actually included for the current workspace/task
- the remaining coordinator heuristics are now split into explicit runtime components: phase tracking in `runtime.phases`, response repair in `runtime.repair`, and completion/self-critique policy in `runtime.completion_policy`
- `ConversationRuntime.run_turn(...)` now advances explicit turn phases (`prepare`, `assistant`, `repair`, `tools`, `critique`, `completion`, `finalize`) and persists the active phase into session state while also emitting runtime events for the CLI/TUI
- the TUI status line and CLI/session inspection surfaces now expose the active turn phase while a turn is in flight, which makes Loader's mid-turn behavior much easier to debug than the earlier implicit branch structure
- Loader now has first-class permission operator commands:
  - `loader permissions show` displays the active mode, prompting state, rules source, validity, counts, and normalized allow/deny/ask rules
  - `loader permissions check` dry-runs one hypothetical tool request and reports the normalized input summary, required mode, allow/deny/ask decision, matched rule, and policy reason
- `loader permissions check` supports both JSON object arguments and practical positional input mapping for common tools such as `bash`, `read`, `write`, `edit`, `patch`, `glob`, `grep`, `git`, and read-only memory/notepad lookups
- `loader doctor` remediation now points operators toward `loader permissions show` / `loader permissions check` instead of leaving permission debugging as a code/JSON-reading exercise
- doctor/status/session output now uses more consistent permission terminology around mode, prompting, rules, and source instead of mixing several labels for the same policy concepts

### Verification

- `uv run pytest -q` is green: `176 passed`
- `tests/test_prompt_builder.py` covers section rendering, native-vs-ReAct formatting, and prompt-builder persistence metadata
- `tests/test_runtime_phases.py` covers repair/completion phase transitions and active phase bookkeeping
- `tests/test_inspection.py` now covers `loader permissions show`, `loader permissions check`, invalid JSON input handling, invalid-rule visibility, prompt/policy metadata in status/session surfaces, and the existing doctor/session inspection behavior
- targeted `ruff` checks are green for `src/loader/runtime/inspection.py`, `tests/test_inspection.py`, and import ordering in `src/loader/cli/main.py`
- the full Sprint 00-07 parity baseline stayed green through the prompt/phase split and permission CLI rollout

### Residual debt

- `src/loader/runtime/conversation.py` is slimmer than before Sprint 08, but it still coordinates workflow routing and phase transitions with a heuristic branch structure rather than a more formal state machine
- prompt construction is now inspectable and sectioned, but Loader still does not offer prompt previews/diffs, a richer prompt-contract parity harness, or operator controls for temporarily adjusting prompt sections
- `loader permissions show/check` make the policy operable, but authoring/editing rules is still file-based and there is still no first-class preview UX for comparing multiple rule sets or applying temporary session overrides
- doctor/status/session terminology is more coherent now, but the product still stops short of the richer policy UX and sandbox semantics used by the references
- the explicit turn phases improve observability, but they are still runtime bookkeeping around heuristics, not yet a deeper workflow-state contract on the level of OMX's more opinionated routing discipline
