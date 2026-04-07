# Sprint 11: Semantic Signals, Clarify Strategy, and Orchestrator Split

## Prerequisites

Sprint 10

## Goals

Turn Loader's new workflow policy from a better scorecard into a more structured workflow contract, and keep shrinking the coordinator so policy and orchestration live in dedicated runtime seams instead of collecting back inside `conversation.py`.

Sprint 10 was a meaningful step forward. Loader now has scored routing, bounded clarify follow-through, plan refresh, and a persisted workflow timeline. That closes a real gap with claw-code and OMX, but the audit is honest about what still hurts:

- workflow scoring is still hand-tuned and text-heuristic rather than driven by a typed signal model
- clarify has follow-through now, but the questioning strategy is still generic and shallow compared with OMX's deep-interview discipline
- plan freshness is still mostly file-drift based instead of understanding broader semantic invalidation
- workflow history is inspectable, but not yet filtered or summarized around the most useful operator questions
- `conversation.py` is smaller than it was, but it still coordinates more workflow behavior than the refs

The next leverage point is to stop asking only "what pressure score won?" and start asking "what concrete workflow signals are in play, which task boundaries remain unresolved, and which orchestration module should own the next move?"

This sprint is about workflow structure:

- route policy consumes typed workflow signals rather than leaning so heavily on inline heuristics
- clarify becomes intent-aware instead of merely multi-round
- replan discipline becomes more semantic than touched-file drift alone
- workflow inspection becomes more useful for debugging why Loader stayed in or re-entered a lane
- `conversation.py` shrinks again because orchestration moves into dedicated runtime modules

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/modes/base.ts`
- `refs/oh-my-codex/skills/deep-interview/SKILL.md`
- `refs/oh-my-codex/skills/ralplan/SKILL.md`

## Deliverables

### 1. Typed workflow-signal extraction instead of score inputs assembled inline

Sprint 10 made routing scored. Sprint 11 should make the inputs first-class.

Implementation targets:

- introduce a dedicated workflow-signal module under `src/loader/runtime/`, for example around:
  - ambiguity signals
  - complexity signals
  - mutation / verification pressure
  - unresolved clarification slots
  - artifact availability and freshness
  - explicit user workflow requests
  - recent workflow timeline pressure
- separate signal extraction from route scoring so policy code can reason over a typed signal packet rather than rebuilding context ad hoc
- persist enough of the winning signal context to explain:
  - why clarify won over execute
  - why plan refresh was triggered
  - why direct execution was still allowed despite ambiguity
- keep route scoring tunable, but move the fragile task-text heuristics out of the coordinator path

The goal is not to build a giant intent engine. The goal is to make workflow policy more explainable, testable, and less accidental.

### 2. Intent-aware clarify strategy instead of generic follow-up rounds

Loader can now clarify more than once, but it still asks questions in a relatively flat way.

Implementation targets:

- define typed clarify objectives or slots such as:
  - desired outcome
  - acceptance criteria
  - constraints
  - non-goals
  - risk boundaries
- choose the next clarify question from unresolved slots instead of using a mostly generic follow-up loop
- adapt clarify behavior based on signal severity and task class while preserving a hard upper bound
- persist why clarify stopped:
  - enough boundaries gathered
  - budget exhausted
  - route pressure shifted toward plan or execute
  - explicit user answer narrowed the scope sufficiently
- carry unresolved slots forward into workflow state and artifacts so later plan/execute decisions can explain what was still uncertain

This is how Loader gets closer to OMX's deeper interview rigor without turning every task into a long questionnaire.

### 3. Semantic artifact invalidation and stronger re-plan discipline

Sprint 10 made plan refresh possible. Sprint 11 should make refresh triggers smarter.

Implementation targets:

- enrich planning artifacts with more structured metadata where it materially helps, for example:
  - expected touchpoints
  - acceptance-criteria anchors
  - planned files or subsystems
  - known risks or assumptions
- define broader invalidation triggers beyond file drift, for example:
  - verification evidence contradicts the plan assumptions
  - the implementation touched files or subsystems outside the expected scope
  - acceptance criteria changed materially after clarify or verification
  - the current task wording narrowed or expanded after the plan was written
- distinguish between:
  - targeted plan refresh
  - clarify reentry
  - full re-plan
- keep the runtime disciplined: prefer the smallest valid recovery move instead of restarting workflow lanes casually

This should move Loader closer to claw-code's stronger artifact discipline, where plans remain live contracts instead of just persisted markdown.

### 4. Workflow inspection that answers operator questions more directly

Sprint 10 made workflow history visible. Sprint 11 should make it more usable.

Implementation targets:

- extend `loader workflow show` with higher-signal inspection affordances such as:
  - filtering by mode or event kind
  - limiting to the most recent meaningful items
  - clearer summaries for refresh, reentry, and clarify-budget outcomes
- expose the signal/reason context that most directly answers questions like:
  - why did Loader ask again?
  - why did Loader refresh the plan?
  - why did Loader skip verify?
- keep session surfaces concise by surfacing only the most recent or most important workflow events by default
- avoid building a visual UI in this sprint; prioritize text inspection that reduces debugging time immediately

The goal is not prettier output. The goal is faster workflow debugging and better operator trust.

### 5. Continue shrinking `conversation.py` into a coordinator over runtime modules

Sprint 10 improved the split, but the coordinator still owns too much sequencing logic.

Implementation targets:

- extract additional orchestration seams under `src/loader/runtime/`, likely around:
  - signal extraction
  - clarify-lane control
  - plan refresh / invalidation decisions
  - workflow timeline append policy
- make `ConversationRuntime.run_turn(...)` read more like:
  - collect turn state
  - compute workflow signals
  - ask policy/orchestrator for the next lane decision
  - delegate lane execution
  - persist summary and timeline outcomes
- keep completion and downstream workflow handoff logic out of the signal-extraction path
- avoid replacing one monolith with another; new orchestration modules should have narrow responsibilities and direct tests

A good outcome is that `conversation.py` keeps shrinking because ownership is clearer, not because behavior gets hidden.

## Testing strategy

- unit coverage for:
  - typed workflow-signal extraction and normalization
  - route-policy scoring over structured signals
  - clarify-slot progression and stop reasons
  - semantic invalidation triggers and targeted recovery selection
- CLI coverage for:
  - `loader workflow show` filtering and summarization
  - session/workflow output for clarify exhaustion, plan refresh, and reentry reasons
- deterministic/runtime coverage for:
  - ambiguous tasks where clarify chooses different follow-up questions based on unresolved slots
  - verification failure that triggers plan refresh vs clarify reentry based on typed invalidation reasons
  - tasks that remain executable even with mild ambiguity because stronger signals favor direct execution
  - Sprint 00-10 parity scenarios staying green after the workflow-policy split deepens again
- regression coverage:
  - route policy should consume typed signals rather than rebuilding them ad hoc inside the coordinator
  - workflow inspection should continue to work after session resume and compaction

## Definition of done

- Loader extracts typed workflow signals before route scoring
- clarify behavior is intent-aware and persists why it continued or stopped
- plan refresh uses richer invalidation reasons than file drift alone
- workflow inspection better explains reentry, refresh, and clarify behavior
- `conversation.py` is slimmer again and more coordinator-like
- the full parity baseline remains green after the deeper workflow-policy split

## Explicitly out of scope

- full OMX-style consensus planning
- a visual workflow timeline UI
- a first-class permission rule editor
- AST-aware, LSP-aware, or symbol-aware editing
- multi-agent or team orchestration

## Audit

### Landed

- Loader now extracts typed workflow signals in `src/loader/runtime/workflow_signals.py`, and route decisions persist `signal_summary` context so we can explain why clarify, plan, or direct execute won without rebuilding those heuristics inside the coordinator
- clarify is now intent-aware instead of generic: `src/loader/runtime/clarify_strategy.py` defines explicit slots such as desired outcome, non-goals, acceptance criteria, constraints, decision boundaries, and likely touchpoints, and the runtime now persists why clarify continued or stopped around those slots
- replan discipline is broader and more honest: `src/loader/runtime/artifact_invalidation.py` can now distinguish targeted plan refresh, clarify reentry, and full re-plan based on semantic drift instead of only touched-file mismatch, and `src/loader/runtime/conversation.py` routes those recovery moves explicitly
- workflow inspection is more useful for actual operator questions: `loader workflow show` now supports mode/kind filtering and entry limits, and `src/loader/runtime/inspection.py` plus `src/loader/cli/main.py` surface concise workflow highlights for re-asks, reentries, refreshes, and verify skips
- `conversation.py` is slimmer again because clarify/plan lane execution moved into `src/loader/runtime/workflow_lanes.py`, which now owns lane prompts, artifact writes, clarify follow-up handling, and plan todo seeding while the main runtime acts more like a coordinator

### Verification

- `uv run pytest -q` is green: `197 passed`
- `tests/test_workflow_signals.py` covers typed signal extraction, recent timeline pressure, and persisted `signal_summary` state
- `tests/test_clarify_strategy.py` covers slot prioritization and targeted clarify questions
- `tests/test_artifact_invalidation.py` covers semantic invalidation and recovery-mode selection
- `tests/test_workflow_runtime.py` covers intent-aware clarify continuation, targeted plan refresh, and full re-plan through clarify reentry
- `tests/test_inspection.py` covers `loader workflow show` filtering/highlights plus session/status workflow inspection
- targeted `ruff` checks are green for `src/loader/runtime/workflow_signals.py`, `src/loader/runtime/clarify_strategy.py`, `src/loader/runtime/artifact_invalidation.py`, `src/loader/runtime/workflow_policy.py`, `src/loader/runtime/workflow_lanes.py`, `src/loader/runtime/conversation.py`, `src/loader/runtime/inspection.py`, and the new/expanded workflow inspection tests
- `uv run python -m compileall src/loader/cli/main.py` is green; full-file `ruff` on `src/loader/cli/main.py` still inherits the repo's older line-length backlog, so CLI verification here is anchored primarily by the inspection command tests

### Residual debt

- typed workflow signals are a better contract than inline heuristics, but they are still built from hand-tuned text/runtime cues rather than OMX-style ambiguity scoring, evidence passes, or richer task semantics
- clarify is now slot-driven, but it still stops well short of OMX's deep-interview pressure-pass discipline, repository-backed fact gathering, and task-profile-dependent depth
- artifact invalidation is broader now, but it is still lightweight and text-based; Loader does not yet reason over richer artifact metadata, deeper verification contradictions, or more explicit assumption tracking
- `loader workflow show` now answers the common operator questions much better, but it still lacks artifact diffs, prompt-history comparison, and richer timeline drill-down ergonomics
- `src/loader/runtime/conversation.py` is more coordinator-like than before Sprint 11, but it still owns the main turn loop, completion policy handoff, and some recovery sequencing that claw-code keeps in even more dedicated seams
