# Sprint 10: Route Pressure, Clarify Depth, and Workflow Timeline

## Prerequisites

Sprint 09

## Goals

Turn Loader's workflow routing from a better heuristic into a more opinionated workflow policy, and make that policy inspectable over time instead of only at the latest state.

Sprint 09 gave Loader a validated turn state machine, typed workflow decisions, and prompt preview. That was the right contract layer, but the audit is honest about what still hurts:

- workflow routing is still threshold-based and lighter than OMX's route discipline
- clarify and plan are still mostly one-shot preprocessors instead of deeper workflow lanes
- status/session surfaces can explain the latest decision, but not the sequence of decisions that got Loader there
- `conversation.py` is slimmer, but it still owns route/handoff/skip logic that should live in dedicated workflow policy code

The next leverage point is to stop asking only "what mode are we in now?" and start asking "what workflow pressure led us here, what is still unresolved, and what should happen next if the task keeps moving?"

This sprint is about workflow rigor:

- routing becomes a scored workflow policy instead of a small threshold router
- clarify and plan become more durable lanes with bounded depth and freshness rules
- workflow history becomes a persisted timeline instead of only last-known fields
- `conversation.py` gets thinner again by delegating route pressure and handoff policy

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/skills/deep-interview/SKILL.md`
- `refs/oh-my-codex/skills/ralplan/SKILL.md`

## Deliverables

### 1. Workflow policy engine instead of threshold-only routing

Sprint 09 made workflow decisions typed. Sprint 10 should make them more deliberate.

Implementation targets:

- replace the current simple threshold router with a workflow-policy module that evaluates route pressure using typed signals, for example:
  - ambiguity
  - complexity
  - mutability / verification pressure
  - artifact availability
  - artifact freshness
  - explicit user request
  - unresolved assumptions
- produce a scored route evaluation that explains:
  - which mode won
  - why it won
  - what the runner-up pressure was
  - whether a downstream mode is already scheduled
- keep initial route, handoff, and reentry decisions on the same policy surface instead of splitting them between the router and coordinator code
- make non-mutating "skip verify" behavior a typed workflow decision instead of an implicit completion branch

The goal is not a giant planning engine. The goal is to make Loader's workflow behavior less accidental and easier to tune deliberately.

### 2. Deeper clarify lane with bounded follow-through

Loader can clarify today, but it still behaves like a one-question prelude.

Implementation targets:

- allow clarify to continue for more than one round when the first answer leaves key boundaries unresolved
- keep that depth bounded with an explicit clarify budget and escalation rules
- persist unresolved assumptions or open questions in workflow state so later turns can explain why clarify continued or stopped
- distinguish between:
  - clarify completed cleanly
  - clarify exhausted its budget
  - clarify was bypassed by a stronger route decision
- keep the UX disciplined: no deep-interview sprawl, only focused rounds that materially reduce ambiguity

This is how Loader gets closer to OMX's deeper interview behavior without turning every task into a questionnaire.

### 3. Plan freshness and re-plan discipline

Loader now persists plans, but it still treats them as mostly static once created.

Implementation targets:

- add typed freshness checks for clarify briefs and plan artifacts so Loader can detect when a plan no longer matches the task state
- define explicit refresh triggers, for example:
  - task meaning changed after clarification
  - verify/fix reentry reveals plan gaps
  - touched files drift outside the planned touchpoints
  - acceptance criteria changed materially
- route stale artifacts through a typed re-plan or plan-refresh decision instead of silently continuing with outdated plans
- keep refresh lightweight: prefer targeted plan refresh over full workflow restart when possible

This brings Loader closer to claw-code's stronger execution discipline: artifacts should steer the turn, not become dead files on disk.

### 4. Persisted workflow timeline and operator surfaces

Sprint 09 exposed the latest workflow reason. Sprint 10 should expose workflow history.

Implementation targets:

- persist workflow timeline entries in session/runtime state, including:
  - route decisions
  - handoffs
  - reentries
  - clarify-budget outcomes
  - plan refresh decisions
  - the latest prompt-contract metadata attached to those moments when relevant
- add a first-class inspection surface such as `loader workflow show` for summarizing that timeline without a live model call
- extend `loader session show` to surface the most recent workflow timeline items when present
- keep the output operator-friendly:
  - newest-important events first
  - concise summaries
  - artifact paths or mode transitions only when they help explain behavior

The goal is to make Loader's workflow evolution debuggable, not just its final state.

### 5. Continue slimming `conversation.py`

Sprint 09 improved the coordinator. Sprint 10 should keep the split moving in the same direction.

Implementation targets:

- extract route-pressure evaluation, clarify-budget handling, and artifact-freshness checks into dedicated runtime modules
- make `ConversationRuntime.run_turn(...)` primarily:
  - collect turn context
  - ask the workflow policy for a route or reentry decision
  - delegate clarify/plan/execute/verify work
  - append timeline entries
  - finalize the turn summary
- avoid rebuilding the monolith by placing all new workflow-policy logic under `src/loader/runtime/`

A good outcome is that `conversation.py` keeps shrinking because policy is becoming modular, not because the behavior is disappearing.

## Testing strategy

- unit coverage for:
  - workflow-policy score breakdowns and winning-route selection
  - clarify-budget continuation vs exhaustion
  - artifact-freshness detection and targeted plan refresh triggers
  - workflow timeline persistence and serialization
- CLI coverage for:
  - `loader workflow show`
  - workflow timeline rendering in `loader session show`
- deterministic/runtime coverage for:
  - ambiguous tasks that require more than one clarify round before execution
  - verify/fix reentry that triggers plan refresh instead of blindly continuing
  - non-mutating tasks skipping verify through a typed workflow decision
  - Sprint 00-09 parity scenarios staying green after the workflow-policy split
- regression coverage:
  - route and reentry choices should come from the workflow-policy layer, not ad hoc coordinator branches
  - persisted workflow history should survive session resume and inspection

## Definition of done

- Loader routes through a scored workflow-policy engine instead of a small threshold router
- clarify can continue for bounded, explainable follow-up rounds when ambiguity remains
- plan artifacts can be marked stale and refreshed through typed workflow decisions
- workflow history is persisted and inspectable, not only the latest decision
- `conversation.py` is slimmer again and more policy-driven
- the full parity baseline remains green after the workflow-policy and timeline work

## Explicitly out of scope

- full OMX-style consensus planning
- a visual workflow timeline UI
- a first-class permission rule editor
- AST-aware, LSP-aware, or symbol-aware editing
- multi-agent or team orchestration
