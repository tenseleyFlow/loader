# Sprint 12: Interview Pressure, Semantic Evidence, and Turn Orchestration

## Prerequisites

Sprint 11

## Goals

Turn Loader's newer workflow structure into a more disciplined execution contract by deepening clarify beyond slot selection, making semantic invalidation rely on richer evidence than text overlap alone, and shrinking the main turn loop into a clearer orchestration shell.

Sprint 11 closed several real gaps. Loader now has typed workflow signals, slot-aware clarify, semantic invalidation, better workflow inspection, and a slimmer coordinator. That is meaningful progress toward claw-code and OMX, but the audit is honest about what still hurts:

- typed workflow signals are still hand-tuned runtime heuristics rather than a deeper ambiguity/evidence model
- clarify is more intentional now, but it still lacks OMX's pressure-pass discipline, evidence-chasing, and codebase-backed interview style
- artifact invalidation is broader than file drift, but it still reasons from lightweight text overlap instead of richer structured evidence
- `conversation.py` is smaller, but it still owns the main assistant/recovery/completion orchestration loop that the refs spread across narrower runtime seams

The next leverage point is to stop treating clarify as "ask a better next question" and start treating it as "run a bounded interview with explicit pressure passes, factual grounding, and a stronger handoff contract for later execution."

This sprint is about execution rigor:

- clarify gains pressure-pass behavior instead of only slot-follow-up behavior
- semantic invalidation uses richer structured evidence and contradiction tracking
- the main turn loop shrinks again by delegating orchestration checkpoints into dedicated runtime modules
- Loader gets closer to closed-source agentic tools not by more prompt prose, but by stronger workflow contracts

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/modes/base.ts`
- `refs/oh-my-codex/skills/deep-interview/SKILL.md`
- `refs/oh-my-codex/skills/ralplan/SKILL.md`

## Deliverables

### 1. Pressure-pass clarify controller instead of slot selection alone

Sprint 11 made clarify targeted. Sprint 12 should make it disciplined.

Implementation targets:

- introduce a dedicated clarify controller under `src/loader/runtime/` that tracks:
  - current interview stage
  - weakest clarity dimension
  - whether a pressure pass has occurred
  - whether non-goals and decision boundaries are explicit
  - how much interview budget remains
- extend clarify reasoning beyond "what slot is unresolved?" to also ask:
  - was the last answer too broad?
  - has this assumption been challenged yet?
  - do we still need an example, counterexample, tradeoff, or explicit stop boundary?
- persist clarify progress in structured form so later workflow decisions can explain:
  - which dimension clarify was targeting
  - whether Loader was still gathering boundaries
  - whether it stopped because the budget was exhausted or because readiness gates were met
- keep it bounded and pragmatic:
  - no unbounded interviews
  - no long questionnaires
  - one question at a time with explicit stop conditions

The goal is not to copy OMX wholesale. The goal is to adopt the parts that materially reduce misaligned execution and premature planning.

### 2. Codebase-backed clarify grounding and stronger requirement artifacts

Sprint 11 still relies mostly on the user answer plus task text. Sprint 12 should let clarify lean on facts Loader can gather directly.

Implementation targets:

- add a lightweight preflight/context seam for brownfield tasks that can feed clarify with discovered facts before asking the user for repository details
- prefer evidence-backed clarify questions when Loader already knows something, for example:
  - "I found X in Y. Should this change follow that pattern?"
  - "The current touchpoints appear to be A and B. Should I keep C out of scope?"
- persist richer clarify artifact metadata where it helps downstream runtime behavior, for example:
  - explicit non-goal status
  - explicit decision-boundary status
  - whether a pressure pass occurred
  - likely touchpoint evidence
  - inferred vs confirmed boundaries
- keep this grounded in Loader's existing tool surface rather than inventing a large research subsystem

This moves Loader closer to OMX's "reduce user effort and don't ask for facts we can discover" principle.

### 3. Structured semantic evidence for invalidation and replan decisions

Sprint 11 improved invalidation, but it still reasons mostly from text coverage. Sprint 12 should give recovery choices a stronger evidence model.

Implementation targets:

- define a structured invalidation/evidence contract under `src/loader/runtime/`, for example around:
  - confirmed touchpoints
  - inferred touchpoints
  - acceptance anchors
  - contradicted assumptions
  - verification contradiction signals
  - changed user boundaries after clarify
- teach invalidation to distinguish:
  - plan mismatch
  - brief contradiction
  - verification contradiction
  - stale assumptions
- improve recovery selection so Loader can explain not only what it chose, but what evidence forced that choice
- preserve "smallest valid recovery move first" as the governing behavior

This is how Loader gets from "semantic-ish refresh" to a more trustworthy workflow contract.

### 4. Turn orchestration split beyond lane execution

Sprint 11 moved clarify/plan lanes out. Sprint 12 should keep shrinking the top-level turn loop.

Implementation targets:

- extract additional runtime seams under `src/loader/runtime/`, likely around:
  - turn preparation/bootstrap
  - workflow recovery/reentry control
  - completion/continuation orchestration
  - assistant-response repair routing
- make `ConversationRuntime.run_turn(...)` read more like:
  - initialize turn state
  - prepare workflow contract
  - delegate iteration/orchestration helpers
  - finalize summary
- avoid creating a new monolith module; prefer narrow orchestration seams with direct tests

A good outcome is that the turn loop becomes easier to reason about and less likely to collect ad hoc behavior again.

### 5. Workflow/operator surfaces that explain evidence, not just decisions

Sprint 11 made `loader workflow show` more useful. Sprint 12 should make it explain the evidence behind recovery and clarify pressure more directly.

Implementation targets:

- extend workflow inspection surfaces to show:
  - whether a pressure pass occurred
  - which clarify dimension was active
  - which evidence triggered refresh or reentry
  - which assumptions were still unresolved
- keep the default UX concise, but expose richer detail when explicitly requested
- avoid a visual UI in this sprint; prioritize text surfaces that make the runtime easier to debug immediately

## Testing strategy

- unit coverage for:
  - clarify pressure-pass progression and readiness gates
  - codebase-backed clarify question selection from discovered facts
  - structured invalidation evidence and contradiction handling
  - new orchestration seams preserving current turn behavior
- CLI coverage for:
  - workflow inspection showing clarify pressure/evidence
  - session/workflow output for contradiction-driven reentry
- deterministic/runtime coverage for:
  - ambiguous brownfield tasks where Loader asks evidence-backed clarify questions
  - tasks that need an assumption/tradeoff pressure pass before planning
  - verification contradictions that trigger targeted refresh vs full re-plan
  - Sprint 00-11 parity scenarios staying green after the deeper orchestration split
- regression coverage:
  - clarify should not ask the user for repository facts Loader can gather directly
  - orchestration extraction should not regress the verify/fix or permission/runtime contracts

## Definition of done

- clarify uses a bounded pressure-pass controller rather than slot selection alone
- brownfield clarify can ask evidence-backed questions from discovered facts
- invalidation relies on richer structured evidence and contradiction tracking
- workflow/operator surfaces explain clarify and recovery evidence more directly
- `conversation.py` is slimmer again and more orchestration-shell-like
- the full parity baseline remains green after the deeper clarify/orchestration split

## Explicitly out of scope

- full OMX-style consensus planning
- a visual workflow timeline UI
- a first-class permission rule editor
- AST-aware, LSP-aware, or symbol-aware editing
- multi-agent or team orchestration
