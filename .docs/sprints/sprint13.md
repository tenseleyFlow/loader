# Sprint 13: Turn Policy Narrowing, Assumption Ledger, and Artifact Diffs

## Prerequisites

Sprint 12

## Goals

Turn Loader's newly controllerized runtime into a more semantically explicit workflow system by shrinking the still-heavy `turn_iteration` seam, promoting assumptions and contradictions into first-class workflow state, and giving operators diff-oriented artifact visibility instead of only latest-state inspection.

Sprint 12 was a real structural win. Loader now has pressure-pass clarify, codebase-backed grounding, structured recovery evidence, and a controller-shaped runtime shell. That meaningfully closes the gap with claw-code and OMX. The audit is also honest about what still hurts:

- `turn_iteration.py` is still carrying a lot of repair, tool-routing, and completion policy in one seam
- contradiction and invalidation evidence are richer than before, but they are still mostly runtime-authored summaries rather than a reusable semantic ledger
- operator surfaces can explain "why did this happen?" better than before, but they still cannot show "what changed?" across briefs, plans, verification, or prompt contracts
- Loader now has better workflow discipline, but it still lacks some of the day-two operator ergonomics that make claw-code and OMX easier to trust during long tasks

The next leverage point is to stop treating semantic drift and operator visibility as one-off summaries and start treating them as durable contracts:

- the turn runtime should classify and route assistant output through narrower policy seams
- assumptions, contradictions, and acceptance anchors should survive across workflow phases as explicit state
- inspection should be able to show diffs between the artifacts and prompt contracts that drove behavior

This sprint is about making Loader more inspectable and less accidental:

- `turn_iteration` shrinks into narrower policy-oriented seams
- workflow invalidation gains an explicit assumption/contradiction ledger
- operator tooling gains artifact and prompt diff visibility
- Loader gets closer to claw-code not just in structure, but in debuggability

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/policy_engine.rs`
- `refs/claw-code/rust/crates/runtime/src/prompt.rs`
- `refs/claw-code/PARITY.md`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/modes/base.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`
- `refs/oh-my-codex/skills/deep-interview/SKILL.md`
- `refs/oh-my-codex/skills/ralplan/SKILL.md`

## Deliverables

### 1. Split `turn_iteration` into narrower response-policy seams

Sprint 12 made `conversation.py` coordinator-shaped. Sprint 13 should keep the same discipline for the still-heavy iteration seam.

Implementation targets:

- extract narrower helpers under `src/loader/runtime/`, likely around:
  - assistant-response classification
  - repair routing
  - final-answer routing
  - tool-batch routing
  - no-tool completion handoff
- make `turn_iteration.py` read more like:
  - request assistant turn
  - classify response
  - delegate the winning route
  - return loop-state deltas
- keep the main behavior unchanged while reducing policy density per module
- add direct controller tests so future iteration changes do not depend only on broad runtime integration coverage

The goal is not more files for their own sake. The goal is to make assistant-turn behavior easier to tune deliberately.

### 2. Assumption and contradiction ledger instead of one-off evidence summaries

Sprint 12 introduced richer drift evidence. Sprint 13 should make that evidence durable and reusable.

Implementation targets:

- define a typed workflow ledger contract under `src/loader/runtime/` for:
  - explicit assumptions
  - confirmed assumptions
  - contradicted assumptions
  - acceptance anchors
  - open decision boundaries
  - closed decision boundaries
- thread that ledger through clarify, planning, verification, and recovery instead of only summarizing evidence at refresh time
- persist enough structure to answer:
  - which assumption was invalidated?
  - which workflow phase introduced it?
  - what evidence contradicted it?
  - whether the contradiction forced refresh, reentry, or only inspection visibility
- keep the first version pragmatic and text-first; do not try to build a symbolic reasoning engine

This is how Loader gets from "richer summaries" to a more explicit semantic workflow contract.

### 3. Artifact and prompt diff surfaces for operators

Loader can now show the latest prompt and workflow timeline. Sprint 13 should help operators see what changed.

Implementation targets:

- add diff-oriented inspection surfaces, likely around:
  - clarify brief vs refreshed brief
  - old plan vs refreshed plan
  - workflow ledger changes across reentry
  - prompt metadata or prompt-body diffs across relevant turns
- keep the product surface text-first and operator-friendly, for example via:
  - `loader workflow show --diff`
  - `loader prompt diff`
  - or an equivalent `loader artifact show` family if that is cleaner
- include concise change summaries by default and fuller diffs when explicitly requested
- avoid a visual UI in this sprint; prioritize fast CLI/TUI debugging value

The goal is to make workflow changes legible, not just persisted.

### 4. Workflow/operator surfaces that explain semantic change, not only event history

Sprint 12 improved evidence visibility. Sprint 13 should improve semantic visibility.

Implementation targets:

- extend inspection surfaces so they can show:
  - which assumptions remain open
  - which assumptions were contradicted
  - which acceptance anchors changed across clarify/plan/verify
  - whether a refresh was forced by contradiction, touchpoint drift, or acceptance drift
- preserve concise defaults so everyday status remains readable
- make session/workflow output useful for long-running or resumed tasks, not only single-turn debugging

This brings Loader closer to claw-code's stronger operator trust model.

### 5. Keep the parity baseline honest while the runtime narrows again

Sprint 12 closed a big structural loop. Sprint 13 should protect that gain.

Implementation targets:

- add direct tests for the newly split iteration policy seams
- extend workflow/inspection coverage for diff and ledger behavior
- keep existing parity scenarios green after the iteration split
- update `PARITY.md` and the sprint audit only after the new surfaces and contracts are actually covered

## Testing strategy

- unit coverage for:
  - response classification and per-route delegation
  - assumption-ledger updates and contradiction recording
  - artifact/prompt diff formatting and summaries
  - workflow refresh decisions reading from the new ledger state
- CLI coverage for:
  - prompt/artifact/workflow diff surfaces
  - workflow/session output for contradiction-led refreshes
- deterministic/runtime coverage for:
  - a clarify answer that seeds assumptions later contradicted during verification
  - a plan refresh where the operator surface can show exactly what changed
  - a resumed session where workflow inspection still reflects semantic ledger state
  - Sprint 00-12 parity scenarios staying green after the deeper iteration split
- regression coverage:
  - iteration refactors should not regress verify/fix, permission, or explore contracts
  - diff surfaces should read persisted artifacts/session state rather than reconstructing history heuristically

## Definition of done

- `turn_iteration.py` is slimmer and delegates through narrower response-policy seams
- assumptions and contradictions are persisted as explicit workflow state
- operators can inspect artifact or prompt diffs from the product surface
- workflow inspection explains semantic change, not only route history
- the full parity baseline remains green after the deeper iteration split

## Explicitly out of scope

- full OMX-style consensus planning
- a visual workflow diff UI
- AST-aware, LSP-aware, or symbol-aware editing
- a first-class permission rule editor
- multi-agent or team orchestration
