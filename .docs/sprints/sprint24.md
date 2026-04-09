# Sprint 24: TUI Runtime Convergence, Verification Lifecycle, and Facade Narrowing

## Prerequisites

Sprint 23

## Goals

Take the next honest step after Sprint 23: stop treating the TUI as the last major product path that still defaults to `Agent` by habit, widen the verification-observation story from “what ran” toward “what is planned, pending, or stale,” and keep narrowing the public shell without pretending Loader is ready to delete it.

Sprint 23 changed the remaining debt in a useful way:

- non-TUI CLI, `loader explore`, and the scripted runtime harness now use the runtime-first owner seam below `Agent`
- verification observations now enter the canonical workflow timeline closer to execution through per-command verification events
- operator surfaces can now show which runtime-owner path produced the current session/accountability state
- but the TUI still routes through the public `Agent` shell even though it primarily consumes runtime-owned behavior
- verification observations still say more about commands that ran than about commands that are only planned, still pending, or now stale
- Loader is much closer to an explicit public/runtime boundary, but it still has not fully converged on what must remain public-shell-only versus what can be runtime-first by default

Sprint 24 should keep using the references as architectural guardrails, not as a feature-copy list.

The standard remains:

- use claw-code to sharpen runtime/bootstrap ownership, event accountability, and narrower UI-facing shell seams
- use OMX to sharpen verifier-lifecycle visibility and auditability around pending vs observed proof
- do not add work just because the refs have it
- do add work when the refs show that Loader is still too shell-bound, too late in its verification story, or too ambiguous about what the public shell still owns

`audit.txt` remains a guardrail against wrapper-heavy drift and compatibility-by-habit. It is not the factual roadmap.

The references for this sprint are:

- `refs/claw-code/rust/crates/runtime/src/bootstrap.rs`
- `refs/claw-code/rust/crates/runtime/src/session_control.rs`
- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/lane_events.rs`
- `refs/claw-code/rust/crates/runtime/src/green_contract.rs`
- `refs/claw-code/PARITY.md`
- `refs/oh-my-codex/src/verification/verifier.ts`
- `refs/oh-my-codex/src/autoresearch/runtime.ts`
- `refs/oh-my-codex/src/autoresearch/contracts.ts`
- `refs/oh-my-codex/src/hooks/session.ts`
- `.docs/PARITY.md`
- `.docs/audit.txt`
- `.docs/audit_sprints/trunk_sitrep.md`
- `.docs/sprints/sprint23.md`

## Deliverables

### 1. Move the TUI onto the runtime-first owner seam

Sprint 23 made runtime-first real in the CLI and scripted harness. Sprint 24 should make the TUI stop depending on `Agent` by default when it mostly consumes runtime-owned behavior.

Implementation targets:

- inventory what the TUI actually needs from its execution owner across:
  - `src/loader/ui/app.py`
  - `src/loader/ui/adapter.py`
  - `src/loader/runtime/public_shell.py`
  - `src/loader/runtime/runtime_handle.py`
  - `src/loader/cli/main.py`
- define or refine a small shell-owner contract for the TUI instead of letting `Agent` be the assumed type
- migrate the TUI launch path to the runtime-first owner where that does not weaken the public compatibility story
- preserve explicit public compatibility where it still matters, but stop using `Agent` as the default UI owner by habit

The goal is not to remove `Agent` from the codebase. The goal is to make the biggest remaining real product path stop depending on it unnecessarily.

### 2. Expand verification observations to planned, pending, and stale states

Sprint 23 captures commands that ran. Sprint 24 should make Loader more honest about verification work that has not yet happened or is no longer fresh.

Implementation targets:

- inventory where verification lifecycle facts already exist before or beyond command execution across:
  - `src/loader/runtime/dod.py`
  - `src/loader/runtime/finalization.py`
  - `src/loader/runtime/task_completion.py`
  - `src/loader/runtime/workflow_policy.py`
  - `src/loader/runtime/policy_timeline.py`
  - `src/loader/runtime/workflow_timeline_read_model.py`
- define which observation kinds Loader should represent directly, such as:
  - verification planned
  - verification pending
  - verification stale
  - verification intentionally skipped
  - verification observed as passed or failed
- keep those lifecycle facts inside the canonical policy/accountability story instead of inventing a second peer model

The goal is to make Loader answer “what proof is still pending?” as directly as it can already answer “what proof did we observe?”

### 3. Tighten the public facade boundary around runtime-first defaults

Sprint 23 improved real internal adoption, but the long-term shell boundary is still not quite settled enough.

Implementation targets:

- inventory what remains in `src/loader/agent/loop.py` that is:
  - true public compatibility API
  - UI integration glue
  - leftover runtime ownership
- move any still-obviously-runtime ownership lower when that reduces ambiguity
- add or extend boundary tests so future work does not drift back toward `Agent` as the default internal seam after Sprint 24

The goal is not to shrink files for vanity. The goal is to make the remaining public shell answerable and intentionally narrow.

### 4. Improve operator visibility for owner path and verification lifecycle

Once Loader can show runtime-owner provenance and richer verification lifecycle state, the existing product surfaces should make that easier to audit.

Implementation targets:

- improve the current surfaces so users can answer:
  - which owner path produced this session?
  - what verification is planned, pending, stale, skipped, or observed?
  - what evidence is still needed versus already satisfied?
- prefer improving:
  - `loader status`
  - `loader session show`
  - `loader workflow show`
  - the TUI status surface
  over inventing a new command unless one is clearly cleaner

The goal is to make Loader easier to audit live and after the fact, not simply more verbose.

## Testing strategy

- unit coverage for:
  - the TUI/runtime shell-owner contract
  - runtime-first TUI launch routing
  - planned/pending/stale verification-observation normalization and projection
- runtime coverage for:
  - the TUI or UI-facing shell path using the runtime-first owner without losing steering, confirmation, or question handling
  - policy/accountability views that now distinguish pending vs observed verification
- regression coverage for:
  - no drift back toward `Agent` as the default owner for internal UI/runtime integrations
  - no duplicate verification lifecycle truth beside the canonical policy timeline
  - no regression in Sprint 23's runtime-owner visibility and execution-time verification observations

## Definition of done

- the TUI uses a runtime-first owner seam below `Agent`, or any remaining public-shell dependency is explicit and justified
- Loader preserves richer verification lifecycle state than just “command ran” within the canonical policy/accountability story
- the public facade boundary is narrower or more explicitly defended on purpose
- existing status/session/workflow/TUI surfaces expose the stronger owner-path and verification-lifecycle story without multiplying product commands
- Sprint 23's runtime-first integration and verification-producer gains remain green

## Explicitly out of scope

- deleting `Agent` as the public compatibility surface
- full claw-code policy-engine parity
- model-authored verifier narratives as a required runtime dependency
- AST-aware semantic diffs
- a broad visual workflow UI redesign
- multi-agent or team orchestration
