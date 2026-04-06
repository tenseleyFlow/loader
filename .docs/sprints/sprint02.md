# Sprint 02: Definition of Done and Verify/Fix Loop

## Prerequisites

Sprint 01

## Goals

Replace heuristic completion with an evidence-backed completion contract. This is the single highest-leverage behavioral change in the plan and the direct answer to the user's stated complaints:

- finishing too early without followup
- weak tool follow-through
- poor task closure

The reference for the contract shape is `refs/oh-my-codex/skills/ralph/SKILL.md` (the persistence-until-done protocol) and `refs/oh-my-codex/src/verification/verifier.ts` (task-size-aware evidence scaling).

This sprint deliberately runs *before* permission modes (Sprint 03). Permissions are a safety win; this is the behavior win, and the user asked for behavior first.

## Deliverables

### 1. Definition of Done object

Add a `DefinitionOfDone` dataclass that every non-trivial task carries from start to finish:

```python
@dataclass
class DefinitionOfDone:
    task_statement: str
    acceptance_criteria: list[str]          # the testable claims the task must satisfy
    verification_commands: list[str]        # commands whose output is the evidence
    pending_items: list[str]                # outstanding subtasks (zero before completion)
    completed_items: list[str]              # finished subtasks (audit trail)
    evidence: list[VerificationEvidence]    # populated by the verify phase
    confidence: Literal["high", "medium", "low"]
    status: Literal["draft", "in_progress", "verifying", "fixing", "done", "failed"]
```

The DoD object is constructed at task entry and updated as the runtime advances. It is the single source of truth for "is this task done?".

### 2. Verify phase in the runtime

Add an explicit verify phase to `runtime.conversation` that runs *after* the model thinks it is done but *before* the turn returns a final answer.

The verify phase:

- runs each `verification_commands` entry as a tool call
- captures stdout, stderr, and exit code as `VerificationEvidence`
- attaches each piece of evidence to the DoD object
- gates completion on (a) all verification commands exited zero, (b) `pending_items` is empty, (c) the model has produced an evidence summary that references the captured output

If any of those gates fail, the runtime moves into the fix phase rather than completing.

### 3. Fix loop

Verification failure does not return to the user. It returns to execution with:

- the failed evidence attached to the next prompt
- a structured "what failed and why" message
- a bounded retry budget (default 3 attempts; configurable per task via DoD)
- escalation to the user only when the budget is exhausted

This is the same shape as `refs/oh-my-codex/skills/ralph/SKILL.md` Step 7.5–7.6 (mandatory verification + regression re-verification + retry on failure).

### 4. Task-size-aware evidence requirements

Borrow the sizing model from `refs/oh-my-codex/src/verification/verifier.ts:99-106`:

- **small** (≤3 files, <100 lines changed): minimal verification — typecheck + tests for affected modules
- **standard** (≤15 files, <500 lines): full stack — typecheck + tests + lint + smoke
- **large** (>15 files or >500 lines): comprehensive — typecheck + tests + lint + integration + regression

Sizing is computed from the actual tool call history of the turn, not guessed from the prompt.

Conversational/lookup-only tasks (no tool calls that mutate state) skip the verify phase entirely. This is what keeps simple tasks cheap.

### 5. Minimum `.loader/` directory layout

Create the minimum state directory shape so DoD objects can be persisted:

```text
.loader/
└── dod/
    └── {timestamp}-{task-slug}.json
```

This is intentionally narrow. The full session/memory/compaction layout is Sprint 05's job. Sprint 02 only needs somewhere to put DoD objects so that fix-loop continuations can reload them after process restarts, and so that future sprints have a directory to extend.

`.loader/` should be added to `.gitignore` as part of this sprint.

### 6. CLI/TUI surfaces for the DoD state

The TUI status line and the non-TUI CLI output both need to surface:

- current DoD status (`draft` / `in_progress` / `verifying` / `fixing` / `done`)
- pending items count
- last verification result

This is what makes the contract visible to the user instead of hidden inside the runtime.

## Testing strategy

- a task with verification commands cannot complete without evidence (assert: completion attempted before evidence collected → routed to verify phase)
- a verification failure routes to fix loop, not to final answer
- the fix loop respects the retry budget and escalates to the user on exhaustion
- a conversational task (no mutating tool calls) skips verify entirely
- DoD objects round-trip through `.loader/dod/` and survive a simulated process restart
- task sizing classifies correctly across small/standard/large boundaries
- the CLI/TUI status surfaces show the right phase at each transition

## Definition of done

- Loader no longer relies on heuristic continuation prompts alone
- completion is explicit and evidence-backed
- `DefinitionOfDone` objects exist on disk under `.loader/dod/`
- failed verification cannot escape into a "looks done" final answer
- simple tasks stay cheap (verify is skipped); complex tasks enter the verify/fix loop automatically
- the user can see the DoD phase from the CLI and TUI
