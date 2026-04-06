# Sprint 04: Mode Router, Clarify Mode, and Plan Artifacts

## Prerequisites

Sprint 03

## Goals

Teach Loader to separate kinds of work instead of improvising one workflow for everything. Builds on the DoD contract from Sprint 02 and the hook lifecycle from Sprint 03.

This sprint is the direct answer to:

- spending too long on simple tasks (modes route lookups out of the full loop)
- overthinking small work (clarify only fires for genuinely ambiguous prompts)
- jumping into execution without a plan on complex work

The references are `refs/oh-my-codex/skills/deep-interview/SKILL.md` (clarify mode) and `refs/oh-my-codex/skills/ralplan/SKILL.md` (plan mode). Loader should copy the artifact discipline first; the full Planner/Architect/Critic loop and the ambiguity-scoring formula are stretch goals, not Sprint 04 requirements.

## Deliverables

### 1. Tool prerequisites pulled forward from Sprint 06

The clarify and DoD work both need tools that don't exist yet. Add them now rather than at the end of the plan:

- **`TodoWrite`** — task/todo tracking. The "zero pending tasks" gate in Sprint 02's DoD contract is currently empty because there is no tool to write tasks into. Without `TodoWrite`, Sprint 02's contract has a hole.
- **`AskUserQuestion`** — structured user-question surface. The clarify mode needs a way to ask one question per round (per `deep-interview/SKILL.md`) rather than embedding questions in free-form responses.

These two tools are the minimum new tool surface this sprint introduces. The broad expansion (diff/patch-aware editing, git helpers, web fetch, etc.) stays in Sprint 06.

Both tools declare their permission level (`read-only` for `TodoWrite` writing to `.loader/`, `read-only` for `AskUserQuestion`) and integrate with the hook lifecycle from Sprint 03.

### 2. Mode router

Introduce a router that selects a mode at task entry based on the task shape and explicit user intent:

- `clarify` — fired when ambiguity score crosses a threshold or the user invokes `--clarify`
- `plan` — fired for complex tasks (heuristic on prompt length / signal density, or explicit `--plan`)
- `execute` — the default for concrete actionable prompts
- `verify` — already exists from Sprint 02; the router wires it in as the gate after `execute`

The router does not have to be smart in this sprint. It needs to be *explicit*: the chosen mode is logged, surfaced in the TUI status line, and recorded in the DoD object.

A heuristic-only first pass is fine. Borrowing the OMX deep-interview ambiguity formula is a stretch goal.

### 3. Clarify artifact

When the router selects `clarify` mode, the runtime drives a one-question-per-round loop (using `AskUserQuestion`) and writes a task brief to `.loader/briefs/{timestamp}-{slug}.md` containing:

- task statement
- desired outcome
- in-scope items
- non-goals
- decision boundaries
- constraints
- likely touchpoints
- assumptions

This is a simplified port of `refs/oh-my-codex/skills/deep-interview/SKILL.md` Phase 4. Loader does not need the full ambiguity scoring or pressure-pass discipline yet. It needs the artifact.

The brief is then handed off as input to the next mode (`plan` or `execute`), and the DoD object's `acceptance_criteria` is seeded from the brief.

### 4. Planning artifacts

When the router selects `plan` mode, the runtime produces two persistent artifacts under `.loader/plans/{timestamp}-{slug}/`:

- `implementation.md` — what files change, in what order, with what risks
- `verification.md` — the verification commands and acceptance criteria that will populate the DoD object

These do not need full OMX ralplan complexity (Planner/Architect/Critic with iteration cap). They need to:

- exist on disk
- survive across turns (so a process restart can resume planning)
- feed directly into the DoD object created by Sprint 02
- be visible to the user via the TUI

### 5. Mode-specific prompts

Replace Sprint 00's single generic system prompt with mode-specific prompts:

- `clarify` mode prompt enforces "ask one question, do not propose solutions yet"
- `plan` mode prompt enforces "produce two artifacts, do not start writing code"
- `execute` mode prompt is the current Loader system prompt, trimmed (the global "no numbered steps" rule that breaks reporting tasks should be relaxed)
- `verify` mode prompt enforces "run the verification commands, attach evidence, do not declare done without zero failures"

Mode-specific prompts are how the routing decision actually changes model behavior.

### 6. Wire artifacts into the DoD object

The DoD object from Sprint 02 gains optional links:

- `clarify_brief: Path | None`
- `implementation_plan: Path | None`
- `verification_plan: Path | None`

When verify phase runs, it pulls verification commands from `verification.md` if present, falling back to the inline DoD list otherwise.

## Testing strategy

- ambiguous prompts route into `clarify` (assert via the mock harness)
- complex prompts route into `plan`
- simple lookups route directly into `execute` and skip the verify phase entirely
- clarify briefs round-trip through `.loader/briefs/`
- planning artifacts round-trip through `.loader/plans/`
- the DoD object correctly absorbs `acceptance_criteria` from a clarify brief and `verification_commands` from a plan
- mode-specific prompts produce mode-appropriate behavior under the mock harness
- a verify failure that returns to `execute` (per Sprint 02's fix loop) does not re-trigger `clarify` or `plan`

## Definition of done

- Loader routes tasks into modes instead of treating every prompt the same way
- clarify briefs and planning artifacts exist on disk and feed the DoD object
- `TodoWrite` and `AskUserQuestion` tools exist and are wired into the hook lifecycle
- mode-specific prompts replace the single generic prompt
- simple tasks stay lightweight; complex tasks become more structured
- the TUI surfaces the active mode
