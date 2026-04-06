# Loader Deep Dive: Gaps, Strengths, and a Path Toward Claw-Like Behavior

Date: 2026-04-06

## Scope and assumptions

This report compares three things:

1. `Loader` itself
2. `refs/claw-code`, using the Rust workspace under `refs/claw-code/rust/` as the canonical runtime
3. `refs/oh-my-codex` as the workflow-layer parent repo

Assumption: `oh-my-codex` is the correct “parent repo” for this exercise. That assumption is based on:

- `refs/claw-code/README.md`
- `refs/claw-code/PHILOSOPHY.md`
- the fact that `refs/claw-code` explicitly describes `src/` as a companion Python/reference workspace, not the primary runtime

If you meant a different parent, we should rerun the comparison against that repo, but this is a solid first pass.

## Executive summary

Loader has the right instincts but is operating at the wrong layer.

The codebase already knows that models need:

- planning help
- recovery help
- confidence checks
- completion checks
- safe tool use

But Loader mostly tries to enforce those after the model has already started drifting. `claw-code` and `oh-my-codex` get better behavior because they shape the work before, during, and after the model call:

- before: explicit mode selection, clarification, approved planning artifacts
- during: durable runtime state, richer tool surface, explicit permission model, session persistence
- after: verification protocols, completion gates, retry/fix loops, parity harnesses, operator diagnostics

The biggest lesson is not “copy their prompt.”

The biggest lesson is:

> Loader needs a stronger execution contract, not just stronger prompting.

If we want Loader to feel closer to `claw-code` regardless of model choice, the highest-leverage work is:

1. replace the monolithic heuristic loop with a typed turn engine
2. add durable workflow/state artifacts
3. make “definition of done” evidence-based instead of heuristic
4. add real permission/safety boundaries around tools
5. build a parity harness so we can improve behavior intentionally

## Method

I reviewed:

- Loader source under `src/loader/`
- Loader tests under `tests/`
- `refs/claw-code/README.md`
- `refs/claw-code/USAGE.md`
- `refs/claw-code/PARITY.md`
- `refs/claw-code/PHILOSOPHY.md`
- `refs/claw-code/rust/crates/runtime/*`
- `refs/claw-code/rust/crates/tools/src/lib.rs`
- `refs/oh-my-codex/README.md`
- `refs/oh-my-codex/AGENTS.md`
- `refs/oh-my-codex/skills/deep-interview/SKILL.md`
- `refs/oh-my-codex/skills/ralplan/SKILL.md`
- `refs/oh-my-codex/skills/ralph/SKILL.md`
- `refs/oh-my-codex/src/modes/base.ts`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/mcp/memory-server.ts`
- `refs/oh-my-codex/src/verification/verifier.ts`
- `refs/oh-my-codex/src/cli/doctor.ts`
- `refs/oh-my-codex/src/scripts/notify-hook.ts`

I also ran Loader verification commands:

- `uv run pytest`
  - failed during collection
  - discovered `refs/claw-code/tests/*`
  - also failed to import `loader`
- `uv run --with pytest --with pytest-asyncio python -m pytest tests -q`
  - 56 passed
  - 3 failed

That matters because some of Loader’s runtime paths are clearly under-tested.

## What Loader already does well

### 1. Loader is small, understandable, and hackable

This is a real advantage.

`src/loader/` is about 55 source files, and the core agent behavior is easy to locate. Compared to `claw-code` and especially OMX, Loader is much easier to refactor aggressively.

### 2. Loader is genuinely local-first

The Ollama-first posture is simple and useful. A lot of the complexity in `claw-code` and OMX comes from supporting broad operational surfaces, multiple runtimes, OAuth, MCP, tmux/team flows, and richer tool ecosystems. Loader can keep its local-first identity while still copying the good execution ideas.

### 3. Loader already contains the seeds of a better system

These are the right instincts:

- project context detection in `src/loader/context/project.py`
- runtime safeguards in `src/loader/agent/safeguards.py`
- recovery categorization in `src/loader/agent/recovery.py`
- optional decomposition / critique / confidence / verification / completion checks in `src/loader/agent/reasoning.py`
- a decent Textual app in `src/loader/ui/app.py`

The problem is not that Loader lacks ideas.

The problem is that these ideas are bolted onto one big runtime loop instead of being elevated into the architecture.

### 4. The TUI is a meaningful strength

Loader’s TUI already gives you:

- model selection
- streaming output
- approval handling
- status line updates
- tool widgets

That is more product surface than many small local agents. It is worth keeping.

## Where Loader is weak today

### 1. Loader’s product surface is not trustworthy yet

The most visible sign is the README:

- `README.md:1-2` still says “FortranGoingOnForty” and “A tutorial on using Fortran for beginners.”

That looks small, but it reflects a bigger problem: Loader is missing operational polish and self-diagnosis. `claw-code` and OMX both treat installability, health checks, and discoverability as product requirements. Loader currently feels like an experiment more than a tool.

### 2. Loader’s main runtime is too monolithic and too heuristic

`src/loader/agent/loop.py` is the heart of Loader, and it is doing too much:

- prompt construction
- streaming output handling
- raw tool-call extraction
- duplicate tool execution flows
- recovery
- validation
- rollback tracking
- completion nudging
- loop detection
- steering
- partial planning
- decomposition

The result is a loop that is hard to reason about and easy to destabilize.

The core design smell is that Loader tries to recover from model misbehavior in-place instead of enforcing a stronger turn protocol.

### 3. Loader has a real runtime contract bug in tool-result handling

**Verified directly against the code.** There is a concrete mismatch between `Message` and the loop:

- `src/loader/llm/base.py:33-39` defines `Message` with `role`, `content`, `tool_calls`, and `tool_results`. There is no `tool_call_id` field on `Message` — that field belongs to the separate `ToolResult` dataclass at `src/loader/llm/base.py:25-30`.
- `src/loader/agent/loop.py:885` and `src/loader/agent/loop.py:906` both construct `Message(role=Role.TOOL, content=..., tool_call_id=tool_call.id)`.

Both call sites will raise `TypeError: Message.__init__() got an unexpected keyword argument 'tool_call_id'` the moment they execute. They live on the duplicate-suppression and pre-validation branches of the loop, which means they have **zero** integration coverage today. This single bug is the proof that the test harness gap is real and that Sprint 00 must precede any behavioral work.

### 4. Loader duplicates tool execution logic instead of centralizing it

There are effectively two execution paths:

- the normal native/ReAct tool path
- the “raw JSON extracted tool call” path

Those paths duplicate:

- duplicate checking
- validation
- confirmation behavior
- result recording
- loop/error handling

That makes behavior inconsistent and increases the chance that fixes in one path never land in the other.

`claw-code`’s `ConversationRuntime::run_turn()` is much tighter: receive assistant output, extract tool uses, authorize, execute, append tool results, repeat.

### 5. Loader’s system prompt is too shallow and too rigid

`src/loader/agent/prompts.py:148-208` gives Loader a generic “use tools immediately / no code blocks / no numbered steps / read files before editing” prompt.

This is too blunt.

Problems:

- it treats all tasks like immediate tool-execution tasks
- it globally bans numbered steps, which is bad for planning/reporting tasks
- it does not define modes
- it does not encode verification expectations
- it does not encode completion criteria
- it does not distinguish “clarify”, “plan”, “execute”, and “verify”

OMX is much better here. It does not just say “do the task.” It routes the task into a workflow lane with an explicit contract.

### 6. Loader’s tool surface is too thin

Loader has 6 default tools:

- `read`
- `write`
- `edit`
- `glob`
- `bash`
- `grep`

That is enough for toy execution, but not enough for strong agent behavior.

What is missing compared to `claw-code` / OMX:

- task/todo tracking
- structured ask-user surfaces
- memory/notepad
- doctor/status/session tooling
- git-aware helpers
- explore vs full-execution split
- diff/patch-aware editing
- web/search/fetch surfaces
- structured output surfaces
- subagent/team coordination surfaces
- MCP-backed state and memory

The result is that Loader has to keep too much in the prompt and too much in ephemeral model state.

### 7. Loader’s safety model is primitive

Loader’s current protection model is mostly:

- “safe commands” vs “ask for confirmation”
- destructive tool flags

Problems in practice:

- no permission modes like `read-only`, `workspace-write`, `danger-full-access`
- no strong workspace boundary checks
- no binary-file guards
- no file size limits
- no symlink escape protection
- no command semantics beyond a short safe list

Evidence:

- `src/loader/tools/file_tools.py` reads/writes resolved paths directly
- `src/loader/tools/shell_tools.py` uses `create_subprocess_shell()` on arbitrary shell strings
- `src/loader/tools/shell_tools.py:13-20` uses a short safe command set, but no mode-based authorization model

By comparison, `claw-code` has:

- `PermissionPolicy`
- `PermissionEnforcer`
- workspace boundary checks
- binary/size guards in file ops
- permission-mode aware tool definitions

That does not just make it safer. It makes the agent more predictable.

### 8. Loader’s “definition of done” is heuristic, not contractual

The user complaint about “spending too long on simple tasks or finishing early without followup” is visible directly in the code.

Loader’s current strategy is:

- heuristically decide whether the response looks premature
- nudge the model to continue
- maybe ask it to confirm completion

See:

- `src/loader/agent/reasoning.py:721-854`

This is well-intentioned, but it is still guesswork.

It does not require:

- explicit acceptance criteria
- a verification plan
- fresh command evidence
- zero pending tasks
- a final sign-off phase

OMX’s `ralph` workflow does.

That difference is enormous.

### 9. Loader has no durable workflow state

Loader has plans, decomposition, and completion logic, but they live inside one run and disappear.

Missing pieces:

- persisted mode state
- session memory
- approved plan artifacts
- PRD / test-spec artifacts
- progress ledger
- durable “what was already decided”
- resume-safe task state

OMX writes state under `.omx/` and uses that to keep the workflow coherent across retries, handoffs, and interruptions. Loader currently depends on in-memory context plus prompt history only.

### 10. Loader is too backend-specific and too capability-fragile

Despite defining an abstract LLM backend, Loader is effectively Ollama-only today.

Evidence:

- `src/loader/cli/main.py` supports only `ollama`
- `src/loader/llm/ollama.py` hardcodes native tool support by model-name substring matching

This is fragile for behavior matching “with any model chosen.”

What Loader needs instead is:

- a provider-independent tool-calling contract
- explicit capability profiles
- distinct fallback strategies for native tools vs text tool calling
- prompts/workflows that degrade gracefully

### 11. Loader’s tests are not protecting the real runtime

Loader’s test suite is mostly:

- tool unit tests
- parsing tests
- recovery tests

That is useful, but insufficient.

The current state:

- `uv run pytest` fails by default after adding `refs/`
- the repo does not scope pytest discovery
- the “normal” targeted run needs `--with pytest --with pytest-asyncio`
- even then, 3 tests fail
- there are no strong turn-loop integration tests
- there is no deterministic mock backend harness comparable to `claw-code`

This is why structural issues like the `tool_call_id` mismatch can survive.

## What `claw-code` gets right

## 1. The runtime contract is explicit

`refs/claw-code/rust/crates/runtime/src/conversation.rs` is the biggest thing Loader should study.

The core `run_turn()` flow is clean:

1. append user message to session
2. stream assistant response
3. build a typed assistant message
4. extract tool uses
5. run permission checks
6. execute tool
7. append tool result message
8. repeat until no more tool uses
9. optionally compact session
10. return a typed turn summary

That is much more trustworthy than Loader’s current “stream + parse + filter + maybe reparse + maybe extract raw JSON + maybe duplicate path” approach.

## 2. Session persistence and compaction are first-class

`claw-code` treats long-lived sessions as a product feature:

- persisted sessions
- resume support
- usage tracking
- compaction thresholds
- summarized continuation messages

Relevant files:

- `refs/claw-code/rust/crates/runtime/src/conversation.rs`
- `refs/claw-code/rust/crates/runtime/src/compact.rs`
- `refs/claw-code/rust/crates/runtime/src/summary_compression.rs`
- `refs/claw-code/rust/crates/runtime/src/usage.rs`

This matters because good agent behavior is often continuity behavior.

## 3. Permissions are part of the runtime, not just UI confirmation

`claw-code` has an actual permission model with three layers:

- **Mode layer** — `PermissionMode` enum with `ReadOnly`, `WorkspaceWrite`, `DangerFullAccess`, `Prompt`, and `Allow` (`refs/claw-code/rust/crates/runtime/src/permissions.rs:8-27`)
- **Per-tool requirement layer** — every `ToolSpec` declares the minimum mode it requires, mapped in `PermissionPolicy.tool_requirements`
- **Rule layer** — three rule lists (`allow_rules`, `deny_rules`, `ask_rules`) for context-specific overrides on top of the mode/requirement check

Plus typed authorization outcomes, file-write boundary logic, and bash gating.

Relevant files:

- `refs/claw-code/rust/crates/runtime/src/permission_enforcer.rs`
- `refs/claw-code/rust/crates/runtime/src/permissions.rs`

Loader needs this badly. The mode layer alone is the high-leverage start; the rule layer can come later.

## 4. File and shell operations are engineered, not just exposed

`claw-code`’s file layer includes:

- max read size
- max write size
- binary detection
- workspace-boundary validation
- structured patch outputs

Relevant file:

- `refs/claw-code/rust/crates/runtime/src/file_ops.rs`

Loader’s file tools are functional, but too permissive and too simplistic to support strong autonomous behavior.

## 5. Hooks and lifecycle surfaces give the runtime escape valves

`claw-code` has pre-tool and post-tool hooks, including failure hooks.

That is important because not every behavioral improvement should live inside the model prompt. Hooks let the system inject policy, observability, and guardrails without changing the LLM call itself.

Relevant files:

- `refs/claw-code/rust/crates/runtime/src/hooks.rs`
- `refs/claw-code/rust/crates/runtime/src/conversation.rs`

## 6. The project is honest about parity and weaknesses

`refs/claw-code/PARITY.md` is one of the best engineering lessons in the whole comparison.

It does three things Loader does not yet do:

- names what is actually shipped
- names what is still shallow or stubbed
- ties roadmap claims to concrete evidence

That alone reduces thrash.

Loader needs a similar parity/backlog document for runtime behavior.

## 7. Diagnostics and operator surfaces are part of the product

`claw-code` exposes operational commands like:

- `status`
- `sandbox`
- `agents`
- `mcp`
- `skills`
- `doctor`
- session resume

This is not just convenience. It makes the system inspectable. Loader currently hides too much inside the runtime.

## Where `claw-code` is still incomplete

It is worth staying honest here too.

Even `claw-code` admits some shallowness in `PARITY.md`:

- some surfaces are registry-backed approximations, not deep external integrations
- session compaction parity is still open
- token accounting accuracy is still open
- some tool surfaces remain shallow or partially stubbed

That is useful because the goal is not blind imitation. The goal is to copy the parts that most affect day-to-day behavior.

## What OMX adds that Loader is currently missing almost entirely

`claw-code` gives a better runtime. OMX gives a better workflow.

This is where most of Loader’s “definition of done” and “follow-through” problems are answered.

### 1. Clarification is a mode, not an ad hoc question

`deep-interview` is not “ask a question if confused.”

It is a formal ambiguity-reduction workflow with:

- a context snapshot
- one-question rounds
- ambiguity scoring
- explicit non-goals
- explicit decision boundaries
- a crystallized artifact for downstream execution

Relevant files:

- `refs/oh-my-codex/skills/deep-interview/SKILL.md`

Loader currently has no equivalent. It either acts immediately or tries to self-nudge mid-flight.

### 2. Planning is artifact-based and consensus-based

`ralplan` is much more than “make a numbered list.”

It includes:

- Planner / Architect / Critic loops
- max iteration handling
- planning completion gates
- PRD and test-spec artifacts
- approved handoff into execution

Relevant files:

- `refs/oh-my-codex/skills/ralplan/SKILL.md`
- `refs/oh-my-codex/src/ralplan/runtime.ts`
- `refs/oh-my-codex/src/planning/artifacts.ts`

Loader’s `Plan` object is fine as a local helper, but it is nowhere near this level of control.

### 3. “Done” is a workflow contract in Ralph

This is the single biggest lesson for Loader.

Ralph encodes:

- persistence until done
- mandatory verification
- architect verification
- retry/fix loops
- state transitions
- explicit cleanup on completion
- a final checklist

Relevant file:

- `refs/oh-my-codex/skills/ralph/SKILL.md`

This directly addresses the exact Loader problems you named:

- weak tool follow-through
- finishing too early
- spending too long in loops
- poor task closure

### 4. Workflow state lives outside the prompt

OMX stores durable mode state under `.omx/` and exposes it through state tools.

Relevant files:

- `refs/oh-my-codex/src/modes/base.ts`
- `refs/oh-my-codex/src/mcp/state-server.ts`
- `refs/oh-my-codex/src/mcp/memory-server.ts`

That means:

- progress survives interruptions
- execution can be resumed
- handoffs are grounded
- context can be audited
- the model does not have to remember everything itself

### 5. Memory and notepad are explicit tools

OMX has project memory and a notepad.

That sounds small, but it matters a lot for agent stability. It gives the system somewhere to store:

- conventions
- known build commands
- temporary working notes
- durable directives

Relevant file:

- `refs/oh-my-codex/src/mcp/memory-server.ts`

Loader currently rediscovers too much per turn.

### 6. Verification is standardized

OMX has verification instructions that scale by task size and explicitly require evidence.

Relevant file:

- `refs/oh-my-codex/src/verification/verifier.ts`

Loader has completion heuristics. OMX has verification policy.

That is the difference between “the model sounded done” and “the system proved done.”

### 7. Doctor / explore / sparkshell reduce prompt waste

OMX distinguishes:

- health checking (`doctor`)
- lightweight read-only exploration (`explore`)
- bounded shell-native inspection (`sparkshell`)

That is smart.

It keeps the main execution loop from becoming the only place everything happens.

Relevant files:

- `refs/oh-my-codex/src/cli/doctor.ts`
- `refs/oh-my-codex/src/cli/explore.ts`
- `refs/oh-my-codex/src/cli/sparkshell.ts`

### 8. Follow-through is supported outside the agent context window

The idle notifications, leader nudges, and continuation prompts in OMX are important.

Relevant file:

- `refs/oh-my-codex/src/scripts/notify-hook.ts`

This is one of the deeper design differences:

- Loader tries to keep the model on-task from inside the loop
- OMX also nudges, monitors, and routes from outside the loop

That is a more robust design.

## Comparison matrix

| Area | Loader today | `claw-code` | OMX lesson | Takeaway for Loader |
|---|---|---|---|---|
| Runtime loop | monolithic, heuristic-heavy | typed turn engine | separate mode/workflow from turn runtime | split Loader runtime first |
| Tool surface | 6 basic tools | 49 exposed tool specs on main | tools should include workflow/state surfaces | add stateful and diagnostic tools |
| Permissions | confirmation-only | permission policy + enforcer | safety belongs in runtime | add modes and boundaries |
| Completion | heuristic continuation prompt | stronger runtime summaries | Ralph gives evidence-backed done gates | replace “maybe done” with explicit verification |
| Planning | ephemeral numbered list | some plan surfaces | ralplan = persisted, reviewed planning | persist plan artifacts |
| Memory/state | none | sessions + compaction + tracing | `.omx/` mode state + memory | add `.loader/` state dir |
| Diagnostics | minimal | status/sandbox/doctor/session | doctor/explore/sparkshell | make Loader inspectable |
| Testing | unit-heavy, no runtime harness | mock parity harness | workflow runtime is tested like product behavior | build scripted runtime tests |
| Extensibility | none | hooks, plugins, MCP surfaces | workflow and notification hooks | add lifecycle hooks later |
| Multi-agent | none | agent/team surfaces | team + ralph staffing | defer until solo runtime is trustworthy |

## Why Loader’s current weaknesses produce the behavior you described

### Poor tool use

Root causes:

- shallow tool surface
- brittle prompt contract
- native-vs-ReAct bifurcation
- duplicated execution code paths
- no typed runtime contract for tool results

### Weak follow-through

Root causes:

- no persistent task state
- no approved plan artifact
- no explicit verification lane
- no final completion checklist

### Finishing early

Root causes:

- completion is heuristic
- no required evidence model
- no acceptance criteria artifact
- no final “prove it” pass

### Spending too long on simple tasks

Root causes:

- the runtime loop tries too many recoveries in one place
- the system prompt does not distinguish task modes cleanly
- there is no “lightweight inspect” lane like `explore`
- the model often has to infer the workflow instead of being routed into one

### Model sensitivity

Root causes:

- behavior is prompt-and-heuristic driven
- capability detection is backend-specific and brittle
- no workflow artifacts that survive model variance

This is why copying OMX’s workflow ideas is so high leverage. It reduces how much we ask the model to improvise.

## Concrete implementation targets

These are ordered by impact on Loader behavior, not by code convenience.

### Target 1: Introduce a real turn engine

Goal:

- replace the current giant loop with a smaller, typed conversation runtime

Implementation target:

- create a new `src/loader/runtime/` package
- move message/session/tool-result logic out of `src/loader/agent/loop.py`
- give tool results a first-class typed representation
- unify native, ReAct, and extracted-tool execution through one executor path

Why:

- this is the foundation for every other improvement

### Target 2: Add persistent Loader state under `.loader/`

Goal:

- make workflow state durable instead of prompt-only

Implementation target:

- `.loader/state/`
- `.loader/sessions/`
- `.loader/plans/`
- `.loader/notepad.md`
- `.loader/project-memory.json`

Why:

- Loader needs somewhere to store progress, acceptance criteria, and recovered knowledge

### Target 3: Separate task modes

Goal:

- stop treating all requests like immediate tool-execution requests

Implementation target:

- mode router with at least:
  - `clarify`
  - `plan`
  - `execute`
  - `verify`

Why:

- this is the minimum structure needed to stop overthinking simple work and underthinking complex work

### Target 4: Replace heuristic completion with an evidence-backed done contract

Goal:

- make completion explicit and testable

Implementation target:

- define a `DefinitionOfDone` object per task
- require:
  - acceptance criteria
  - verification commands
  - evidence summary
  - zero pending task items

Why:

- this is the main fix for premature completion

### Target 5: Add `deep-interview`-lite and `ralplan`-lite equivalents

Goal:

- pull ambiguity reduction and planning review out of the middle of execution

Implementation target:

- `clarify` mode writes a task brief
- `plan` mode writes:
  - a short implementation plan
  - a test/verification plan

Do not try to copy every OMX feature immediately. Copy the artifact discipline first.

### Target 6: Build a real permission model

Goal:

- move from confirmation prompts to policy-based authorization

Implementation target:

- permission modes:
  - `read-only`
  - `workspace-write`
  - `danger-full-access`
- tool specs declare required permission
- file writes enforce workspace boundaries
- shell commands go through command classification

Why:

- this is both safety and behavior quality

### Target 7: Harden file and shell tools

Goal:

- make tool use trustworthy enough for automation

Implementation target:

- size limits
- binary detection
- symlink/traversal protection
- structured patch/diff return values
- shell command semantics and mutability classification

### Target 8: Add `loader doctor`, `loader status`, and `loader session`

Goal:

- make Loader operable as a product

Implementation target:

- backend health
- model capability snapshot
- workspace detection
- write-access detection
- test/build command detection
- active session summary

Why:

- better operator feedback means less guesswork in the agent loop

### Target 9: Add memory/notepad tools

Goal:

- give Loader durable short-term and long-term memory

Implementation target:

- read/write project memory
- append working notes
- store user directives and repo conventions

Why:

- this reduces re-discovery and improves follow-through across turns

### Target 10: Add a lightweight read-only inspect lane

Goal:

- avoid using the full agent loop for every lookup

Implementation target:

- `loader explore` or equivalent internal mode
- optimized for:
  - file/symbol lookup
  - pattern discovery
  - relationship questions

Why:

- simple tasks should stay cheap and fast

### Target 11: Add a parity harness

Goal:

- improve behavior intentionally instead of impressionistically

Implementation target:

- scripted mock backend scenarios for:
  - simple read
  - multi-tool turn
  - denied permission
  - write/edit success
  - verification-required task
  - premature completion rejection
  - looped/duplicate action prevention

Why:

- this is how Loader becomes reliable

### Target 12: Add workflow-aware prompts and capability profiles

Goal:

- make Loader less brittle across models

Implementation target:

- replace one generic system prompt with mode-specific prompts
- add provider/model capability profiles:
  - native tools
  - streaming
  - context budget
  - preferred tool-call format
  - verification strictness

Why:

- behavior should be shaped by runtime policy, not guessed from model substrings

## Priority order

This section was rewritten after a deeper validation pass against the actual code in `refs/claw-code` and `refs/oh-my-codex`, plus firsthand spot-checks of Loader's runtime. The deeper review confirmed every load-bearing claim in this report and surfaced one structural reorder: **the Definition-of-Done work is the user's actual pain point and should land before permission modes**, not after, because permissions are a safety win and DoD is the behavior win.

### P0: Stabilize before changing behavior (Sprint 00)

- write a failing regression test for the `tool_call_id` bug at `agent/loop.py:885,906` *first*, before any harness work — it proves the bug is real and proves the harness exists in one move
- scope pytest discovery so `refs/` stops contaminating collection
- exclude `refs/` from ruff and mypy too
- make `uv run pytest` work out of the box
- port the scenario taxonomy from `refs/claw-code/rust/crates/rusty-claude-cli/tests/mock_parity_harness.rs`
- rewrite `README.md` (currently still says "FortranGoingOnForty")
- baseline parity checklist for current runtime behavior

### P1: Replace the loop with a real runtime (Sprint 01)

- new `src/loader/runtime/` package with a typed turn engine
- unify the native, ReAct, and "extracted JSON fallback" tool execution paths into one executor
- fix the named bugs from Sprint 00's failing tests (`tool_call_id`, duplicate execution path)
- replace substring-based `NATIVE_TOOL_MODELS`/`NO_TOOL_MODELS` model detection with a `runtime/capabilities.py` profile system — Loader needs to behave consistently across model choices
- structured `TurnSummary` output

### P2: The behavior fix the user actually asked for (Sprint 02)

- `DefinitionOfDone` object per task: acceptance criteria, verification commands, evidence summary, pending/completed task items
- explicit verify phase that runs the verification commands and gates completion on evidence
- fix loop: verification failure returns to execution, not to final answer
- minimum `.loader/` directory shape (`.loader/dod/`) — full session/memory layout deferred to Sprint 05

This is the highest-leverage behavioral change in the entire plan and is the direct answer to "finishing too early" and "weak follow-through."

### P3: Safety as policy, not as confirmation prompt (Sprint 03)

- permission modes: `read-only`, `workspace-write`, `danger-full-access`
- three-event tool lifecycle hooks (`pre_tool_use`, `post_tool_use`, `post_tool_use_failure`) modeled directly on `refs/claw-code/rust/crates/runtime/src/hooks.rs`
- refactor `safeguards.py` (duplicate detection, validation, rollback) into pre-tool hook implementations rather than ad-hoc method calls
- file operation hardening (workspace boundary, symlink, size limits, binary detection, structured patches)
- shell operation hardening
- expose active mode in CLI/TUI status

Hooks land alongside permissions because every later sprint hangs new behavior (verification, validation, observability) on the same lifecycle.

### P4: Stop improvising one workflow for everything (Sprint 04)

- mode router: clarify, plan, execute, verify (verify already exists from Sprint 02)
- clarify artifact written to `.loader/briefs/`
- planning artifacts (implementation plan + verification plan) written to `.loader/plans/` and fed into the existing DoD object
- tool prerequisites pulled forward from Sprint 06: `TodoWrite` (the "zero pending tasks" gate is empty without it) and `AskUserQuestion` (clarify rounds)

### P5: Durable continuity (Sprint 05)

- full `.loader/` state directory under the layout already started in Sprint 02
- session persistence and resume
- transcript compaction with priority-aware summarization (model the design on `refs/claw-code/rust/crates/runtime/src/summary_compression.rs`)
- memory/notepad surfaces
- usage/cost tracking

### P6: Operability and tool-surface expansion (Sprint 06)

- `loader doctor`, `loader status`, `loader session`
- read-only explore lane
- broader tool surface (diff/patch-aware editing, git helpers, structured ask-user, etc.) — `TodoWrite` and `AskUserQuestion` already exist from Sprint 04

### Deferred indefinitely

- workflow hooks beyond the runtime tool lifecycle (notification/idle nudges, leader monitoring)
- task/team/subagent orchestration
- broad MCP ecosystem
- richer plugin systems

These are real wins in `claw-code`/OMX, but Loader should not pursue them until the solo runtime is trustworthy.

## What Loader should copy directly, and what it should not

### Copy directly

- typed turn runtime
- permission model
- file/shell hardening
- session persistence
- compaction
- doctor/status/session surfaces
- workflow artifacts
- evidence-backed verification
- parity harness discipline

### Copy in simplified form

- deep-interview
- ralplan
- ralph
- memory/notepad
- explore vs full-execution split

### Do not copy blindly yet

- full tmux/team runtime
- huge command surface
- Discord/openclaw notification stack
- broad MCP ecosystem

Loader should first become a trustworthy single-agent local runtime. After that, team orchestration will actually help.

## Recommended Loader architecture direction

If we want behavior closer to `claw-code` without losing Loader’s simplicity, I would steer toward:

### Layer 1: Runtime core

- typed `TurnRuntime`
- `SessionStore`
- `PermissionPolicy`
- `ToolExecutor`
- `VerificationEngine`

### Layer 2: Workflow layer

- `ClarifyWorkflow`
- `PlanWorkflow`
- `ExecuteWorkflow`
- `VerifyWorkflow`

### Layer 3: Product surfaces

- TUI
- CLI
- `doctor`
- `status`
- `session`
- `explore`

### Layer 4: Optional future orchestration

- hooks
- background verification
- multi-agent/task orchestration

That is a better fit for Loader than trying to clone all of OMX wholesale.

## Immediate conclusions

1. Loader’s biggest problems are architectural, not just prompt-related.
2. `claw-code` is strongest where Loader is weakest: runtime contract, permissions, sessions, diagnostics, parity.
3. OMX is strongest where Loader is currently almost absent: clarification, planning discipline, durable state, completion/verification loops.
4. The fastest path to “better model behavior today” is not adding more heuristics. It is adding:
   - workflow artifacts
   - explicit verification
   - persistent state
   - a smaller, more trustworthy turn engine

## Sprint scaffolding

After the deeper validation pass the original five-sprint plan was reshaped into seven sprints. The reshape splits the most ambitious sprint (the old Sprint 03, which bundled mode router + clarify + plan + DoD + verify/fix into one) and reorders so the user's actual pain point lands sooner. Sprint scaffolding lives under:

- `.docs/sprints/index.md`
- `.docs/sprints/sprint00.md` — Foundation, Measurement, and Parity Harness
- `.docs/sprints/sprint01.md` — Turn Engine, Tool Contract, and Capability Profiles
- `.docs/sprints/sprint02.md` — Definition of Done and Verify/Fix Loop
- `.docs/sprints/sprint03.md` — Permission Modes and Tool Lifecycle Hooks
- `.docs/sprints/sprint04.md` — Mode Router, Clarify, and Plan Artifacts
- `.docs/sprints/sprint05.md` — Session State, Memory, and Compaction
- `.docs/sprints/sprint06.md` — Doctor, Explore, Status, and Tool Surface Expansion

## Recommended next move

Start with Sprint 00, and start Sprint 00 with the failing regression test.

Reason:

- Loader needs a measurable baseline and a safer runtime before adding more behavior
- the `tool_call_id` bug at `agent/loop.py:885,906` is proof that untested code paths are silently broken
- writing the failing test first proves both the bug and the harness in one move
- otherwise every feature sprint will be built on unstable agent semantics

The execution phase should then be:

1. lock down the runtime and test harness (Sprint 00)
2. replace the loop with a typed runtime and capability profiles (Sprint 01)
3. define and enforce the completion contract (Sprint 02)
4. add the policy-based safety layer with hooks (Sprint 03)
5. add workflow modes and planning artifacts on top (Sprint 04)
6. then widen the durability and product surfaces (Sprints 05 and 06)

## Plan adjustments after deeper review

The following changes were applied to the original report after a firsthand validation pass against the actual code in `refs/claw-code` and `refs/oh-my-codex`, plus spot-checks of Loader's runtime.

### Verified directly against the code

- **`tool_call_id` bug confirmed at `src/loader/agent/loop.py:885` and `:906`.** Both call sites construct `Message(role=Role.TOOL, content=..., tool_call_id=tool_call.id)`, but `Message` (`src/loader/llm/base.py:33-39`) has no such field. They live on the duplicate-suppression and pre-validation branches and would crash on first execution. Zero integration coverage.
- **Pytest discovery is broken by default.** `uv run pytest --collect-only` picks up `refs/claw-code/tests/test_porting_workspace.py` and fails to import `loader` because there is no `tool.pytest.ini_options` block in `pyproject.toml`.
- **Loop monolith confirmed by line counts.** `agent/loop.py` is 1929 LOC, `agent/reasoning.py` is 1196, `agent/safeguards.py` is 1079 — roughly 4200 lines of orchestration in one cluster.
- **claw-code's `run_turn()` shape** is exactly as the report describes. Read directly at `refs/claw-code/rust/crates/runtime/src/conversation.rs:295-470`. Typed message build → tool extraction → pre-hook → permission check → execute → post-hook (success or failure variant) → typed `ConversationMessage::tool_result()` → push → repeat. ~175 lines of clean code.
- **claw-code permission modes** are `ReadOnly` / `WorkspaceWrite` / `DangerFullAccess` (plus `Prompt` and `Allow`), defined at `refs/claw-code/rust/crates/runtime/src/permissions.rs:8-27`. The 10MB read/write caps, binary detection, workspace boundary check, and structured patch outputs in `file_ops.rs` are all real.
- **claw-code hooks** are `PreToolUse` / `PostToolUse` / `PostToolUseFailure`, defined at `refs/claw-code/rust/crates/runtime/src/hooks.rs:19-34` and wired into the conversation loop at lines 371, 427-453.
- **OMX skills are real and even more rigorous than the report described.** `ralplan` enforces a max-5-iteration Critic loop with sequential Architect→Critic ordering. `ralph` has explicit phase enums (`starting`/`executing`/`verifying`/`fixing`/`complete`/`failed`/`cancelled`) persisted via `state_write` to `.omx/state/{mode}-state.json`. The verifier in `src/verification/verifier.ts` scales by task size with concrete file-count thresholds.

### Corrected facts

- **Tool count: 49, not 40.** `refs/claw-code/rust/crates/tools/src/lib.rs` exposes 49 `ToolSpec` entries in `mvp_tool_specs()`. Doesn't change the lesson, but worth knowing.
- **claw-code permissions have a third layer.** Beyond `PermissionMode` and per-tool requirements, `PermissionPolicy` carries three rule lists (`allow_rules`, `deny_rules`, `ask_rules`) for context-specific overrides. Loader can land the mode layer first and defer the rule layer.
- **claw-code summary compression is sophisticated.** It's not message-level truncation — it's line-level prioritization with deduplication and budget enforcement at `refs/claw-code/rust/crates/runtime/src/summary_compression.rs`. Sprint 05 should model on this rather than reinventing.

### Structural plan changes

- **The old Sprint 03 was split.** It bundled mode router + clarify + plan + DoD + verify/fix into one sprint, which is essentially "ralplan + ralph + verifier" simultaneously. The DoD/verify-fix half became the new Sprint 02 (highest-leverage behavioral fix). The mode router / clarify / plan half became the new Sprint 04.
- **The old Sprint 02 (permissions) became the new Sprint 03** and was reordered to land *after* DoD. Permissions are a safety win, not a behavior win, and the user's actual complaints are about behavior. DoD lands first.
- **Hooks landed in the same sprint as permissions.** The original plan split them across sprints; that creates rework because every later runtime addition (verification, observability, validation) wants the same lifecycle. Sprint 03 owns both.
- **Capability profiles became a Sprint 01 deliverable.** They were Target 12 in the original report and orphaned from the sprint plan. They belong in the runtime layer and are critical for the user's "behave consistently across model choices" goal.
- **The minimum `.loader/` directory shape moves to Sprint 02** (just `.loader/dod/`). The full session/memory/compaction layout stays in Sprint 05. This unblocks Sprint 02 and Sprint 04 from waiting on Sprint 05.
- **`TodoWrite` and `AskUserQuestion` move from Sprint 06 to Sprint 04** as prerequisites for the clarify mode and the "zero pending tasks" gate. The broad tool-surface expansion stays in Sprint 06.
- **Sprint 00's first deliverable is now the failing regression test** for the `tool_call_id` bug, before any harness work. It proves the bug and proves the harness exist in one move.
