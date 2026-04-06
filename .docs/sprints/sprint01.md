# Sprint 01: Turn Engine, Tool Contract, and Capability Profiles

## Prerequisites

Sprint 00

## Goals

Replace Loader's current monolithic loop with a smaller, typed, explicit turn engine, fix the structural bugs Sprint 00's tests are catching, and stop guessing model capabilities from substring matches.

The reference for this sprint is `refs/claw-code/rust/crates/runtime/src/conversation.rs:295-470` — about 175 lines that do what `agent/loop.py` takes ~1500 lines to half-do. Loader's runtime should aim for that shape, not that language.

## Deliverables

### 1. New runtime package

Create a dedicated runtime layer:

```text
src/loader/runtime/
├── conversation.py      # the typed turn engine (analog of conversation.rs)
├── session.py           # message history, ownership of the conversation state
├── executor.py          # the unified tool execution path (see deliverable 2)
├── events.py            # the typed AgentEvent surface, moved out of agent/loop.py
├── tracing.py           # debug/observability hooks
└── capabilities.py      # see deliverable 4
```

The new runtime package owns runtime correctness. `agent/loop.py` becomes a thin orchestration layer that builds prompts and dispatches to `runtime.conversation.run_turn()`. Reasoning helpers stay in `agent/reasoning.py` but are *called* by the runtime, not embedded inside its loop.

### 2. Unified tool execution path

Loader currently has two effectively distinct execution paths:

- the main native/ReAct path
- the "raw extracted JSON tool call" fallback path (used when the model leaks tool syntax through streaming)

These paths duplicate confirmation, validation, dedup, and result-recording logic. Fixes in one rarely land in the other.

Sprint 01 collapses both into a single `runtime.executor.ToolExecutor`. The executor owns:

- authorization (delegating to whatever permission contract exists today; the policy layer arrives in Sprint 03)
- duplicate suppression
- tool execution
- tool-result message construction (using the corrected `Message` schema from deliverable 3)
- tracing
- error classification

There is exactly one path tool calls flow through, regardless of how they were extracted from the model output.

### 3. Correct tool-result message model — fix the named bug

**Bug A** (caught by Sprint 00's failing regression test): `src/loader/agent/loop.py:885` and `:906` construct `Message(role=Role.TOOL, content=..., tool_call_id=tool_call.id)` against a `Message` dataclass at `src/loader/llm/base.py:33-39` that has no `tool_call_id` field. Sprint 00 wrote the regression test; Sprint 01 makes it pass.

The fix is *not* to add a `tool_call_id` kwarg to `Message`. The fix is to introduce an explicit tool-result message representation — either a `ToolResultMessage` class or a richer `Message` schema where tool-result rows carry a typed `ToolResult` payload (the existing `ToolResult` dataclass at `src/loader/llm/base.py:25-30` already has `tool_call_id`, `content`, `is_error` and is the right shape).

**Bug B** (uncovered by deliverable 2): the duplicate execution path. Same fix mechanism — there is one executor, so there is one place that constructs tool-result messages.

The Sprint 00 regression tests gate this work. They must turn green before the sprint can close.

### 4. Capability profiles — replace substring-based model detection

Loader currently decides whether to use native tool calling or ReAct prompting by substring-matching against two hard-coded sets in `src/loader/llm/ollama.py`:

```python
NATIVE_TOOL_MODELS = {"llama3.1", "llama3.2", ...}
NO_TOOL_MODELS = {"phi", "gemma", ...}
```

This is brittle. Adding a new model means editing source. Models that should work do not, and vice versa. The user explicitly wants Loader to behave consistently across model choices.

Replace this with a `runtime/capabilities.py` module that defines a `CapabilityProfile` dataclass:

- `supports_native_tools: bool`
- `supports_streaming: bool`
- `context_window: int`
- `preferred_tool_call_format: Literal["native", "json_tag", "bracket"]`
- `verification_strictness: Literal["lax", "standard", "strict"]`
- `notes: list[str]`

Profiles are resolved by:

1. explicit user override (CLI flag or config file)
2. exact model-name match in a built-in registry
3. heuristic fallback (probe `/api/show` from Ollama, inspect `details.families`, fall back to safe defaults)

The runtime asks the profile what to do; it never substring-matches model names.

### 5. Turn summary output

Each completed turn produces a structured `TurnSummary` containing:

- assistant messages
- tool results
- iterations
- failures
- verification status (filled in by Sprint 02)
- usage metadata if available

Modeled on `refs/claw-code/rust/crates/runtime/src/conversation.rs:110-117` (`TurnSummary` struct).

## Testing strategy

- Sprint 00's `tool_call_id` regression test passes
- Sprint 00's duplicate-suppression scenario passes through the unified executor
- new integration tests cover native-tool turns and ReAct-style turns going through the same executor
- regression tests prove that extracted-fallback and native calls share one path (e.g., assert the same trace events fire from both entry points)
- capability profiles have unit tests for the resolution priority order
- a `TurnSummary` smoke test asserts the structured output is populated for a multi-tool turn

## Definition of done

- `agent/loop.py` is no longer the sole owner of runtime correctness — it delegates to `runtime.conversation`
- tool execution logic is centralized in `runtime.executor`
- the message schema is internally consistent and the named `tool_call_id` bug is fixed
- capability profiles replace substring-based model detection
- full-turn tests cover the critical runtime paths
- the parity checklist from Sprint 00 reflects the new state
