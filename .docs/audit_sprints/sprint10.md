# Sprint 10: Runtime Context and Ownership Inversion

## Prerequisites

Sprint 09

## Goals

Make the runtime a real runtime instead of a set of helper classes that reach back into `Agent`.

Sprint 01 split the loop across files, and Sprints 07-08 continued that split. The next step is to replace `agent: Any` with a typed boundary so runtime correctness no longer depends on reaching into `self.agent.*` from every helper.

## Deliverables

### 1. Introduce a typed runtime context

Add a typed context layer under `src/loader/runtime/` that owns the state and services the runtime actually needs, such as:

- session access
- backend access
- registry and tool schemas
- permission policy and hook manager
- workflow mode and capability profile
- prompt-building inputs
- runtime-scoped callbacks/services that still need legacy behavior during migration

The key rule is that runtime helpers accept this context, not `Agent`.

### 2. Migrate runtime helpers off `self.agent`

Migrate at least these modules to the new context boundary:

- `runtime/conversation.py`
- `runtime/assistant_turns.py`
- `runtime/tool_batches.py`
- `runtime/finalization.py`
- `runtime/repair.py`
- `runtime/completion_policy.py`
- `runtime/explore.py`

During this sprint it is acceptable for the `Agent` to construct the context and act as an adapter. It is not acceptable for runtime modules to keep reaching through that adapter directly.

### 3. Introduce typed service seams for legacy dependencies

Where runtime logic still depends on legacy behavior, surface that dependency explicitly as a typed service or callback protocol instead of an object reach-in. Likely seams include:

- self-critique
- loop detection
- text-loop detection
- action verification and recovery bookkeeping
- any remaining stream filtering or steering hooks

This keeps Sprint 11 focused on deleting heuristics rather than disentangling call sites.

### 4. Isolation-friendly runtime tests

Add or update tests so the migrated runtime helpers can be exercised with lightweight fake contexts instead of full `Agent` instances.

## Commit slicing

- one commit for the new runtime-context types and adapters
- one commit per migrated runtime module or closely related module pair
- one commit for test harness/fake-context support
- one commit for any follow-up cleanup that removes obsolete `agent: Any` plumbing

## Testing strategy

- `uv run pytest -q`
- unit coverage for runtime-context construction and validation
- focused tests showing migrated helpers work with fake contexts
- regression coverage proving session state, permission state, and phase state still flow through the runtime correctly

## Definition of done

- `src/loader/runtime/` no longer treats `Agent` as its primary data model
- the major runtime helpers run on typed context/service inputs instead of `self.agent.*`
- `agent: Any` is removed from runtime constructors in the core turn path
- the remaining legacy dependencies are explicit and typed instead of hidden object reach-ins

## Explicitly out of scope

- deleting recovery heuristics en masse
- redesigning the raw-text parser
- clarify/plan protocol changes
- large-scale legacy-file breakup beyond the context boundary needed for runtime ownership
