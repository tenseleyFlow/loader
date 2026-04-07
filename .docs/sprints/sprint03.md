# Sprint 03: Permission Modes and Tool Lifecycle Hooks

## Prerequisites

Sprint 02

## Goals

Move Loader from confirmation-only safety to policy-based runtime safety, and introduce the tool lifecycle hooks that all subsequent runtime behavior should hang on.

The two deliverable groups land together because they share the same code path. claw-code's `conversation.rs:370-453` shows the pattern: every tool call flows through pre-hook → permission check → execute → post-hook (success or failure variant). Loader needs the same shape so that Sprint 04's mode router and Sprint 05's session/memory work plug into a stable lifecycle instead of patching `loop.py` again.

This sprint deliberately runs *after* the DoD work in Sprint 02, because permissions are a safety win, not a behavior win, and the user asked for behavior first.

## Deliverables

### 1. Permission modes

Add explicit runtime permission modes mirroring `refs/claw-code/rust/crates/runtime/src/permissions.rs:8-27`:

- `read-only`
- `workspace-write`
- `danger-full-access`

(claw-code also has `Prompt` and `Allow` variants — Loader can defer those.)

Each tool declares the minimum permission level it requires via a `required_permission` attribute on `Tool`. The default registry's tools map roughly to:

- `read-only`: `read`, `glob`, `grep`
- `workspace-write`: `write`, `edit`
- `danger-full-access`: `bash`

### 2. PermissionPolicy

A `PermissionPolicy` object owns the active mode and the per-tool requirement map:

```python
@dataclass
class PermissionPolicy:
    active_mode: PermissionMode
    tool_requirements: dict[str, PermissionMode]
    workspace_root: Path
```

The rule layer (`allow_rules` / `deny_rules` / `ask_rules`) from claw-code is **deferred**. Sprint 03 only needs modes and tool requirements; rules can come later if they prove necessary.

### 3. Tool lifecycle hooks — three events

Add a three-event hook lifecycle modeled directly on `refs/claw-code/rust/crates/runtime/src/hooks.rs:19-34`:

- `pre_tool_use` — runs before authorization; can override input, deny, or inject messages
- `post_tool_use` — runs after successful execution; can modify output or add follow-up messages
- `post_tool_use_failure` — runs after a tool error; separate from `post_tool_use` so failure handling does not have to branch on `is_error`

The `runtime.executor.ToolExecutor` from Sprint 01 calls hooks at each lifecycle point. Hook results are typed (allow / deny / cancel / fail / inject-message / override-permission) and the executor merges hook feedback into the tool result message.

This is the most important architectural piece in the sprint, because Sprints 04, 05, and 06 will all want to add lifecycle behavior. Without hooks, every later sprint would patch `loop.py` again.

### 4. Refactor safeguards.py into hook implementations

`src/loader/agent/safeguards.py` (1079 LOC) currently does duplicate detection, validation, and rollback tracking via ad-hoc method calls inside `loop.py`. Refactor each of those into a `pre_tool_use` hook implementation:

- `DuplicateActionHook` — what `safeguards.check_duplicate()` does today
- `ActionValidationHook` — what `safeguards.validate_action()` does today
- `RollbackTrackingHook` — what `loop.py:917-935` does today

The streaming filter (`CodeBlockFilter`) is a separate concern and stays for now, but Sprint 03 should mark it as "candidate for removal once the typed runtime makes the leakage it filters impossible."

### 5. File operation hardening

Match the safety guards in `refs/claw-code/rust/crates/runtime/src/file_ops.rs`:

- workspace boundary enforcement (with `canonicalize()` before the boundary check, to defeat symlink escapes)
- file size limits (10 MB read, 10 MB write — same as claw-code's `MAX_READ_SIZE` / `MAX_WRITE_SIZE`)
- binary file detection (NUL byte in the first 8 KB)
- structured patch metadata for edits/writes (return `StructuredPatchHunk` data alongside the human-readable diff)

These guards live in the file tools themselves, not in hooks — they are intrinsic to the operation, not policy.

### 6. Shell operation hardening

- command mutability classification (read-only vs mutating, used by the `read-only` mode policy)
- read-only safe-command policy
- prompt-mode authorization path
- structured stderr/exit-code result
- output truncation with metadata when output exceeds a budget

### 7. CLI/TUI visibility

Expose the active permission mode in:

- the TUI status line
- the non-TUI CLI startup banner
- `loader status` (which lands in Sprint 06, but wire the data source now)

Show the mode with a color hint: green for `read-only`, yellow for `workspace-write`, red for `danger-full-access`.

## Testing strategy

- `read-only` mode denies writes and mutating shell commands; verify via the unified executor and the hook lifecycle
- `workspace-write` allows in-repo changes but denies writes outside `workspace_root`
- `danger-full-access` allows everything
- file boundary tests cover `../`, symlink escape (canonicalized first), binary, and oversize cases
- shell tests cover read-only safe-command policy and output truncation
- hook lifecycle tests assert all three events fire in order, and that a `pre_tool_use` deny short-circuits execution but still produces a typed tool-result message
- the refactored `DuplicateActionHook` / `ActionValidationHook` / `RollbackTrackingHook` produce the same observable behavior as the old `safeguards.py` paths (use Sprint 00's harness scenarios as the regression suite)
- verifying that hooks compose: a `pre_tool_use` deny + a `post_tool_use_failure` hook still emits exactly one tool-result message

## Definition of done

- Loader has explicit permission modes
- file and shell safety rules are enforced in the runtime, not in the UI
- the three-event tool lifecycle is in place and `safeguards.py` has been refactored into hook implementations
- the streaming filter is annotated as deprecated-pending-removal
- the safety behavior is covered by automated tests
- the CLI/TUI surfaces the active mode
- Sprint 04, 05, and 06 have a clean lifecycle to plug into instead of patching `loop.py`

## Audit Notes

Audit checkpoint on 2026-04-06:

- added `PermissionMode`, `PermissionPolicy`, and lazy runtime exports under `src/loader/runtime/permissions.py` and `src/loader/runtime/__init__.py`
- refactored tool execution so `ToolExecutor` now runs hooks before and after policy evaluation in `src/loader/runtime/executor.py`
- added lifecycle hook infrastructure in `src/loader/runtime/hooks.py`, including `DuplicateActionHook`, `ActionValidationHook`, `RollbackTrackingHook`, and a success-side action-history hook for loop/dedup tracking
- hardened file and search tools with canonicalized workspace-root enforcement, symlink escape blocking, binary detection, file-size limits, and structured patch metadata
- hardened shell execution with permission classification, structured truncation metadata, and mode-aware authorization
- surfaced the active permission mode in the CLI startup banner and the TUI status line, and wired `Agent.active_permission_mode` as the current data source for later status/session work
- full verification is green at `uv run pytest -q` with 106 passing tests

Residual debt after Sprint 03:

- Loader now has mode-based permission policy, but the richer rule system (`allow` / `deny` / `ask`) is still deferred
- destructive operations still flow through the legacy confirmation path after policy allows them, so Loader has not fully matched claw-code's prompt/allow permission model
- shell mutability classification is still heuristic and intentionally conservative rather than deeply semantic
