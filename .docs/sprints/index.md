# Loader Sprint Index

These sprints translate the `REPORT.md` findings into implementation lanes with clear deliverables, test strategy, and definition-of-done checkpoints.

The plan was reshaped after a deeper validation pass against `refs/claw-code` and `refs/oh-my-codex`. The reshape (a) splits the most ambitious sprint, (b) reorders so the user's actual pain point (premature completion, weak follow-through) lands sooner, and (c) lands hooks alongside permissions so later sprints have a clean lifecycle to hang behavior on.

## Phase 1: Runtime Foundation

- [Sprint 00](sprint00.md) — Foundation, Measurement, and Parity Harness
- [Sprint 01](sprint01.md) — Turn Engine, Tool Contract, and Capability Profiles

## Phase 2: Behavioral Contract

- [Sprint 02](sprint02.md) — Definition of Done and Verify/Fix Loop

## Phase 3: Safety and Workflow Discipline

- [Sprint 03](sprint03.md) — Permission Modes and Tool Lifecycle Hooks
- [Sprint 04](sprint04.md) — Mode Router, Clarify, and Plan Artifacts

## Phase 4: Durability and Product Surfaces

- [Sprint 05](sprint05.md) — Session State, Memory, and Compaction
- [Sprint 06](sprint06.md) — Doctor, Explore, Status, and Tool Surface Expansion

## Phase 5: Execution Policy and Runtime Simplification

- [Sprint 07](sprint07.md) — Rule-Based Permissions and Runtime Decomposition

## Phase 6: Prompt Contract and Operator Ergonomics

- [Sprint 08](sprint08.md) — Prompt Builder, Runtime Phases, and Permission Operator UX

## Phase 7: Workflow State Discipline

- [Sprint 09](sprint09.md) — Turn State Machine, Workflow Contracts, and Prompt Preview

## Working principles

- Each sprint must end with stronger runtime reliability, not just more features.
- Prefer behavior that improves any capable model over model-specific prompting tricks.
- Add new workflow/state surfaces only when they reduce prompt pressure or improve verification.
- No sprint is complete until its behavior is covered by automated tests or a deterministic harness.
- When adding lifecycle behavior (validation, dedup, verification, observability), prefer hooking into the tool lifecycle from Sprint 03 over patching the runtime loop directly.
