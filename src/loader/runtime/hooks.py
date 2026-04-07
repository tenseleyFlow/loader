"""Tool lifecycle hooks for Loader runtime execution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ..agent.reasoning import RollbackPlan, create_rollback_plan_for_action, is_destructive_tool
from ..agent.safeguards import ActionTracker, PreActionValidator
from ..llm.base import ToolCall
from ..tools.base import Tool, ToolRegistry
from ..tools.base import ToolResult as RegistryToolResult
from .memory import MemoryStore
from .permissions import PermissionOverride, PermissionPolicy


class HookEvent(StrEnum):
    """Lifecycle hook events for one tool call."""

    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"


class HookDecision(StrEnum):
    """Terminal and non-terminal hook decisions."""

    CONTINUE = "continue"
    DENY = "deny"
    CANCEL = "cancel"
    FAIL = "fail"


@dataclass(slots=True)
class HookContext:
    """Context passed to hook implementations."""

    tool_call: ToolCall
    tool: Tool | None
    registry: ToolRegistry
    permission_policy: PermissionPolicy
    source: str
    skip_duplicate_check: bool = False
    record_action: bool = True
    result: RegistryToolResult | None = None
    output: str | None = None
    is_error: bool = False


@dataclass(slots=True)
class HookResult:
    """Result from one hook invocation."""

    decision: HookDecision = HookDecision.CONTINUE
    message: str | None = None
    injected_messages: list[str] = field(default_factory=list)
    permission_override: PermissionOverride | None = None
    permission_reason: str | None = None
    updated_arguments: dict[str, Any] | None = None
    output_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal_state: str | None = None


@dataclass(slots=True)
class HookRunSummary:
    """Aggregated result across all hooks for one lifecycle stage."""

    tool_call: ToolCall
    decision: HookDecision = HookDecision.CONTINUE
    message: str | None = None
    injected_messages: list[str] = field(default_factory=list)
    permission_override: PermissionOverride | None = None
    permission_reason: str | None = None
    output_override: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    terminal_state: str | None = None


class ToolHook(Protocol):
    """Protocol for async tool lifecycle hooks."""

    async def pre_tool_use(self, context: HookContext) -> HookResult: ...

    async def post_tool_use(self, context: HookContext) -> HookResult: ...

    async def post_tool_use_failure(self, context: HookContext) -> HookResult: ...


class BaseToolHook:
    """Default no-op implementation for tool hooks."""

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        return HookResult()

    async def post_tool_use(self, context: HookContext) -> HookResult:
        return HookResult()

    async def post_tool_use_failure(self, context: HookContext) -> HookResult:
        return HookResult()


class HookManager:
    """Runs tool hooks across Loader's three lifecycle events."""

    def __init__(self, hooks: Iterable[ToolHook] | None = None) -> None:
        self.hooks = list(hooks or [])

    async def run_pre_tool_use(self, context: HookContext) -> HookRunSummary:
        return await self._run_event(HookEvent.PRE_TOOL_USE, context)

    async def run_post_tool_use(self, context: HookContext) -> HookRunSummary:
        return await self._run_event(HookEvent.POST_TOOL_USE, context)

    async def run_post_tool_use_failure(self, context: HookContext) -> HookRunSummary:
        return await self._run_event(HookEvent.POST_TOOL_USE_FAILURE, context)

    async def _run_event(
        self,
        event: HookEvent,
        context: HookContext,
    ) -> HookRunSummary:
        summary = HookRunSummary(tool_call=context.tool_call)
        for hook in self.hooks:
            if event == HookEvent.PRE_TOOL_USE:
                result = await hook.pre_tool_use(context)
            elif event == HookEvent.POST_TOOL_USE:
                result = await hook.post_tool_use(context)
            else:
                result = await hook.post_tool_use_failure(context)

            summary.injected_messages.extend(result.injected_messages)
            summary.metadata.update(result.metadata)
            if result.permission_override is not None:
                summary.permission_override = result.permission_override
                summary.permission_reason = result.permission_reason
            if result.output_override is not None:
                summary.output_override = result.output_override
                context.output = result.output_override
            if result.updated_arguments is not None:
                updated_call = ToolCall(
                    id=context.tool_call.id,
                    name=context.tool_call.name,
                    arguments=dict(result.updated_arguments),
                )
                context.tool_call = updated_call
                summary.tool_call = updated_call
            if result.message is not None:
                summary.message = result.message
            if result.terminal_state is not None:
                summary.terminal_state = result.terminal_state
            if result.decision != HookDecision.CONTINUE:
                summary.decision = result.decision
                return summary

        return summary


class DuplicateActionHook(BaseToolHook):
    """Pre-hook that cancels already-completed duplicate actions."""

    def __init__(self, action_tracker: ActionTracker) -> None:
        self.action_tracker = action_tracker

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if context.skip_duplicate_check:
            return HookResult()
        is_duplicate, reason = self.action_tracker.check_tool_call(
            context.tool_call.name,
            context.tool_call.arguments,
        )
        if not is_duplicate:
            return HookResult()
        return HookResult(
            decision=HookDecision.CANCEL,
            message=f"[Skipped - duplicate action: {reason}]",
            terminal_state="duplicate",
        )


class ActionValidationHook(BaseToolHook):
    """Pre-hook that blocks invalid or dangerous tool arguments."""

    def __init__(self, validator: PreActionValidator) -> None:
        self.validator = validator

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        validation = self.validator.validate(
            context.tool_call.name,
            context.tool_call.arguments,
        )
        if validation.valid:
            messages: list[str] = []
            if validation.reason and validation.severity == "warning":
                messages.append(f"[Validation warning] {validation.reason}")
            return HookResult(injected_messages=messages)

        message = f"[Blocked - {validation.reason}]"
        if validation.suggestion:
            message += f" Suggestion: {validation.suggestion}"
        return HookResult(
            decision=HookDecision.DENY,
            message=message,
            terminal_state="blocked",
        )


class RollbackTrackingHook(BaseToolHook):
    """Pre-hook that tracks rollback actions for destructive tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        rollback_plan: RollbackPlan | None,
    ) -> None:
        self.registry = registry
        self.rollback_plan = rollback_plan

    async def pre_tool_use(self, context: HookContext) -> HookResult:
        if self.rollback_plan is None:
            return HookResult()
        if not is_destructive_tool(
            context.tool_call.name,
            context.tool_call.arguments,
        ):
            return HookResult()

        async def read_file_for_backup(path: str) -> str:
            read_result = await self.registry.execute("read", file_path=path)
            return read_result.output if not read_result.is_error else ""

        rollback_action = await create_rollback_plan_for_action(
            context.tool_call.name,
            context.tool_call.arguments,
            read_file_for_backup,
        )
        if rollback_action is None:
            return HookResult()

        self.rollback_plan.actions.append(rollback_action)
        return HookResult(metadata={"rollback_action": rollback_action})


class ActionHistoryHook(BaseToolHook):
    """Post-hook that records successful actions for deduplication and loop checks."""

    def __init__(self, action_tracker: ActionTracker) -> None:
        self.action_tracker = action_tracker

    async def post_tool_use(self, context: HookContext) -> HookResult:
        if not context.record_action:
            return HookResult()
        self.action_tracker.record_tool_call(
            context.tool_call.name,
            context.tool_call.arguments,
        )
        return HookResult()


class MemoryLifecycleHook(BaseToolHook):
    """Mirror durable memory updates into the session notepad."""

    async def post_tool_use(self, context: HookContext) -> HookResult:
        if context.result is None or context.result.is_error:
            return HookResult()

        store = MemoryStore(context.permission_policy.workspace_root)
        if context.tool_call.name == "project_memory_add_note":
            category = str(context.tool_call.arguments.get("category", "")).strip()
            content = str(context.tool_call.arguments.get("content", "")).strip()
            if category and content:
                store.append_notepad_working(
                    f"Remembered note [{category}]: {content}"
                )
        elif context.tool_call.name == "project_memory_add_directive":
            directive = str(context.tool_call.arguments.get("directive", "")).strip()
            priority = str(
                context.tool_call.arguments.get("priority", "normal")
            ).strip()
            if directive:
                store.append_notepad_working(
                    f"Remembered directive [{priority}]: {directive}"
                )
        return HookResult()


def build_default_tool_hooks(
    *,
    action_tracker: ActionTracker,
    validator: PreActionValidator,
    registry: ToolRegistry,
    rollback_plan: RollbackPlan | None,
) -> HookManager:
    """Build Loader's default tool hook stack for one runtime turn."""

    return HookManager(
        [
            DuplicateActionHook(action_tracker),
            ActionValidationHook(validator),
            RollbackTrackingHook(registry, rollback_plan),
            ActionHistoryHook(action_tracker),
            MemoryLifecycleHook(),
        ]
    )
