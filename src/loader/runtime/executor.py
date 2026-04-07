"""Unified tool execution path for runtime turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ..agent.parsing import format_tool_result
from ..llm.base import Message, ToolCall
from ..tools.base import ConfirmationRequired, ToolRegistry
from ..tools.base import ToolResult as RegistryToolResult
from ..tools.workflow_tools import UserQuestionHandler
from .hooks import HookContext, HookDecision, HookManager
from .permissions import PermissionDecision, PermissionMode, PermissionPolicy
from .recovery import ErrorCategory, categorize_error
from .tracing import RuntimeTracer

BrowserConfirmation = Callable[[str, str, str], Awaitable[bool]] | None
ConfirmationEmitter = Callable[[str, str, str], Awaitable[None]] | None


class ToolExecutionState(StrEnum):
    """Outcome states for one tool call."""

    EXECUTED = "executed"
    DUPLICATE = "duplicate"
    BLOCKED = "blocked"
    DECLINED = "declined"


@dataclass
class ToolExecutionOutcome:
    """Structured outcome for one tool call."""

    tool_call: ToolCall
    state: ToolExecutionState
    message: Message
    event_content: str
    is_error: bool
    result_output: str
    error_category: ErrorCategory | None = None
    registry_result: RegistryToolResult | None = None
    hook_messages: list[str] | None = None
    rollback_action: Any | None = None
    required_permission: PermissionMode | None = None


class ToolExecutor:
    """Centralizes duplicate checks, validation, execution, and result messages."""

    def __init__(
        self,
        registry: ToolRegistry,
        tracer: RuntimeTracer,
        permission_policy: PermissionPolicy,
        hooks: HookManager | None = None,
    ) -> None:
        self.registry = registry
        self.tracer = tracer
        self.permission_policy = permission_policy
        self.hooks = hooks or HookManager()

    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        on_confirmation: BrowserConfirmation = None,
        on_user_question: UserQuestionHandler | None = None,
        emit_confirmation: ConfirmationEmitter = None,
        source: str,
        skip_duplicate_check: bool = False,
        record_action: bool = True,
        skip_confirmation: bool = False,
    ) -> ToolExecutionOutcome:
        """Execute a tool call through one consistent runtime path."""

        self.tracer.record(
            "tool.received",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            source=source,
        )

        browser_block = self._browser_command_message(tool_call)
        if browser_block is not None:
            return self._blocked_outcome(tool_call, browser_block)

        tool = self.registry.get(tool_call.name)
        hook_context = HookContext(
            tool_call=tool_call,
            tool=tool,
            registry=self.registry,
            permission_policy=self.permission_policy,
            source=source,
            skip_duplicate_check=skip_duplicate_check,
            record_action=record_action,
        )

        pre_hook_summary = await self.hooks.run_pre_tool_use(hook_context)
        tool_call = pre_hook_summary.tool_call

        if pre_hook_summary.decision != HookDecision.CONTINUE:
            terminal_message = pre_hook_summary.message or "[Blocked - hook denied tool use]"
            failure_summary = await self.hooks.run_post_tool_use_failure(
                HookContext(
                    tool_call=tool_call,
                    tool=tool,
                    registry=self.registry,
                    permission_policy=self.permission_policy,
                    source=source,
                    skip_duplicate_check=skip_duplicate_check,
                    record_action=record_action,
                    output=terminal_message,
                    is_error=pre_hook_summary.decision != HookDecision.CANCEL,
                )
            )
            final_output = self._merge_messages(
                terminal_message,
                pre_hook_summary.injected_messages + failure_summary.injected_messages,
            )
            state = self._state_from_hook(
                pre_hook_summary.terminal_state,
                pre_hook_summary.decision,
            )
            self.tracer.record(
                "tool.hook_short_circuit",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                state=state,
            )
            return ToolExecutionOutcome(
                tool_call=tool_call,
                state=state,
                message=Message.tool_result_message(
                    tool_call_id=tool_call.id,
                    display_content=final_output,
                    result_content=final_output,
                    is_error=state != ToolExecutionState.DUPLICATE,
                ),
                event_content=final_output,
                is_error=state != ToolExecutionState.DUPLICATE,
                result_output=final_output,
                error_category=(
                    categorize_error(final_output)
                    if state != ToolExecutionState.DUPLICATE
                    else None
                ),
                hook_messages=pre_hook_summary.injected_messages
                + failure_summary.injected_messages,
                rollback_action=pre_hook_summary.metadata.get("rollback_action"),
            )

        required_permission = (
            tool.get_required_permission(**tool_call.arguments)
            if tool is not None
            else self.permission_policy.required_mode_for(tool_call.name)
        )
        permission_outcome = self.permission_policy.authorize(
            tool_call.name,
            required_mode=required_permission,
            override=pre_hook_summary.permission_override,
            override_reason=pre_hook_summary.permission_reason,
            arguments=tool_call.arguments,
        )
        if permission_outcome.decision == PermissionDecision.DENY:
            denied_output = self._merge_messages(
                f"[Blocked - {permission_outcome.reason}]",
                pre_hook_summary.injected_messages,
            )
            failure_summary = await self.hooks.run_post_tool_use_failure(
                HookContext(
                    tool_call=tool_call,
                    tool=tool,
                    registry=self.registry,
                    permission_policy=self.permission_policy,
                    source=source,
                    skip_duplicate_check=skip_duplicate_check,
                    record_action=record_action,
                    output=denied_output,
                    is_error=True,
                )
            )
            final_output = self._merge_messages(
                denied_output,
                failure_summary.injected_messages,
            )
            self.tracer.record(
                "tool.permission_denied",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                required_mode=required_permission.as_str(),
                active_mode=self.permission_policy.active_mode.as_str(),
            )
            return self._blocked_outcome(
                tool_call,
                final_output,
                hook_messages=pre_hook_summary.injected_messages
                + failure_summary.injected_messages,
                rollback_action=pre_hook_summary.metadata.get("rollback_action"),
                required_permission=required_permission,
            )

        if permission_outcome.decision == PermissionDecision.ASK:
            approved = await self._prompt_for_permission(
                tool_call,
                on_confirmation,
                emit_confirmation,
                permission_outcome.reason,
                skip_confirmation=skip_confirmation,
            )
            if not approved:
                declined_output = self._merge_messages(
                    f"Tool {tool_call.name} was declined by user",
                    pre_hook_summary.injected_messages,
                )
                failure_summary = await self.hooks.run_post_tool_use_failure(
                    HookContext(
                        tool_call=tool_call,
                        tool=tool,
                        registry=self.registry,
                        permission_policy=self.permission_policy,
                        source=source,
                        skip_duplicate_check=skip_duplicate_check,
                        record_action=record_action,
                        output=declined_output,
                        is_error=False,
                    )
                )
                final_output = self._merge_messages(
                    declined_output,
                    failure_summary.injected_messages,
                )
                return ToolExecutionOutcome(
                    tool_call=tool_call,
                    state=ToolExecutionState.DECLINED,
                    message=Message.tool_result_message(
                        tool_call_id=tool_call.id,
                        display_content=final_output,
                        result_content=final_output,
                    ),
                    event_content=final_output,
                    is_error=False,
                    result_output=final_output,
                    hook_messages=pre_hook_summary.injected_messages
                    + failure_summary.injected_messages,
                    rollback_action=pre_hook_summary.metadata.get("rollback_action"),
                    required_permission=required_permission,
                )

        result = await self._execute_registry(
            tool_call,
            on_confirmation,
            on_user_question,
            emit_confirmation,
            skip_confirmation=True,
        )
        registry_result = result
        post_hook_context = HookContext(
            tool_call=tool_call,
            tool=tool,
            registry=self.registry,
            permission_policy=self.permission_policy,
            source=source,
            skip_duplicate_check=skip_duplicate_check,
            record_action=record_action,
            result=registry_result,
            output=registry_result.output,
            is_error=registry_result.is_error,
        )
        if registry_result.is_error:
            post_hook_summary = await self.hooks.run_post_tool_use_failure(post_hook_context)
        else:
            post_hook_summary = await self.hooks.run_post_tool_use(post_hook_context)
        if post_hook_summary.output_override is not None:
            result = RegistryToolResult(
                output=post_hook_summary.output_override,
                is_error=registry_result.is_error,
                metadata=dict(registry_result.metadata),
            )

        result_text = format_tool_result(
            tool_call.name,
            result.output,
            result.is_error,
        )
        final_event_content = self._merge_messages(
            result.output,
            pre_hook_summary.injected_messages + post_hook_summary.injected_messages,
        )
        final_display_content = self._merge_messages(
            result_text,
            pre_hook_summary.injected_messages + post_hook_summary.injected_messages,
        )

        category = categorize_error(result.output) if result.is_error else None
        state = ToolExecutionState.EXECUTED
        if result.output == f"Tool {tool_call.name} was declined by user":
            state = ToolExecutionState.DECLINED

        self.tracer.record(
            "tool.executed",
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            state=state,
            is_error=result.is_error,
            required_permission=required_permission.as_str(),
        )
        return ToolExecutionOutcome(
            tool_call=tool_call,
            state=state,
            message=Message.tool_result_message(
                tool_call_id=tool_call.id,
                display_content=final_display_content,
                result_content=final_event_content,
                is_error=result.is_error,
            ),
            event_content=final_event_content,
            is_error=result.is_error,
            result_output=final_event_content,
            error_category=category,
            registry_result=result,
            hook_messages=pre_hook_summary.injected_messages
            + post_hook_summary.injected_messages,
            rollback_action=pre_hook_summary.metadata.get("rollback_action"),
            required_permission=required_permission,
        )

    def _blocked_outcome(
        self,
        tool_call: ToolCall,
        message: str,
        *,
        hook_messages: list[str] | None = None,
        rollback_action: Any | None = None,
        required_permission: PermissionMode | None = None,
    ) -> ToolExecutionOutcome:
        return ToolExecutionOutcome(
            tool_call=tool_call,
            state=ToolExecutionState.BLOCKED,
            message=Message.tool_result_message(
                tool_call_id=tool_call.id,
                display_content=message,
                result_content=message,
                is_error=True,
            ),
            event_content=message,
            is_error=True,
            result_output=message,
            error_category=categorize_error(message),
            hook_messages=hook_messages,
            rollback_action=rollback_action,
            required_permission=required_permission,
        )

    async def _execute_registry(
        self,
        tool_call: ToolCall,
        on_confirmation: BrowserConfirmation,
        on_user_question: UserQuestionHandler | None,
        emit_confirmation: ConfirmationEmitter,
        *,
        skip_confirmation: bool = False,
    ) -> RegistryToolResult:
        previous_skip = self.registry.skip_confirmation
        if skip_confirmation:
            self.registry.skip_confirmation = True
        try:
            extra_kwargs: dict[str, Any] = {}
            if tool_call.name == "AskUserQuestion":
                extra_kwargs["user_response_handler"] = on_user_question
            return await self.registry.execute(
                tool_call.name,
                **tool_call.arguments,
                **extra_kwargs,
            )
        except ConfirmationRequired as confirmation:
            self.tracer.record(
                "tool.confirmation_requested",
                tool_name=confirmation.tool_name,
                tool_call_id=tool_call.id,
            )
            if emit_confirmation:
                await emit_confirmation(
                    confirmation.tool_name,
                    confirmation.message,
                    confirmation.details,
                )
            if on_confirmation:
                confirmed = await on_confirmation(
                    confirmation.tool_name,
                    confirmation.message,
                    confirmation.details,
                )
            else:
                confirmed = True

            if not confirmed:
                return RegistryToolResult(
                    output=f"Tool {tool_call.name} was declined by user",
                    is_error=False,
                )

            self.registry.skip_confirmation = True
            try:
                extra_kwargs: dict[str, Any] = {}
                if tool_call.name == "AskUserQuestion":
                    extra_kwargs["user_response_handler"] = on_user_question
                return await self.registry.execute(
                    tool_call.name,
                    **tool_call.arguments,
                    **extra_kwargs,
                )
            finally:
                self.registry.skip_confirmation = previous_skip
        finally:
            self.registry.skip_confirmation = previous_skip

    async def _prompt_for_permission(
        self,
        tool_call: ToolCall,
        on_confirmation: BrowserConfirmation,
        emit_confirmation: ConfirmationEmitter,
        reason: str | None,
        *,
        skip_confirmation: bool,
    ) -> bool:
        if skip_confirmation or self.registry.skip_confirmation:
            return True

        message = reason or f"Approve {tool_call.name}"
        details = self._format_permission_details(tool_call, reason)
        if emit_confirmation:
            await emit_confirmation(tool_call.name, message, details)
        if on_confirmation:
            return await on_confirmation(tool_call.name, message, details)
        return False

    @staticmethod
    def _merge_messages(primary: str, extra_messages: list[str]) -> str:
        parts = [message for message in extra_messages if message]
        if primary:
            parts.append(primary)
        return "\n".join(parts)

    @staticmethod
    def _state_from_hook(
        terminal_state: str | None,
        decision: HookDecision,
    ) -> ToolExecutionState:
        if terminal_state == "duplicate" or decision == HookDecision.CANCEL:
            return ToolExecutionState.DUPLICATE
        if terminal_state == "declined":
            return ToolExecutionState.DECLINED
        return ToolExecutionState.BLOCKED

    def _browser_command_message(self, tool_call: ToolCall) -> str | None:
        if tool_call.name != "bash":
            return None

        command = str(tool_call.arguments.get("command", ""))
        browser_terms = ["xdg-open", "open ", "firefox", "chrome", "browser"]
        if any(term in command for term in browser_terms):
            return "[Blocked - Browser/display commands are not supported in the terminal runtime]"
        return None

    def _format_permission_details(
        self,
        tool_call: ToolCall,
        reason: str | None,
    ) -> str:
        tool = self.registry.get(tool_call.name)
        required_permission = (
            tool.get_required_permission(**tool_call.arguments)
            if tool is not None
            else self.permission_policy.required_mode_for(tool_call.name)
        )
        summary = self.permission_policy.authorize(
            tool_call.name,
            required_mode=required_permission,
            arguments=tool_call.arguments,
        )
        details = [
            f"tool={tool_call.name}",
            f"active_mode={summary.active_mode.as_str()}",
            f"required_mode={summary.required_mode.as_str()}",
        ]
        if summary.request is not None:
            details.append(f"input={summary.request.input_summary}")
            if summary.request.path_hint:
                details.append(f"path={summary.request.path_hint}")
        if summary.matched_rule is not None and summary.matched_disposition is not None:
            details.append(
                f"matched_{summary.matched_disposition.value}_rule={summary.matched_rule.raw}"
            )
        if reason:
            details.append(f"reason={reason}")
        return "\n".join(details)
