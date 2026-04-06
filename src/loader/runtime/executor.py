"""Unified tool execution path for runtime turns."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from ..agent.parsing import format_tool_result
from ..agent.recovery import ErrorCategory, categorize_error
from ..llm.base import Message, ToolCall
from ..tools.base import ConfirmationRequired, ToolRegistry
from ..tools.base import ToolResult as RegistryToolResult
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


class ToolExecutor:
    """Centralizes duplicate checks, validation, execution, and result messages."""

    def __init__(self, registry: ToolRegistry, safeguards, tracer: RuntimeTracer) -> None:
        self.registry = registry
        self.safeguards = safeguards
        self.tracer = tracer

    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        on_confirmation: BrowserConfirmation = None,
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

        if not skip_duplicate_check:
            is_duplicate, duplicate_reason = self.safeguards.check_duplicate(
                tool_call.name,
                tool_call.arguments,
            )
            if is_duplicate:
                self.tracer.record(
                    "tool.duplicate",
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    reason=duplicate_reason,
                )
                duplicate_message = f"[Skipped - duplicate action: {duplicate_reason}]"
                return ToolExecutionOutcome(
                    tool_call=tool_call,
                    state=ToolExecutionState.DUPLICATE,
                    message=Message.tool_result_message(
                        tool_call_id=tool_call.id,
                        display_content=duplicate_message,
                        result_content=duplicate_message,
                    ),
                    event_content=duplicate_message,
                    is_error=False,
                    result_output=duplicate_message,
                )

        validation = self.safeguards.validate_action(tool_call.name, tool_call.arguments)
        if not validation.valid:
            error_message = f"[Blocked - {validation.reason}]"
            if validation.suggestion:
                error_message += f" Suggestion: {validation.suggestion}"
            self.tracer.record(
                "tool.blocked",
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                reason=validation.reason,
            )
            return self._blocked_outcome(tool_call, error_message)

        result = await self._execute_registry(
            tool_call,
            on_confirmation,
            emit_confirmation,
            skip_confirmation=skip_confirmation,
        )
        result_text = format_tool_result(
            tool_call.name,
            result.output,
            result.is_error,
        )
        if record_action and not result.is_error:
            self.safeguards.record_action(tool_call.name, tool_call.arguments)

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
        )
        return ToolExecutionOutcome(
            tool_call=tool_call,
            state=state,
            message=Message.tool_result_message(
                tool_call_id=tool_call.id,
                display_content=result_text,
                result_content=result.output,
                is_error=result.is_error,
            ),
            event_content=result.output,
            is_error=result.is_error,
            result_output=result.output,
            error_category=category,
            registry_result=result,
        )

    def _blocked_outcome(self, tool_call: ToolCall, message: str) -> ToolExecutionOutcome:
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
        )

    async def _execute_registry(
        self,
        tool_call: ToolCall,
        on_confirmation: BrowserConfirmation,
        emit_confirmation: ConfirmationEmitter,
        *,
        skip_confirmation: bool = False,
    ) -> RegistryToolResult:
        previous_skip = self.registry.skip_confirmation
        if skip_confirmation:
            self.registry.skip_confirmation = True
        try:
            return await self.registry.execute(tool_call.name, **tool_call.arguments)
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
                return await self.registry.execute(tool_call.name, **tool_call.arguments)
            finally:
                self.registry.skip_confirmation = previous_skip
        finally:
            self.registry.skip_confirmation = previous_skip

    def _browser_command_message(self, tool_call: ToolCall) -> str | None:
        if tool_call.name != "bash":
            return None

        command = str(tool_call.arguments.get("command", ""))
        browser_terms = ["xdg-open", "open ", "firefox", "chrome", "browser"]
        if any(term in command for term in browser_terms):
            return "[Blocked - Browser/display commands are not supported in the terminal runtime]"
        return None
