"""Typed turn engine for Loader runtime execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..agent.parsing import parse_tool_calls
from ..agent.reasoning import (
    RollbackPlan,
    TaskCompletionCheck,
    create_rollback_plan_for_action,
    detect_premature_completion,
    estimate_complexity,
    get_continuation_prompt,
    get_token_budget,
    is_destructive_tool,
    should_self_critique,
)
from ..agent.recovery import RecoveryContext, format_failure_message, format_recovery_prompt
from ..llm.base import Message, Role, ToolCall
from .events import AgentEvent, TurnSummary
from .executor import ToolExecutionState, ToolExecutor
from .tracing import RuntimeTracer

EventSink = Callable[[AgentEvent], Awaitable[None]]
ConfirmationHandler = Callable[[str, str, str], Awaitable[bool]] | None


@dataclass
class AssistantTurn:
    """Assistant output for one iteration of the conversation loop."""

    content: str
    response_content: str
    tool_calls: list[ToolCall]
    pending_tool_calls_seen: set[str] = field(default_factory=set)
    usage: dict[str, int] = field(default_factory=dict)


class ConversationRuntime:
    """Runs one explicit conversation turn against the current session."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.tracer = RuntimeTracer()
        self.executor = ToolExecutor(agent.registry, agent.safeguards, self.tracer)

    async def run_turn(
        self,
        task: str,
        emit: EventSink,
        on_confirmation: ConfirmationHandler = None,
        original_task: str | None = None,
    ) -> TurnSummary:
        """Run one task turn and return a structured summary."""

        await self._prepare_runtime_capabilities()

        iterations = 0
        final_response = ""
        actions_taken: list[str] = []
        continuation_count = 0
        empty_retry_count = 0
        max_empty_retries = 5
        extracted_iterations = 0
        max_extracted_iterations = 3
        consecutive_errors = 0

        complexity = estimate_complexity(task)
        max_tokens, _ = get_token_budget(complexity)
        effective_max_tokens = min(self.agent.config.max_tokens, max(max_tokens, 512))

        rollback_plan = RollbackPlan() if self.agent.config.reasoning.rollback else None
        summary = TurnSummary(final_response="")

        while iterations < self.agent.config.max_iterations:
            iterations += 1
            summary.iterations = iterations
            self.tracer.record("turn.iteration_started", iteration=iterations)

            if iterations == 1 and len(self.agent.messages) == 1:
                task_lower = task.lower()
                action_keywords = [
                    "create",
                    "write",
                    "make",
                    "run",
                    "execute",
                    "build",
                    "install",
                    "delete",
                    "remove",
                    "add",
                    "edit",
                    "modify",
                    "update",
                    "fix",
                ]
                if any(keyword in task_lower for keyword in action_keywords):
                    self.agent.session.append(Message(role=Role.ASSISTANT, content="["))

            steering_messages = self.agent._drain_steering_queue()
            for steering_message in steering_messages:
                await emit(AgentEvent(type="steering", content=steering_message))
                self.agent.session.append(
                    Message(
                        role=Role.USER,
                        content=f"[USER INTERRUPTION]: {steering_message}",
                    )
                )

            await emit(AgentEvent(type="thinking"))
            assistant_turn = await self._request_assistant_turn(
                emit=emit,
                max_tokens=effective_max_tokens,
            )
            self._merge_usage(summary.usage, assistant_turn.usage)

            content = assistant_turn.content
            response_content = assistant_turn.response_content
            tool_calls = list(assistant_turn.tool_calls)
            pending_tool_calls_seen = set(assistant_turn.pending_tool_calls_seen)

            if not content.strip():
                empty_retry_count += 1
                if empty_retry_count <= max_empty_retries:
                    task_context = original_task or task
                    retry_prompts = [
                        "Great! Now let me proceed with the task. I'll start by using my tools.",
                        "I understand. Let me create that now using my tools (write, bash, etc.).",
                        f"Proceeding with: {task_context[:80]}. I'll use the write tool to create the files.",
                        "Starting now. First step: create the necessary files and directories.",
                        f"Let me complete this task step by step. The goal is: {task_context[:100]}",
                    ]
                    prompt = retry_prompts[min(empty_retry_count - 1, len(retry_prompts) - 1)]
                    self.agent.session.append(Message(role=Role.ASSISTANT, content=prompt))
                    continue

                final_response = (
                    "I need a bit more direction. What specifically would you like me to create or do?"
                )
                summary.final_response = final_response
                summary.failures.append("assistant returned empty output repeatedly")
                await emit(AgentEvent(type="response", content=final_response))
                break

            if self.agent.use_react:
                parsed = parse_tool_calls(content)
                tool_calls = parsed.tool_calls
                content = parsed.content

                if parsed.is_final_answer and not tool_calls:
                    assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                    self.agent.session.append(assistant_message)
                    summary.assistant_messages.append(assistant_message)
                    final_response = content
                    summary.final_response = final_response
                    self.tracer.record("turn.completed", reason="final_answer")
                    await emit(AgentEvent(type="response", content=final_response))
                    break

            tool_source = "native"
            if not tool_calls:
                raw_tool_calls = self.agent._extract_raw_json_tool_calls(response_content)
                if raw_tool_calls:
                    tool_calls = raw_tool_calls
                    tool_source = "raw_text"
                    await emit(AgentEvent(type="clear_stream"))

            if tool_calls:
                if tool_source == "raw_text":
                    extracted_iterations += 1
                    if extracted_iterations > max_extracted_iterations:
                        final_response = (
                            content
                            + "\n\nLet me know if you'd like me to continue or make changes."
                        )
                        assistant_message = Message(role=Role.ASSISTANT, content=response_content)
                        self.agent.session.append(assistant_message)
                        summary.assistant_messages.append(assistant_message)
                        summary.final_response = final_response
                        summary.failures.append("raw tool extraction exceeded iteration budget")
                        await emit(AgentEvent(type="response", content=final_response))
                        break

                assistant_message = Message(
                    role=Role.ASSISTANT,
                    content=response_content,
                    tool_calls=tool_calls,
                )
                self.agent.session.append(assistant_message)
                summary.assistant_messages.append(assistant_message)
                self.tracer.record(
                    "assistant.tool_batch",
                    tool_count=len(tool_calls),
                    source=tool_source,
                )

                for tool_call in tool_calls:
                    cfg = self.agent.config.reasoning

                    if cfg.confidence_scoring:
                        context = "\n".join(
                            message.content[:500]
                            for message in self.agent.messages[-5:]
                            if message.content
                        )
                        confidence = await self.agent._assess_confidence(
                            tool_call.name,
                            tool_call.arguments,
                            context,
                        )
                        await emit(
                            AgentEvent(
                                type="confidence",
                                content=f"Confidence: {confidence.level.name} ({confidence.score}/5)",
                                confidence=confidence,
                                tool_name=tool_call.name,
                            )
                        )
                        if confidence.score < cfg.min_confidence_for_action:
                            low_confidence_message = (
                                "[LOW CONFIDENCE WARNING] The planned action has low confidence "
                                f"({confidence.level.name}).\n"
                                f"Reasoning: {confidence.reasoning}\n"
                                f"Risks: {', '.join(confidence.risks)}\n"
                                "Consider an alternative approach or gather more information first."
                            )
                            self.agent.session.append(
                                Message(role=Role.USER, content=low_confidence_message)
                            )
                            continue

                    if tool_call.id not in pending_tool_calls_seen:
                        await emit(
                            AgentEvent(
                                type="tool_call",
                                tool_name=tool_call.name,
                                tool_args=tool_call.arguments,
                            )
                        )

                    actions_taken.append(f"{tool_call.name}: {str(tool_call.arguments)[:100]}")

                    if rollback_plan and is_destructive_tool(tool_call.name, tool_call.arguments):

                        async def read_file_for_backup(path: str) -> str:
                            read_result = await self.agent.registry.execute("read", file_path=path)
                            return read_result.output if not read_result.is_error else ""

                        rollback_action = await create_rollback_plan_for_action(
                            tool_call.name,
                            tool_call.arguments,
                            read_file_for_backup,
                        )
                        if rollback_action:
                            rollback_plan.actions.append(rollback_action)
                            if self.agent.config.reasoning.show_rollback_plan:
                                await emit(
                                    AgentEvent(
                                        type="rollback",
                                        content=f"Rollback tracked: {rollback_action.description}",
                                        rollback_action=rollback_action,
                                    )
                                )

                    outcome = await self.executor.execute_tool_call(
                        tool_call,
                        on_confirmation=on_confirmation,
                        emit_confirmation=self._emit_confirmation(emit),
                        source=tool_source,
                    )

                    if (
                        outcome.state == ToolExecutionState.EXECUTED
                        and outcome.is_error
                        and self.agent.config.auto_recover
                    ):
                        recovery_result = await self._handle_recovery(tool_call, outcome, emit)
                        if recovery_result is not None:
                            summary.tool_result_messages.append(recovery_result)
                            self.agent.session.append(recovery_result)
                            continue

                    if outcome.state == ToolExecutionState.EXECUTED and not outcome.is_error:
                        self.agent._recovery_context = None
                        is_loop, loop_description = self.agent.safeguards.detect_loop()
                        if is_loop:
                            final_response = (
                                "I noticed I was repeating the same actions. "
                                "Let me know what you'd like me to do differently."
                            )
                            summary.final_response = final_response
                            summary.failures.append(loop_description)
                            loop_message = Message(role=Role.ASSISTANT, content=final_response)
                            self.agent.session.append(loop_message)
                            summary.assistant_messages.append(loop_message)
                            await emit(
                                AgentEvent(
                                    type="error",
                                    content=(
                                        f"Loop detected: {loop_description}. "
                                        "Stopping to prevent repetitive behavior."
                                    ),
                                )
                            )
                            await emit(AgentEvent(type="response", content=final_response))
                            return self._finalize_summary(summary)

                    if outcome.is_error:
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0

                    await emit(
                        AgentEvent(
                            type="tool_result",
                            content=outcome.event_content,
                            tool_name=tool_call.name,
                            is_error=outcome.is_error,
                        )
                    )

                    if (
                        cfg.verification
                        and outcome.state == ToolExecutionState.EXECUTED
                        and not outcome.is_error
                    ):
                        verification = await self.agent._verify_action(
                            tool_call.name,
                            tool_call.arguments,
                            outcome.result_output,
                        )
                        await emit(
                            AgentEvent(
                                type="verification",
                                content=f"Verified: {verification.verified}",
                                verification=verification,
                                tool_name=tool_call.name,
                            )
                        )
                        if not verification.verified and verification.needs_correction:
                            correction_message = (
                                "[VERIFICATION FAILED] The action did not produce expected results.\n"
                                f"Discrepancies: {', '.join(verification.discrepancies)}\n"
                                f"Suggestion: {verification.correction_suggestion}"
                            )
                            self.agent.session.append(
                                Message(role=Role.USER, content=correction_message)
                            )
                            continue

                    self.agent.session.append(outcome.message)
                    summary.tool_result_messages.append(outcome.message)

                if consecutive_errors >= 3:
                    final_response = (
                        "I ran into some issues. Let me know if you'd like me to try a different approach."
                    )
                    summary.final_response = final_response
                    summary.failures.append("three consecutive tool errors")
                    await emit(AgentEvent(type="response", content=final_response))
                    break

                continue

            if self.agent._contains_unexecuted_code(response_content):
                if iterations < self.agent.config.max_iterations - 1:
                    self.agent.session.append(Message(role=Role.ASSISTANT, content=response_content))
                    self.agent.session.append(
                        Message(
                            role=Role.USER,
                            content=(
                                "CRITICAL ERROR: You are PRETENDING to use tools instead of actually "
                                "using them.\n\n"
                                "DO NOT write:\n"
                                "- 'Used bash tool with command...' (THIS IS FAKE)\n"
                                "- 'Created a file using the write tool...' (THIS IS FAKE)\n"
                                "- 'Here is what I did:' followed by descriptions\n"
                                "- Numbered steps or instructions\n"
                                "- Code blocks for me to copy\n\n"
                                "Your tool calls MUST go through the proper tool interface.\n"
                                "Writing 'Used bash tool...' does NOT execute anything!\n\n"
                                "ACTUALLY call the tools using the tool_call mechanism.\n"
                                "DO IT NOW - stop narrating and start executing."
                            ),
                        )
                    )
                    continue

            if (
                not self.agent.use_react
                and len(actions_taken) == 0
                and iterations < self.agent.config.max_iterations - 2
            ):
                deflection_phrases = ["you can", "you should", "you could", "try running"]
                if any(phrase in content.lower() for phrase in deflection_phrases):
                    self.agent.session.append(Message(role=Role.ASSISTANT, content=response_content))
                    self.agent.session.append(
                        Message(
                            role=Role.USER,
                            content="Please use your tools to execute the task rather than telling me what to do.",
                        )
                    )
                    continue

            cfg = self.agent.config.reasoning
            if cfg.self_critique and len(content) > 100:
                is_code_response = "```" in content or any(
                    keyword in content.lower()
                    for keyword in ["def ", "function ", "class ", "import "]
                )
                if should_self_critique(content, is_code=is_code_response):
                    critique = await self.agent._self_critique(content, task)
                    await emit(
                        AgentEvent(
                            type="critique",
                            content=f"Self-critique: {len(critique.issues_found)} issues found",
                            critique=critique,
                        )
                    )
                    if critique.can_revise():
                        revision_message = (
                            "[SELF-CRITIQUE] Review your response:\n"
                            f"Issues found: {', '.join(critique.issues_found)}\n"
                            f"Suggestions: {', '.join(critique.suggestions)}\n\n"
                            "Please provide an improved response addressing these issues."
                        )
                        self.agent.session.append(Message(role=Role.ASSISTANT, content=response_content))
                        self.agent.session.append(Message(role=Role.USER, content=revision_message))
                        critique.revision_count += 1
                        continue

            is_text_loop, loop_description = self.agent.safeguards.detect_text_loop(content)
            if is_text_loop:
                final_response = (
                    "I seem to be repeating myself. Let me know if you'd like me to try a different approach."
                )
                summary.final_response = final_response
                summary.failures.append(loop_description)
                final_message = Message(role=Role.ASSISTANT, content=final_response)
                self.agent.session.append(final_message)
                summary.assistant_messages.append(final_message)
                await emit(
                    AgentEvent(
                        type="error",
                        content=f"Text loop detected: {loop_description}. Stopping.",
                    )
                )
                await emit(AgentEvent(type="response", content=final_response))
                return self._finalize_summary(summary)

            self.agent.safeguards.record_response(content)
            effective_task = original_task or task
            if cfg.completion_check and continuation_count < cfg.max_continuation_prompts:
                is_premature = (
                    detect_premature_completion(effective_task, content, actions_taken)
                    if cfg.use_quick_completion
                    else False
                )
                if is_premature:
                    continuation_count += 1
                    continuation_prompt = get_continuation_prompt(
                        effective_task,
                        actions_taken,
                        content,
                    )
                    await emit(
                        AgentEvent(
                            type="completion_check",
                            content=f"Task may be incomplete ({len(actions_taken)} actions taken)",
                            completion_check=TaskCompletionCheck(
                                original_task=effective_task,
                                is_complete=False,
                                accomplished=[action.split(":")[0] for action in actions_taken],
                                continuation_prompt=continuation_prompt,
                            ),
                        )
                    )
                    self.agent.session.append(Message(role=Role.ASSISTANT, content=response_content))
                    self.agent.session.append(Message(role=Role.USER, content=continuation_prompt))
                    continue

            final_response = content
            if actions_taken and final_response.strip() and not final_response.rstrip().endswith("?"):
                final_response = (
                    final_response.rstrip()
                    + "\n\nWould you like me to make any changes or additions?"
                )

            final_message = Message(role=Role.ASSISTANT, content=response_content)
            self.agent.session.append(final_message)
            summary.assistant_messages.append(final_message)

            if rollback_plan and rollback_plan.actions:
                await emit(
                    AgentEvent(
                        type="rollback_summary",
                        content=f"Rollback plan: {len(rollback_plan.actions)} action(s) tracked",
                        rollback_plan=rollback_plan,
                    )
                )

            summary.final_response = final_response
            await emit(AgentEvent(type="response", content=final_response))
            break

        return self._finalize_summary(summary)

    async def _request_assistant_turn(
        self,
        *,
        emit: EventSink,
        max_tokens: int,
    ) -> AssistantTurn:
        self.agent.safeguards.code_filter.reset()
        tools = None if self.agent.use_react else self.agent.registry.get_schemas()
        self.tracer.record(
            "assistant.requested",
            use_react=self.agent.use_react,
            stream=self.agent.config.stream,
            max_tokens=max_tokens,
        )

        if self.agent.config.stream:
            full_content = ""
            full_content_unfiltered = ""
            tool_calls: list[ToolCall] = []
            pending_tool_calls_seen: set[str] = set()

            async for chunk in self.agent.backend.stream(
                messages=self.agent.session.build_request_messages(),
                tools=tools,
                temperature=self.agent.config.temperature,
                max_tokens=max_tokens,
            ):
                filtered_content = ""
                if chunk.content:
                    filtered_content = self.agent.safeguards.filter_stream_chunk(chunk.content)
                    full_content_unfiltered += chunk.content

                if filtered_content or chunk.is_done:
                    await emit(
                        AgentEvent(
                            type="stream",
                            content=filtered_content,
                            is_stream_end=chunk.is_done,
                        )
                    )

                if self.agent.safeguards.should_steer():
                    steering_message = self.agent.safeguards.get_steering_message()
                    if steering_message:
                        self.agent._steering_queue.put_nowait(steering_message)

                if chunk.pending_tool_call and chunk.pending_tool_call.id not in pending_tool_calls_seen:
                    pending_tool_calls_seen.add(chunk.pending_tool_call.id)
                    await emit(
                        AgentEvent(
                            type="tool_call",
                            tool_name=chunk.pending_tool_call.name,
                            tool_args=chunk.pending_tool_call.arguments,
                        )
                    )

                if chunk.is_done:
                    full_content = chunk.full_content or full_content_unfiltered
                    tool_calls = chunk.tool_calls

            self.tracer.record(
                "assistant.responded",
                stream=True,
                tool_call_count=len(tool_calls),
                content_length=len(full_content),
            )
            return AssistantTurn(
                content=full_content,
                response_content=full_content,
                tool_calls=tool_calls,
                pending_tool_calls_seen=pending_tool_calls_seen,
            )

        response = await self.agent.backend.complete(
            messages=self.agent.session.build_request_messages(),
            tools=tools,
            temperature=self.agent.config.temperature,
            max_tokens=max_tokens,
        )
        response_content = response.content
        content = self.agent.safeguards.filter_complete_content(response.content)
        tool_calls = response.tool_calls if not self.agent.use_react else []
        if self.agent.safeguards.should_steer():
            steering_message = self.agent.safeguards.get_steering_message()
            if steering_message:
                self.agent._steering_queue.put_nowait(steering_message)
        self.tracer.record(
            "assistant.responded",
            stream=False,
            tool_call_count=len(tool_calls),
            content_length=len(content),
        )
        return AssistantTurn(
            content=content,
            response_content=response_content,
            tool_calls=tool_calls,
            usage=response.usage,
        )

    async def _handle_recovery(
        self,
        tool_call: ToolCall,
        outcome,
        emit: EventSink,
    ) -> Message | None:
        if self.agent._recovery_context is None:
            self.agent._recovery_context = RecoveryContext(
                original_tool=tool_call.name,
                original_args=tool_call.arguments,
                max_retries=self.agent.config.max_recovery_attempts,
            )

        if self.agent._recovery_context.is_similar_attempt(tool_call.name, tool_call.arguments):
            await emit(
                AgentEvent(
                    type="error",
                    content=(
                        "Loop detected: already tried a similar command. "
                        "Try a DIFFERENT approach (e.g., read a config file first)."
                    ),
                    tool_name=tool_call.name,
                )
            )
        else:
            self.agent._recovery_context.add_attempt(
                tool_call.name,
                tool_call.arguments,
                outcome.result_output,
            )

        if self.agent._recovery_context.can_retry():
            attempt_number = len(self.agent._recovery_context.attempts)
            await emit(
                AgentEvent(
                    type="recovery",
                    content=(
                        "Tool failed, attempting recovery "
                        f"({attempt_number}/{self.agent._recovery_context.max_retries})"
                    ),
                    tool_name=tool_call.name,
                    recovery_attempt=attempt_number,
                )
            )
            recovery_prompt = format_recovery_prompt(
                self.agent._recovery_context,
                tool_call.name,
                tool_call.arguments,
                outcome.result_output,
            )
            return Message.tool_result_message(
                tool_call_id=tool_call.id,
                display_content=recovery_prompt,
                result_content=recovery_prompt,
                is_error=True,
            )

        failure_message = format_failure_message(self.agent._recovery_context)
        await emit(
            AgentEvent(
                type="error",
                content=failure_message,
                tool_name=tool_call.name,
            )
        )
        self.agent._recovery_context = None
        return Message.tool_result_message(
            tool_call_id=tool_call.id,
            display_content=(
                f"Observation [{tool_call.name}]: Error: {failure_message}"
            ),
            result_content=failure_message,
            is_error=True,
        )

    def _finalize_summary(self, summary: TurnSummary) -> TurnSummary:
        summary.trace = list(self.tracer.events)
        return summary

    @staticmethod
    def _merge_usage(target: dict[str, int], update: dict[str, int]) -> None:
        for key, value in update.items():
            target[key] = target.get(key, 0) + value

    async def _prepare_runtime_capabilities(self) -> None:
        describe_model = getattr(self.agent.backend, "describe_model", None)
        if callable(describe_model):
            await describe_model()

        previous_profile = self.agent.capability_profile
        self.agent.refresh_capability_profile()
        if self.agent.capability_profile != previous_profile:
            self.tracer.record(
                "runtime.capabilities_refreshed",
                model_name=self.agent.capability_profile.model_name,
                supports_native_tools=self.agent.capability_profile.supports_native_tools,
                preferred_tool_call_format=(
                    self.agent.capability_profile.preferred_tool_call_format
                ),
            )

    @staticmethod
    def _emit_confirmation(emit: EventSink):
        async def _emit(tool_name: str, message: str, details: str) -> None:
            await emit(
                AgentEvent(
                    type="confirmation",
                    tool_name=tool_name,
                    confirm_message=message,
                    confirm_details=details,
                )
            )

        return _emit
