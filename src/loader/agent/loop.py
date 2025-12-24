"""The main agent loop."""

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable

from ..llm.base import LLMBackend, Message, Role, ToolCall
from ..tools.base import ToolRegistry, create_default_registry
from ..context.project import ProjectContext, detect_project
from .prompts import build_system_prompt
from .parsing import parse_tool_calls, format_tool_result
from .planner import Plan, parse_plan, should_plan, format_step_prompt, PLANNING_PROMPT, SHOULD_PLAN_PROMPT
from .recovery import RecoveryContext, format_recovery_prompt, format_failure_message


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_iterations: int = 20
    temperature: float = 0.7
    max_tokens: int = 4096
    force_react: bool = False  # Force ReAct even if model supports native tools
    auto_context: bool = True  # Auto-detect project context on startup
    auto_plan: bool = True  # Auto-plan complex tasks
    auto_recover: bool = True  # Auto-recover from tool errors
    max_recovery_attempts: int = 3  # Max retries per failed tool
    stream: bool = True  # Stream LLM responses for real-time output


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: str  # "thinking", "tool_call", "tool_result", "response", "error", "plan", "step", "recovery", "stream"
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None
    step_info: str | None = None  # For step progress like "[2/5] Doing X"
    recovery_attempt: int | None = None  # For recovery events
    is_stream_end: bool = False  # For stream events - indicates final chunk


class Agent:
    """The main agent that orchestrates the LLM and tools."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        project_root: Path | str | None = None,
    ):
        self.backend = backend
        self.registry = registry or create_default_registry()
        self.config = config or AgentConfig()
        self.messages: list[Message] = []
        self._system_message: Message | None = None
        self._use_react: bool | None = None

        # Recovery tracking
        self._recovery_context: RecoveryContext | None = None

        # Load project context if enabled
        self.project_context: ProjectContext | None = None
        if self.config.auto_context:
            self.project_context = detect_project(project_root)

    @property
    def use_react(self) -> bool:
        """Determine whether to use ReAct prompting or native tools."""
        if self._use_react is not None:
            return self._use_react

        if self.config.force_react:
            self._use_react = True
            return True

        # Check if backend supports native tools
        if hasattr(self.backend, "supports_native_tools"):
            self._use_react = not self.backend.supports_native_tools()
        else:
            # Default to ReAct for unknown backends
            self._use_react = True

        return self._use_react

    def _get_system_message(self) -> Message:
        """Get the system message with current context."""
        if self._system_message is None:
            tool_schemas = self.registry.get_schemas()

            # Pass ProjectContext directly for project-specific tips
            content = build_system_prompt(
                tools=tool_schemas,
                use_react=self.use_react,
                project_context=self.project_context,
            )
            self._system_message = Message(
                role=Role.SYSTEM,
                content=content,
            )
        return self._system_message

    def _build_messages(self) -> list[Message]:
        """Build the full message list for the LLM."""
        return [self._get_system_message()] + self.messages

    async def _should_plan(self, task: str) -> bool:
        """Ask LLM if this task needs planning."""
        if not self.config.auto_plan:
            return False

        prompt = SHOULD_PLAN_PROMPT.format(task=task)
        response = await self.backend.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            temperature=0.3,
            max_tokens=20,
        )
        return should_plan(response.content)

    async def _create_plan(self, task: str) -> Plan:
        """Generate a plan for the task."""
        prompt = PLANNING_PROMPT.format(task=task)
        response = await self.backend.complete(
            messages=[
                self._get_system_message(),
                Message(role=Role.USER, content=prompt),
            ],
            tools=None,
            temperature=0.5,
            max_tokens=500,
        )
        return parse_plan(response.content, goal=task)

    async def run(
        self,
        user_message: str,
        on_event: Callable[[AgentEvent], None] | None = None,
        use_plan: bool | None = None,
    ) -> str:
        """Run the agent with a user message.

        Args:
            user_message: The user's input
            on_event: Optional callback for streaming events
            use_plan: Force planning on/off. None = auto-detect.

        Returns:
            The final response text
        """
        def emit(event: AgentEvent) -> None:
            if on_event:
                on_event(event)

        # Check if we should use planning
        should_use_plan = use_plan
        if should_use_plan is None and self.config.auto_plan:
            emit(AgentEvent(type="thinking"))
            should_use_plan = await self._should_plan(user_message)

        # If planning, create and execute plan
        if should_use_plan:
            plan = await self._create_plan(user_message)
            if plan.steps:
                emit(AgentEvent(type="plan", content=plan.to_prompt()))

                # Execute each step
                while not plan.is_complete():
                    step = plan.next_step()
                    if not step:
                        break

                    emit(AgentEvent(
                        type="step",
                        step_info=f"{plan.progress_str()} {step.description}",
                    ))

                    # Run the step
                    step_prompt = format_step_prompt(plan, step)
                    step_response = await self._run_inner(step_prompt, emit)

                    plan.complete_current()

                # Final summary
                self.messages.append(Message(role=Role.USER, content=user_message))
                summary_prompt = f"I've completed the plan. Summarize what was done:\n{plan.to_prompt()}"
                return await self._run_inner(summary_prompt, emit)

        # No planning - run directly
        self.messages.append(Message(role=Role.USER, content=user_message))
        return await self._run_inner(user_message, emit)

    async def _run_inner(
        self,
        task: str,
        emit: Callable[[AgentEvent], None],
    ) -> str:
        """Inner execution loop without planning."""
        iterations = 0
        final_response = ""

        while iterations < self.config.max_iterations:
            iterations += 1

            # Get completion from LLM
            emit(AgentEvent(type="thinking"))

            # Pass tools only for native tool calling
            tools = None if self.use_react else self.registry.get_schemas()

            # Use streaming or regular completion
            if self.config.stream:
                full_content = ""
                tool_calls: list[ToolCall] = []

                async for chunk in self.backend.stream(
                    messages=self._build_messages(),
                    tools=tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ):
                    if chunk.content:
                        emit(AgentEvent(
                            type="stream",
                            content=chunk.content,
                            is_stream_end=chunk.is_done,
                        ))
                    if chunk.is_done:
                        full_content = chunk.full_content or full_content
                        tool_calls = chunk.tool_calls

                content = full_content
                response_content = full_content
            else:
                response = await self.backend.complete(
                    messages=self._build_messages(),
                    tools=tools,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = response.content
                response_content = response.content
                tool_calls = response.tool_calls if not self.use_react else []

            # Get tool calls - either native or parsed from text
            if self.use_react:
                # Parse tool calls from text (ReAct mode)
                parsed = parse_tool_calls(content)
                tool_calls = parsed.tool_calls
                content = parsed.content

                # Check if this is a final answer
                if parsed.is_final_answer and not tool_calls:
                    final_response = content
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=response_content,  # Keep original for history
                    ))
                    emit(AgentEvent(type="response", content=final_response))
                    break

            # If there are tool calls, execute them
            if tool_calls:
                # Add assistant message with tool calls
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=response_content,
                    tool_calls=tool_calls,
                ))

                # Execute each tool (with recovery logic)
                for tool_call in tool_calls:
                    emit(AgentEvent(
                        type="tool_call",
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments,
                    ))

                    result = await self.registry.execute(
                        tool_call.name,
                        **tool_call.arguments,
                    )

                    # Handle errors with recovery
                    if result.is_error and self.config.auto_recover:
                        # Initialize or update recovery context
                        if self._recovery_context is None:
                            self._recovery_context = RecoveryContext(
                                original_tool=tool_call.name,
                                original_args=tool_call.arguments,
                                max_retries=self.config.max_recovery_attempts,
                            )

                        # Check if this exact call was already tried (loop detection)
                        if self._recovery_context.was_tried(tool_call.name, tool_call.arguments):
                            emit(AgentEvent(
                                type="error",
                                content=f"Loop detected: already tried {tool_call.name} with same args",
                                tool_name=tool_call.name,
                            ))
                        else:
                            # Record this attempt
                            self._recovery_context.add_attempt(
                                tool_call.name,
                                tool_call.arguments,
                                result.output,
                            )

                        # Can we retry?
                        if self._recovery_context.can_retry():
                            attempt_num = len(self._recovery_context.attempts)
                            emit(AgentEvent(
                                type="recovery",
                                content=f"Tool failed, attempting recovery ({attempt_num}/{self._recovery_context.max_retries})",
                                tool_name=tool_call.name,
                                recovery_attempt=attempt_num,
                            ))

                            # Add recovery prompt for LLM
                            recovery_prompt = format_recovery_prompt(
                                self._recovery_context,
                                tool_call.name,
                                tool_call.arguments,
                                result.output,
                            )
                            self.messages.append(Message(
                                role=Role.TOOL,
                                content=recovery_prompt,
                            ))

                            # Continue to let LLM try an alternative
                            continue
                        else:
                            # Max retries exceeded
                            failure_msg = format_failure_message(self._recovery_context)
                            emit(AgentEvent(
                                type="error",
                                content=failure_msg,
                                tool_name=tool_call.name,
                            ))
                            self._recovery_context = None

                            # Add the final error result
                            result_text = format_tool_result(
                                tool_call.name,
                                failure_msg,
                                is_error=True,
                            )
                            self.messages.append(Message(
                                role=Role.TOOL,
                                content=result_text,
                            ))
                            continue
                    else:
                        # Success or no auto-recover - clear recovery context
                        if not result.is_error:
                            self._recovery_context = None

                    emit(AgentEvent(
                        type="tool_result",
                        content=result.output,
                        tool_name=tool_call.name,
                    ))

                    # Add tool result message
                    result_text = format_tool_result(
                        tool_call.name,
                        result.output,
                        result.is_error,
                    )
                    self.messages.append(Message(
                        role=Role.TOOL,
                        content=result_text,
                    ))

                # Continue the loop to get next response
                continue

            # No tool calls - this is the final response
            final_response = content
            self.messages.append(Message(
                role=Role.ASSISTANT,
                content=response_content,
            ))

            emit(AgentEvent(type="response", content=final_response))
            break

        return final_response

    async def run_streaming(
        self,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent with streaming output."""
        # Add user message
        self.messages.append(Message(role=Role.USER, content=user_message))

        iterations = 0
        tools = None if self.use_react else self.registry.get_schemas()

        while iterations < self.config.max_iterations:
            iterations += 1

            yield AgentEvent(type="thinking")

            # Stream the response
            full_content = ""
            tool_calls: list[ToolCall] = []

            async for chunk in self.backend.stream(
                messages=self._build_messages(),
                tools=tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ):
                if chunk.content:
                    full_content += chunk.content
                    yield AgentEvent(type="response", content=chunk.content)

                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

            # In ReAct mode, parse tool calls from text
            if self.use_react:
                parsed = parse_tool_calls(full_content)
                tool_calls = parsed.tool_calls

                if parsed.is_final_answer and not tool_calls:
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=full_content,
                    ))
                    break

            # If there are tool calls, execute them
            if tool_calls:
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=full_content,
                    tool_calls=tool_calls,
                ))

                for tool_call in tool_calls:
                    yield AgentEvent(
                        type="tool_call",
                        tool_name=tool_call.name,
                        tool_args=tool_call.arguments,
                    )

                    result = await self.registry.execute(
                        tool_call.name,
                        **tool_call.arguments,
                    )

                    yield AgentEvent(
                        type="tool_result",
                        content=result.output,
                        tool_name=tool_call.name,
                    )

                    result_text = format_tool_result(
                        tool_call.name,
                        result.output,
                        result.is_error,
                    )
                    self.messages.append(Message(
                        role=Role.TOOL,
                        content=result_text,
                    ))

                continue

            # No tool calls - done
            self.messages.append(Message(
                role=Role.ASSISTANT,
                content=full_content,
            ))
            break

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self._recovery_context = None
