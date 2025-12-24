"""The main agent loop."""

from dataclasses import dataclass
from typing import AsyncIterator, Callable

from ..llm.base import LLMBackend, Message, Role, ToolCall
from ..tools.base import ToolRegistry, create_default_registry
from .prompts import build_system_prompt
from .parsing import parse_tool_calls, format_tool_result


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_iterations: int = 20
    temperature: float = 0.7
    max_tokens: int = 4096
    force_react: bool = False  # Force ReAct even if model supports native tools


@dataclass
class AgentEvent:
    """Event emitted during agent execution."""
    type: str  # "thinking", "tool_call", "tool_result", "response", "error"
    content: str = ""
    tool_name: str | None = None
    tool_args: dict | None = None


class Agent:
    """The main agent that orchestrates the LLM and tools."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
    ):
        self.backend = backend
        self.registry = registry or create_default_registry()
        self.config = config or AgentConfig()
        self.messages: list[Message] = []
        self._system_message: Message | None = None
        self._use_react: bool | None = None

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
            content = build_system_prompt(
                tools=tool_schemas,
                use_react=self.use_react,
            )
            self._system_message = Message(
                role=Role.SYSTEM,
                content=content,
            )
        return self._system_message

    def _build_messages(self) -> list[Message]:
        """Build the full message list for the LLM."""
        return [self._get_system_message()] + self.messages

    async def run(
        self,
        user_message: str,
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> str:
        """Run the agent with a user message.

        Args:
            user_message: The user's input
            on_event: Optional callback for streaming events

        Returns:
            The final response text
        """
        # Add user message
        self.messages.append(Message(role=Role.USER, content=user_message))

        def emit(event: AgentEvent) -> None:
            if on_event:
                on_event(event)

        iterations = 0
        final_response = ""

        while iterations < self.config.max_iterations:
            iterations += 1

            # Get completion from LLM
            emit(AgentEvent(type="thinking"))

            # Pass tools only for native tool calling
            tools = None if self.use_react else self.registry.get_schemas()

            response = await self.backend.complete(
                messages=self._build_messages(),
                tools=tools,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Get tool calls - either native or parsed from text
            tool_calls: list[ToolCall] = []
            content = response.content

            if self.use_react:
                # Parse tool calls from text (ReAct mode)
                parsed = parse_tool_calls(response.content)
                tool_calls = parsed.tool_calls
                content = parsed.content

                # Check if this is a final answer
                if parsed.is_final_answer and not tool_calls:
                    final_response = content
                    self.messages.append(Message(
                        role=Role.ASSISTANT,
                        content=response.content,  # Keep original for history
                    ))
                    emit(AgentEvent(type="response", content=final_response))
                    break
            else:
                # Use native tool calls
                tool_calls = response.tool_calls

            # If there are tool calls, execute them
            if tool_calls:
                # Add assistant message with tool calls
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=tool_calls,
                ))

                # Execute each tool
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
                content=response.content,
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
