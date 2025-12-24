"""The main agent loop."""

import os
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from ..llm.base import LLMBackend, Message, Role, ToolCall
from ..tools.base import ToolRegistry, create_default_registry


SYSTEM_PROMPT = """You are Loader, a helpful AI coding assistant running locally on the user's machine.

You have access to tools to help you accomplish tasks:
- read: Read file contents
- write: Write content to files
- edit: Edit files by replacing text
- glob: Find files matching patterns
- grep: Search for patterns in code
- bash: Execute shell commands

When the user asks you to do something:
1. Think about what you need to do
2. Use tools to gather information or make changes
3. Explain what you did and the results

Be concise but thorough. When editing code, make sure to read the file first to understand the context.

Current working directory: {cwd}
"""


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    max_iterations: int = 20
    temperature: float = 0.7
    max_tokens: int = 4096


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

    def _get_system_message(self) -> Message:
        """Get the system message with current context."""
        if self._system_message is None:
            cwd = os.getcwd()
            self._system_message = Message(
                role=Role.SYSTEM,
                content=SYSTEM_PROMPT.format(cwd=cwd),
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

            response = await self.backend.complete(
                messages=self._build_messages(),
                tools=self.registry.get_schemas(),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # If there are tool calls, execute them
            if response.tool_calls:
                # Add assistant message with tool calls
                self.messages.append(Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                # Execute each tool
                for tool_call in response.tool_calls:
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
                    self.messages.append(Message(
                        role=Role.TOOL,
                        content=f"[{tool_call.name}] {result.output}",
                    ))

                # Continue the loop to get next response
                continue

            # No tool calls - this is the final response
            final_response = response.content
            self.messages.append(Message(
                role=Role.ASSISTANT,
                content=final_response,
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

        while iterations < self.config.max_iterations:
            iterations += 1

            yield AgentEvent(type="thinking")

            # Stream the response
            full_content = ""
            tool_calls: list[ToolCall] = []

            async for chunk in self.backend.stream(
                messages=self._build_messages(),
                tools=self.registry.get_schemas(),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ):
                if chunk.content:
                    full_content += chunk.content
                    yield AgentEvent(type="response", content=chunk.content)

                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls

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

                    self.messages.append(Message(
                        role=Role.TOOL,
                        content=f"[{tool_call.name}] {result.output}",
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
