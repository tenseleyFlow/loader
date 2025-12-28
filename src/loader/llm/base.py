"""Base classes for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Any


class Role(str, Enum):
    """Message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    """A message in the conversation."""
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API calls."""
        result: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        return result


@dataclass
class StreamChunk:
    """A chunk of streaming response."""
    content: str = ""
    full_content: str = ""  # Accumulated full content (only set when is_done=True)
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_done: bool = False
    # Pending tool call detected during streaming (ReAct mode)
    # This allows showing tool widgets as they're detected, before streaming ends
    pending_tool_call: ToolCall | None = None


@dataclass
class CompletionResponse:
    """Complete response from LLM."""
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        """Generate a completion."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is available."""
        ...
