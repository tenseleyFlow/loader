"""Base classes for LLM backends."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize one tool call for persistence."""

        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Restore one tool call from persisted data."""

        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            arguments=dict(data.get("arguments", {})),
        )


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_call_id: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize one tool result for persistence."""

        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        """Restore one tool result from persisted data."""

        return cls(
            tool_call_id=str(data["tool_call_id"]),
            content=str(data.get("content", "")),
            is_error=bool(data.get("is_error", False)),
        )


@dataclass
class Message:
    """A message in the conversation."""
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    @classmethod
    def tool_result_message(
        cls,
        *,
        tool_call_id: str,
        display_content: str,
        result_content: str,
        is_error: bool = False,
    ) -> "Message":
        """Build a tool-result message with a typed tool result payload."""

        return cls(
            role=Role.TOOL,
            content=display_content,
            tool_results=[
                ToolResult(
                    tool_call_id=tool_call_id,
                    content=result_content,
                    is_error=is_error,
                )
            ],
        )

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
        if self.tool_results:
            primary_result = self.tool_results[0]
            result["tool_call_id"] = primary_result.tool_call_id
            result["is_error"] = primary_result.is_error
        return result

    def to_persisted_dict(self) -> dict[str, Any]:
        """Serialize the full message payload for session persistence."""

        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "tool_results": [
                tool_result.to_dict() for tool_result in self.tool_results
            ],
        }

    @classmethod
    def from_persisted_dict(cls, data: dict[str, Any]) -> "Message":
        """Restore a persisted conversation message."""

        return cls(
            role=Role(str(data["role"])),
            content=str(data.get("content", "")),
            tool_calls=[
                ToolCall.from_dict(item) for item in data.get("tool_calls", [])
            ],
            tool_results=[
                ToolResult.from_dict(item) for item in data.get("tool_results", [])
            ],
        )


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
    usage: dict[str, int] = field(default_factory=dict)


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
