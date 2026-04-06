"""LLM backend abstraction layer."""

from typing import Any

from .base import LLMBackend, Message, ToolCall, ToolResult

__all__ = ["LLMBackend", "Message", "ToolCall", "ToolResult", "OllamaBackend"]


def __getattr__(name: str) -> Any:
    """Avoid importing the Ollama backend until a caller actually needs it."""

    if name == "OllamaBackend":
        from .ollama import OllamaBackend

        return OllamaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
