"""LLM backend abstraction layer."""

from .base import LLMBackend, Message, ToolCall, ToolResult
from .ollama import OllamaBackend

__all__ = ["LLMBackend", "Message", "ToolCall", "ToolResult", "OllamaBackend"]
