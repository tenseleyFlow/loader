"""Ollama backend implementation."""

import json
from typing import Any, AsyncIterator

import httpx

from .base import (
    CompletionResponse,
    LLMBackend,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)


class OllamaBackend(LLMBackend):
    """Ollama API backend for local LLM inference."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def health_check(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            # Check if our model (or base name) is available
            return any(self.model in m or m in self.model for m in models)
        except Exception:
            return False

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for Ollama API."""
        formatted = []
        for msg in messages:
            formatted.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        return formatted

    def _format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for Ollama API."""
        if not tools:
            return None

        # Ollama uses a slightly different format
        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            })
        return formatted

    def _parse_tool_calls(self, response_text: str) -> tuple[str, list[ToolCall]]:
        """Parse tool calls from response text.

        Models may format tool calls differently. We handle:
        1. JSON tool call blocks
        2. XML-style <tool_call> blocks
        3. Plain text with no tool calls
        """
        tool_calls = []
        content = response_text

        # Try to find JSON tool calls (common format)
        # Look for patterns like: {"name": "tool_name", "arguments": {...}}
        import re

        # Pattern for tool call JSON blocks
        json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        for i, (name, args_str) in enumerate(matches):
            try:
                args = json.loads(args_str)
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=name,
                    arguments=args,
                ))
                # Remove the tool call from content
                content = re.sub(json_pattern, "", content, count=1)
            except json.JSONDecodeError:
                pass

        return content.strip(), tool_calls

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        """Generate a completion using Ollama."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        response = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")

        # Check for native tool calls first
        tool_calls = []
        if "tool_calls" in message:
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                ))
        else:
            # Try to parse tool calls from text
            content, tool_calls = self._parse_tool_calls(content)

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from Ollama."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()

            full_content = ""
            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                message = data.get("message", {})
                chunk_content = message.get("content", "")
                full_content += chunk_content

                is_done = data.get("done", False)

                if is_done:
                    # Parse any tool calls from the full response
                    _, tool_calls = self._parse_tool_calls(full_content)
                    yield StreamChunk(
                        content=chunk_content,
                        tool_calls=tool_calls,
                        is_done=True,
                    )
                else:
                    yield StreamChunk(content=chunk_content)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
