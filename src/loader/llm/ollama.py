"""Ollama backend implementation."""

import json
from typing import Any, AsyncIterator

import httpx

from ..runtime.capabilities import CapabilityProfile, resolve_capability_profile
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
        timeout: float | None = None,
        force_react: bool = False,
        num_ctx: int = 8192,  # Reasonable context, not too slow
        num_gpu: int = -1,  # Use all GPU layers by default (fast)
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Auto-adjust timeout based on model size
        if timeout is None:
            timeout = self._estimate_timeout(model)
        self.timeout = timeout
        self.force_react = force_react
        self.num_ctx = num_ctx
        self.num_gpu = num_gpu
        self._client = httpx.AsyncClient(timeout=timeout)
        self._supports_native_tools: bool | None = None
        self._model_details_cache: dict[str, Any] | None = None
        self._capability_profile: CapabilityProfile | None = None

    def _build_options(self, temperature: float, max_tokens: int) -> dict:
        """Build Ollama options dict with performance settings."""
        return {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
        }

    def _estimate_timeout(self, model: str) -> float:
        """Estimate appropriate timeout based on model size."""
        model_lower = model.lower()
        # Extract size from model name (e.g., "32b", "70b", "7b")
        import re
        size_match = re.search(r'(\d+)b', model_lower)
        if size_match:
            size = int(size_match.group(1))
            if size >= 70:
                return 900.0  # 15 minutes for 70B+
            elif size >= 30:
                return 600.0  # 10 minutes for 30B+
            elif size >= 13:
                return 300.0  # 5 minutes for 13B+
            elif size >= 7:
                return 180.0  # 3 minutes for 7B+
        return 120.0  # 2 minutes default

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

    async def list_models(self) -> list[dict[str, Any]]:
        """List all available models from Ollama.

        Returns:
            List of model info dicts with 'name', 'size', 'modified' keys.
        """
        try:
            response = await self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return []
            data = response.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m.get("name", ""),
                    "size": m.get("size", 0),
                    "modified": m.get("modified_at", ""),
                    "family": m.get("details", {}).get("family", ""),
                })
            return models
        except Exception:
            return []

    async def describe_model(self) -> dict[str, Any] | None:
        """Fetch and cache Ollama model details for capability resolution."""

        if self._model_details_cache is not None:
            return self._model_details_cache

        if not self.model:
            return None

        try:
            response = await self._client.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
            )
            response.raise_for_status()
            self._model_details_cache = response.json()
        except Exception:
            self._model_details_cache = None

        return self._model_details_cache

    def capability_profile(self) -> CapabilityProfile:
        """Return the resolved capability profile for the current model."""

        if (
            self._capability_profile is None
            or self._capability_profile.model_name != self.model
        ):
            self._capability_profile = resolve_capability_profile(
                self.model,
                model_details=self._model_details_cache,
            )
        return self._capability_profile

    def supports_native_tools(self) -> bool:
        """Check if current model supports native function calling.

        Returns:
            True if model likely supports native tool use, False otherwise
        """
        if self.force_react:
            return False

        if self._capability_profile is not None and self._capability_profile.model_name != self.model:
            self._capability_profile = None
            self._supports_native_tools = None

        if self._supports_native_tools is not None:
            return self._supports_native_tools

        self._supports_native_tools = self.capability_profile().supports_native_tools
        return self._supports_native_tools

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for Ollama API."""
        return [message.to_dict() for message in messages]

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

        import re

        # Pattern for tool call JSON blocks - handle both "arguments" and "parameters"
        json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"(?:arguments|parameters)"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        for i, (name, args_str) in enumerate(matches):
            try:
                args = json.loads(args_str)
                tool_calls.append(ToolCall(
                    id=f"call_{i}",
                    name=name,
                    arguments=args,
                ))
            except json.JSONDecodeError:
                pass

        # Remove tool call JSON from content
        if tool_calls:
            content = re.sub(json_pattern, "", content)

        # Also remove any <tool_call> tags
        content = re.sub(r"</?tool_call>", "", content)

        # Clean up whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

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
            "options": self._build_options(temperature, max_tokens),
        }

        # Only include tools if model supports them
        use_tools = tools and self.supports_native_tools()
        if use_tools:
            payload["tools"] = self._format_tools(tools)

        response = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )

        # Handle errors - gracefully fall back if tools not supported
        if response.status_code == 400:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", "Bad request")

            # If tools not supported, retry without them
            if "does not support tools" in error_msg and use_tools:
                self._supports_native_tools = False  # Remember for future calls
                payload.pop("tools", None)
                response = await self._client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
            else:
                raise ValueError(f"Ollama API error (400): {error_msg}")

        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = message.get("content", "")

        # Check for native tool calls first
        tool_calls = []
        if "tool_calls" in message:
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                # Arguments may be a JSON string or dict
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCall(
                    id=tc.get("id", f"call_{i}"),
                    name=func.get("name", ""),
                    arguments=args,
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
            "options": self._build_options(temperature, max_tokens),
        }

        # Only include tools if model supports them
        use_tools = tools and self.supports_native_tools()
        if use_tools:
            payload["tools"] = self._format_tools(tools)

        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            # Check for 400 errors before streaming
            if response.status_code == 400:
                content = await response.aread()
                try:
                    error_data = json.loads(content)
                    error_msg = error_data.get("error", "Bad request")
                except json.JSONDecodeError:
                    error_msg = content.decode() if content else "Bad request"

                # If tools not supported, we need to retry without them
                if "does not support tools" in error_msg and use_tools:
                    self._supports_native_tools = False  # Remember for future
                    # Fall through to retry below
                else:
                    raise ValueError(f"Ollama API error (400): {error_msg}")
            else:
                response.raise_for_status()
                # Stream the response
                async for chunk in self._stream_response(response):
                    yield chunk
                return

        # Retry without tools if we got here (tools not supported)
        payload.pop("tools", None)
        async with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for chunk in self._stream_response(response):
                yield chunk

    def _debug_log(self, message: str) -> None:
        """Write debug message to log file."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"[ollama] {message}\n")
        except Exception:
            pass

    async def _stream_response(self, response) -> AsyncIterator[StreamChunk]:
        """Internal helper to stream response chunks."""
        import re

        full_content = ""
        display_content = ""  # Content to show (filtered)
        json_buffer = ""  # Buffer for potential tool call JSON
        tool_call_buffer = ""  # Buffer for <tool_call> block content
        in_json_block = False
        in_think_block = False  # For reasoning models like deepseek-r1
        in_tool_call_block = False  # For ReAct <tool_call> tags
        detected_tool_calls: list[ToolCall] = []  # Track tool calls found during streaming
        tool_call_counter = 0

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
                tool_calls = []
                # Check for native tool calls first
                if "tool_calls" in message:
                    self._debug_log(f"is_done: found native tool_calls in message: {len(message['tool_calls'])}")
                    for i, tc in enumerate(message["tool_calls"]):
                        func = tc.get("function", {})
                        # Arguments may be a JSON string or dict
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        tool_calls.append(ToolCall(
                            id=tc.get("id", f"call_{i}"),
                            name=func.get("name", ""),
                            arguments=args,
                        ))
                else:
                    # Use detected tool calls from streaming, or parse from text
                    if detected_tool_calls:
                        self._debug_log(f"is_done: using {len(detected_tool_calls)} detected_tool_calls from streaming")
                        tool_calls = detected_tool_calls
                    else:
                        self._debug_log(f"is_done: parsing tool calls from text (len={len(full_content)})")
                        self._debug_log(f"is_done: full_content = {repr(full_content[:500])}")
                        clean_content, tool_calls = self._parse_tool_calls(full_content)
                        self._debug_log(f"is_done: parsed {len(tool_calls)} tool calls")
                        display_content = clean_content

                self._debug_log(f"is_done: yielding final chunk with {len(tool_calls)} tool_calls")
                yield StreamChunk(
                    content="",  # Don't emit final chunk content (already streamed)
                    full_content=display_content or full_content,
                    tool_calls=tool_calls,
                    is_done=True,
                )
            else:
                # Filter out <think> blocks from reasoning models (deepseek-r1, etc.)
                if "<think>" in chunk_content:
                    in_think_block = True
                    # Keep content before <think>
                    before = chunk_content.split("<think>")[0]
                    if before:
                        display_content += before
                        yield StreamChunk(content=before)
                    continue
                elif in_think_block:
                    if "</think>" in chunk_content:
                        in_think_block = False
                        # Keep content after </think>
                        after = chunk_content.split("</think>")[-1]
                        if after:
                            display_content += after
                            yield StreamChunk(content=after)
                    # Skip content inside think block
                    continue

                # Filter out <tool_call> blocks from ReAct mode - but parse them!
                if "<tool_call>" in chunk_content:
                    in_tool_call_block = True
                    tool_call_buffer = ""  # Reset buffer
                    # Keep content before <tool_call>
                    before = chunk_content.split("<tool_call>")[0]
                    if before:
                        display_content += before
                        yield StreamChunk(content=before)
                    # Start buffering the tool call content
                    after_tag = chunk_content.split("<tool_call>", 1)[-1]
                    if "</tool_call>" in after_tag:
                        # Complete tool call in same chunk
                        tool_json = after_tag.split("</tool_call>")[0]
                        after_close = after_tag.split("</tool_call>", 1)[-1]
                        in_tool_call_block = False
                        # Parse and yield the tool call
                        try:
                            tc_data = json.loads(tool_json.strip())
                            tc = ToolCall(
                                id=f"call_{tool_call_counter}",
                                name=tc_data.get("name", ""),
                                arguments=tc_data.get("arguments", tc_data.get("parameters", {})),
                            )
                            tool_call_counter += 1
                            detected_tool_calls.append(tc)
                            yield StreamChunk(content="", pending_tool_call=tc)
                        except (json.JSONDecodeError, KeyError):
                            pass
                        if after_close.strip():
                            display_content += after_close
                            yield StreamChunk(content=after_close)
                    else:
                        tool_call_buffer = after_tag
                    continue
                elif in_tool_call_block:
                    if "</tool_call>" in chunk_content:
                        in_tool_call_block = False
                        # Complete the tool call buffer
                        tool_json = tool_call_buffer + chunk_content.split("</tool_call>")[0]
                        after_close = chunk_content.split("</tool_call>", 1)[-1]
                        # Parse and yield the tool call
                        try:
                            tc_data = json.loads(tool_json.strip())
                            tc = ToolCall(
                                id=f"call_{tool_call_counter}",
                                name=tc_data.get("name", ""),
                                arguments=tc_data.get("arguments", tc_data.get("parameters", {})),
                            )
                            tool_call_counter += 1
                            detected_tool_calls.append(tc)
                            yield StreamChunk(content="", pending_tool_call=tc)
                        except (json.JSONDecodeError, KeyError):
                            pass
                        if after_close.strip():
                            display_content += after_close
                            yield StreamChunk(content=after_close)
                    else:
                        # Still accumulating tool call content
                        tool_call_buffer += chunk_content
                    continue

                # Filter out tool call JSON from display (bare JSON without tags)
                # Detect start of JSON tool call
                if not in_json_block and '{"name"' in chunk_content:
                    in_json_block = True
                    # Split at the JSON start
                    parts = chunk_content.split('{"name"', 1)
                    if parts[0]:
                        display_content += parts[0]
                        yield StreamChunk(content=parts[0])
                    json_buffer = '{"name"' + parts[1] if len(parts) > 1 else '{"name"'
                elif in_json_block:
                    json_buffer += chunk_content
                    # Check if JSON block closed (simple heuristic)
                    open_braces = json_buffer.count('{')
                    close_braces = json_buffer.count('}')
                    if close_braces >= open_braces and open_braces > 0:
                        # JSON block complete, try to parse it
                        in_json_block = False
                        try:
                            # Find where JSON ends
                            last_brace = json_buffer.rfind('}')
                            json_str = json_buffer[:last_brace + 1]
                            after_json = json_buffer[last_brace + 1:]
                            # Try to parse as tool call
                            tc_data = json.loads(json_str)
                            if "name" in tc_data:
                                tc = ToolCall(
                                    id=f"call_{tool_call_counter}",
                                    name=tc_data.get("name", ""),
                                    arguments=tc_data.get("arguments", tc_data.get("parameters", {})),
                                )
                                tool_call_counter += 1
                                detected_tool_calls.append(tc)
                                yield StreamChunk(content="", pending_tool_call=tc)
                            if after_json.strip():
                                display_content += after_json
                                yield StreamChunk(content=after_json)
                        except (json.JSONDecodeError, KeyError):
                            # Not valid JSON, just discard
                            pass
                        json_buffer = ""
                else:
                    display_content += chunk_content
                    yield StreamChunk(content=chunk_content)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
