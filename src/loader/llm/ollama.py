"""Ollama backend implementation."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..agent.parsing import parse_tool_calls
from ..runtime.capabilities import CapabilityProfile, resolve_capability_profile
from .base import (
    CompletionResponse,
    LLMBackend,
    Message,
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
        num_ctx: int = 16384,  # 16K context; most models support 32K+
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
        self._model_details_loaded_for: str | None = None
        self._capability_profile: CapabilityProfile | None = None

    def _invalidate_model_caches_if_needed(self) -> None:
        """Clear cached capability state when the active model changes."""

        if (
            self._capability_profile is not None
            and self._capability_profile.model_name != self.model
        ):
            self._capability_profile = None
            self._supports_native_tools = None

        if (
            self._model_details_loaded_for is not None
            and self._model_details_loaded_for != self.model
        ):
            self._model_details_cache = None
            self._model_details_loaded_for = None

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

    async def chat_health_check(self) -> tuple[bool, str | None]:
        """Probe whether the live chat endpoint can complete a minimal request."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "stream": False,
            "options": self._build_options(temperature=0.0, max_tokens=8),
        }

        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            if response.status_code == 400:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", "Bad request")
                return False, f"Ollama /api/chat rejected the probe: {error_msg}"
            response.raise_for_status()
        except Exception as exc:
            return False, str(exc)

        return True, None

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

        self._invalidate_model_caches_if_needed()

        if self._model_details_loaded_for == self.model:
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
            self._model_details_loaded_for = self.model
            self._capability_profile = None
        except Exception:
            self._model_details_cache = None
            self._model_details_loaded_for = self.model
            self._capability_profile = None

        return self._model_details_cache

    def capability_profile(self) -> CapabilityProfile:
        """Return the resolved capability profile for the current model."""

        self._invalidate_model_caches_if_needed()
        if self._capability_profile is None:
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

        self._invalidate_model_caches_if_needed()

        if self._supports_native_tools is not None:
            return self._supports_native_tools

        self._supports_native_tools = self.capability_profile().supports_native_tools
        return self._supports_native_tools

    async def probe_native_tool_support(
        self,
        temperature: float = 0.3,
        rounds: int = 3,
        required_passes: int = 3,
        tools: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Probe whether the model *reliably* produces native tool calls.

        Runs ``rounds`` probe calls at the actual runtime ``temperature``
        using the real tool schemas (when provided).  The model must produce
        ``tool_calls`` in at least ``required_passes`` of those rounds.
        """
        if self.force_react:
            self._supports_native_tools = False
            return False

        if tools:
            probe_tools = self._format_tools(tools)
        else:
            probe_tools = [{
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }]

        passes = 0
        for i in range(rounds):
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Run: echo hello"},
                ],
                "tools": probe_tools,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 128,
                    "num_ctx": 4096,
                },
            }
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/chat", json=payload,
                )
                if response.status_code == 400:
                    error = response.json().get("error", "")
                    if "does not support tools" in error:
                        self._supports_native_tools = False
                        self._debug_log("probe: model rejected tools (400)")
                        return False
                response.raise_for_status()
                data = response.json()
                message = data.get("message", {})
                if message.get("tool_calls"):
                    passes += 1
            except Exception:
                pass  # treat failures as non-passes

        native = passes >= required_passes
        self._supports_native_tools = native
        self._debug_log(
            f"probe_native_tool_support: {passes}/{rounds} passed "
            f"(need {required_passes}) → {'native' if native else 'react'}"
        )
        return native

    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for Ollama API.

        Ollama expects tool_calls wrapped in ``{"function": {...}}`` and tool
        results with ``role: "tool"``.  The generic ``Message.to_dict()``
        uses a flat layout, so we re-wrap here.
        """
        formatted = []
        for message in messages:
            entry: dict[str, Any] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_results:
                # Ollama expects tool results as role=tool with the content
                entry["role"] = "tool"
            formatted.append(entry)
        return formatted

    def _format_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Format tools for Ollama API.

        Parameter-level ``description`` fields are stripped because several
        Ollama model renderers (qwen3-coder, qwen2) emit text-based tool
        calls instead of structured ``tool_calls`` when descriptions are
        present in the schema.
        """
        if not tools:
            return None

        formatted = []
        for tool in tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": self._strip_param_descriptions(
                        tool.get("parameters", {}),
                    ),
                },
            })
        return formatted

    @staticmethod
    def _strip_param_descriptions(params: dict[str, Any]) -> dict[str, Any]:
        """Remove description fields from parameter properties."""
        if "properties" not in params:
            return params
        cleaned: dict[str, Any] = {k: v for k, v in params.items() if k != "properties"}
        cleaned["properties"] = {}
        for name, prop in params["properties"].items():
            cleaned["properties"][name] = {
                k: v for k, v in prop.items() if k != "description"
            }
        return cleaned

    @staticmethod
    def _allowed_tool_names(tools: list[dict[str, Any]] | None) -> list[str] | None:
        """Return the tool names currently exposed to the model, if any."""

        if not tools:
            return None
        names = [
            str(tool.get("name", "")).strip()
            for tool in tools
            if isinstance(tool, dict) and tool.get("name")
        ]
        return names or None

    def _parse_tool_calls(
        self,
        response_text: str,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Parse text tool calls through the shared parser."""

        parsed = parse_tool_calls(
            response_text,
            allowed_tool_names=self._allowed_tool_names(tools),
        )
        return parsed.content, parsed.tool_calls

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        """Generate a completion using Ollama."""
        await self.describe_model()

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
            content, tool_calls = self._parse_tool_calls(content, tools=tools)

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
        await self.describe_model()

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
                async for chunk in self._stream_response(response, tools=tools):
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
            async for chunk in self._stream_response(response, tools=tools):
                yield chunk

    def _debug_log(self, message: str) -> None:
        """Write debug message to log file."""
        try:
            with open("/tmp/loader_debug.log", "a") as f:
                f.write(f"[ollama] {message}\n")
        except Exception:
            pass

    async def _stream_response(
        self,
        response,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Internal helper to stream response chunks."""

        full_content = ""
        display_content = ""  # Content to show (filtered)
        in_think_block = False  # For reasoning models like deepseek-r1
        in_tool_call_block = False  # For ReAct <tool_call> tags
        # Ollama sends native tool_calls in non-final chunks; collect them.
        accumulated_tool_calls: list[ToolCall] = []

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

            # Collect native tool_calls from non-final chunks
            if not is_done and "tool_calls" in message:
                for i, tc in enumerate(message["tool_calls"]):
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    accumulated_tool_calls.append(ToolCall(
                        id=tc.get("id", f"call_{len(accumulated_tool_calls)}"),
                        name=func.get("name", ""),
                        arguments=args,
                    ))
                continue

            if is_done:
                tool_calls: list[ToolCall] = []
                # Check for native tool calls in the final chunk
                if "tool_calls" in message:
                    self._debug_log(f"is_done: found native tool_calls in message: {len(message['tool_calls'])}")
                    for i, tc in enumerate(message["tool_calls"]):
                        func = tc.get("function", {})
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

                # Use accumulated tool_calls from earlier chunks
                if not tool_calls and accumulated_tool_calls:
                    tool_calls = accumulated_tool_calls

                if not tool_calls:
                    self._debug_log(f"is_done: parsing tool calls from text (len={len(full_content)})")
                    self._debug_log(f"is_done: full_content = {repr(full_content[:500])}")
                    clean_content, tool_calls = self._parse_tool_calls(
                        full_content,
                        tools=tools,
                    )
                    self._debug_log(f"is_done: parsed {len(tool_calls)} tool calls")
                    display_content = clean_content

                self._debug_log(f"is_done: yielding final chunk with {len(tool_calls)} tool_calls")
                yield StreamChunk(
                    content="",  # Don't emit final chunk content (already streamed)
                    full_content=display_content,
                    tool_calls=tool_calls,
                    is_done=True,
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                    },
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
                    # Keep content before <tool_call>
                    before = chunk_content.split("<tool_call>")[0]
                    if before:
                        display_content += before
                        yield StreamChunk(content=before)
                    # Start buffering the tool call content
                    after_tag = chunk_content.split("<tool_call>", 1)[-1]
                    if "</tool_call>" in after_tag:
                        after_close = after_tag.split("</tool_call>", 1)[-1]
                        in_tool_call_block = False
                        if after_close.strip():
                            display_content += after_close
                            yield StreamChunk(content=after_close)
                    continue
                elif in_tool_call_block:
                    if "</tool_call>" in chunk_content:
                        in_tool_call_block = False
                        after_close = chunk_content.split("</tool_call>", 1)[-1]
                        if after_close.strip():
                            display_content += after_close
                            yield StreamChunk(content=after_close)
                    continue

                display_content += chunk_content
                yield StreamChunk(content=chunk_content)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
