"""Deterministic runtime harness utilities for Loader tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loader.agent.loop import AgentConfig
from loader.llm.base import CompletionResponse, LLMBackend, Message, StreamChunk
from loader.runtime.events import AgentEvent
from loader.runtime.runtime_handle import RuntimeHandle
from loader.tools.base import ToolRegistry, create_default_registry


@dataclass
class BackendInvocation:
    """Record of one backend request made by the agent."""

    mode: str
    messages: list[Message]
    tools: list[dict[str, Any]] | None
    temperature: float
    max_tokens: int


@dataclass
class ScenarioRun:
    """Captured result of a scripted runtime-owner scenario."""

    response: str
    events: list[AgentEvent]
    invocations: list[BackendInvocation]
    agent: RuntimeHandle


class ScriptedBackend(LLMBackend):
    """LLM backend that replays scripted completions or stream chunks."""

    def __init__(
        self,
        *,
        completions: list[CompletionResponse] | None = None,
        streams: list[list[StreamChunk]] | None = None,
        supports_native_tools: bool = True,
    ) -> None:
        self._completions = list(completions or [])
        self._streams = list(streams or [])
        self._supports_native_tools = supports_native_tools
        self.invocations: list[BackendInvocation] = []

    def supports_native_tools(self) -> bool:
        """Mirror Ollama's native-tool capability surface."""

        return self._supports_native_tools

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> CompletionResponse:
        self.invocations.append(
            BackendInvocation(
                mode="complete",
                messages=list(messages),
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not self._completions:
            return CompletionResponse(content="Done.")
        return self._completions.pop(0)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.invocations.append(
            BackendInvocation(
                mode="stream",
                messages=list(messages),
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        if not self._streams:
            raise AssertionError("No scripted stream left for this scenario")
        for chunk in self._streams.pop(0):
            yield chunk

    async def health_check(self) -> bool:
        return True


async def run_scenario(
    prompt: str,
    backend: ScriptedBackend,
    *,
    registry: ToolRegistry | None = None,
    config: AgentConfig | None = None,
    project_root: Path | str | None = None,
    on_confirmation=None,
    on_user_question=None,
) -> ScenarioRun:
    """Run a scripted runtime scenario and collect emitted events."""

    agent = RuntimeHandle(
        backend=backend,
        registry=registry or create_default_registry(project_root),
        config=config or AgentConfig(auto_context=False),
        project_root=project_root,
    )
    events: list[AgentEvent] = []

    async def capture(event: AgentEvent) -> None:
        events.append(event)

    response = await agent.run(
        prompt,
        on_event=capture,
        on_confirmation=on_confirmation,
        on_user_question=on_user_question,
    )
    return ScenarioRun(
        response=response,
        events=events,
        invocations=list(backend.invocations),
        agent=agent,
    )


async def run_explore_scenario(
    prompt: str,
    backend: ScriptedBackend,
    *,
    config: AgentConfig | None = None,
    project_root: Path | str | None = None,
) -> ScenarioRun:
    """Run a scripted explore query through the runtime-first harness."""

    agent = RuntimeHandle(
        backend=backend,
        config=config or AgentConfig(auto_context=False),
        project_root=project_root,
    )
    events: list[AgentEvent] = []

    async def capture(event: AgentEvent) -> None:
        events.append(event)

    response = await agent.run_explore(
        prompt,
        on_event=capture,
    )
    return ScenarioRun(
        response=response,
        events=events,
        invocations=list(backend.invocations),
        agent=agent,
    )
