"""Deterministic runtime harness utilities for Loader tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loader.agent.loop import Agent, AgentConfig, AgentEvent
from loader.llm.base import CompletionResponse, LLMBackend, Message, StreamChunk
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
    """Captured result of a scripted agent scenario."""

    response: str
    events: list[AgentEvent]
    invocations: list[BackendInvocation]
    agent: Agent


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
            raise AssertionError("No scripted completion left for this scenario")
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
) -> ScenarioRun:
    """Run a scripted agent scenario and collect emitted events."""

    agent = Agent(
        backend=backend,
        registry=registry or create_default_registry(),
        config=config or AgentConfig(auto_context=False),
        project_root=project_root,
    )
    events: list[AgentEvent] = []

    async def capture(event: AgentEvent) -> None:
        events.append(event)

    response = await agent.run(prompt, on_event=capture, on_confirmation=on_confirmation)
    return ScenarioRun(
        response=response,
        events=events,
        invocations=list(backend.invocations),
        agent=agent,
    )
