"""Assistant-turn request handling for the typed runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..llm.base import ToolCall
from .context import RuntimeContext
from .events import AgentEvent
from .tracing import RuntimeTracer

EventSink = Callable[[AgentEvent], Awaitable[None]]


@dataclass
class AssistantTurn:
    """Assistant output for one iteration of the conversation loop."""

    content: str
    response_content: str
    tool_calls: list[ToolCall]
    pending_tool_calls_seen: set[str] = field(default_factory=set)
    usage: dict[str, int] = field(default_factory=dict)


class AssistantTurnRequester:
    """Encapsulates assistant request/response handling for one runtime turn."""

    def __init__(self, context: RuntimeContext, tracer: RuntimeTracer) -> None:
        self.context = context
        self.tracer = tracer

    async def request_turn(
        self,
        *,
        emit: EventSink,
        max_tokens: int,
    ) -> AssistantTurn:
        """Request one assistant turn through streaming or complete mode."""

        self.context.safeguards.code_filter.reset()
        compaction = self.context.session.maybe_compact()
        if compaction is not None:
            await emit(
                AgentEvent(
                    type="artifact",
                    content=(
                        f"Compacted {compaction.removed_message_count} older message(s) "
                        "into a continuation summary.\n"
                        f"Input tokens: {compaction.original_input_tokens} -> "
                        f"{compaction.compressed_input_tokens}"
                    ),
                    artifact_kind="session_compaction",
                    artifact_path=str(self.context.session.storage_path),
                )
            )
        tools = None if self.context.use_react else self.context.registry.get_schemas()
        self.tracer.record(
            "assistant.requested",
            use_react=self.context.use_react,
            stream=self.context.config.stream,
            max_tokens=max_tokens,
        )

        if self.context.config.stream:
            full_content = ""
            full_content_unfiltered = ""
            tool_calls: list[ToolCall] = []
            pending_tool_calls_seen: set[str] = set()
            usage: dict[str, int] = {}

            async for chunk in self.context.backend.stream(
                messages=self.context.session.build_request_messages(),
                tools=tools,
                temperature=self.context.config.temperature,
                max_tokens=max_tokens,
            ):
                filtered_content = ""
                if chunk.content:
                    filtered_content = self.context.safeguards.filter_stream_chunk(chunk.content)
                    full_content_unfiltered += chunk.content

                if filtered_content or chunk.is_done:
                    await emit(
                        AgentEvent(
                            type="stream",
                            content=filtered_content,
                            is_stream_end=chunk.is_done,
                        )
                    )

                if self.context.safeguards.should_steer():
                    steering_message = self.context.safeguards.get_steering_message()
                    if steering_message:
                        self.context.legacy.queue_steering_message(steering_message)

                if (
                    chunk.pending_tool_call
                    and chunk.pending_tool_call.id not in pending_tool_calls_seen
                ):
                    pending_tool_calls_seen.add(chunk.pending_tool_call.id)
                    await emit(
                        AgentEvent(
                            type="tool_call",
                            tool_name=chunk.pending_tool_call.name,
                            tool_args=chunk.pending_tool_call.arguments,
                            phase="assistant",
                        )
                    )

                if chunk.is_done:
                    full_content = chunk.full_content or full_content_unfiltered
                    tool_calls = chunk.tool_calls
                    usage = chunk.usage

            self.tracer.record(
                "assistant.responded",
                stream=True,
                tool_call_count=len(tool_calls),
                content_length=len(full_content),
            )
            return AssistantTurn(
                content=full_content,
                response_content=full_content,
                tool_calls=tool_calls,
                pending_tool_calls_seen=pending_tool_calls_seen,
                usage=usage,
            )

        response = await self.context.backend.complete(
            messages=self.context.session.build_request_messages(),
            tools=tools,
            temperature=self.context.config.temperature,
            max_tokens=max_tokens,
        )
        response_content = response.content
        content = self.context.safeguards.filter_complete_content(response.content)
        tool_calls = response.tool_calls if not self.context.use_react else []
        if self.context.safeguards.should_steer():
            steering_message = self.context.safeguards.get_steering_message()
            if steering_message:
                self.context.legacy.queue_steering_message(steering_message)
        self.tracer.record(
            "assistant.responded",
            stream=False,
            tool_call_count=len(tool_calls),
            content_length=len(content),
        )
        return AssistantTurn(
            content=content,
            response_content=response_content,
            tool_calls=tool_calls,
            usage=response.usage,
        )
