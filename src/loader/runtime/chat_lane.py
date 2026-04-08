"""Runtime-owned conversational fast path for non-tool chat turns."""

from __future__ import annotations

from ..llm.base import Message, Role
from .bootstrap import RuntimeBootstrapSource
from .events import AgentEvent

CHAT_SYSTEM_PROMPT = (
    "You are Loader, a friendly local coding assistant. "
    "Respond naturally and briefly to conversational messages. "
    "If the user wants to do a coding task, tell them to describe it. "
    "Keep responses short (1-3 sentences)."
)


class ConversationalTurnRunner:
    """Own the non-tool conversational fast path outside the agent shell."""

    def __init__(self, source: RuntimeBootstrapSource) -> None:
        self.source = source

    async def run(self, user_message: str, emit) -> str:
        """Stream one short conversational reply and persist the transcript."""

        await emit(AgentEvent(type="thinking"))
        self.source.session.append(Message(role=Role.USER, content=user_message))

        recent_messages = (
            self.source.session.messages[-4:]
            if len(self.source.session.messages) > 4
            else self.source.session.messages
        )
        chat_system = Message(role=Role.SYSTEM, content=CHAT_SYSTEM_PROMPT)

        full_content = ""
        async for chunk in self.source.backend.stream(
            messages=[chat_system] + recent_messages,
            tools=None,
            temperature=0.7,
            max_tokens=256,
        ):
            if chunk.content:
                await emit(
                    AgentEvent(
                        type="stream",
                        content=chunk.content,
                        is_stream_end=chunk.is_done,
                    )
                )
                full_content += chunk.content

        self.source.session.append(Message(role=Role.ASSISTANT, content=full_content))
        await emit(AgentEvent(type="response", content=full_content))
        return full_content
