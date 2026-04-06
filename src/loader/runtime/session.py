"""Conversation session ownership for runtime turns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..llm.base import Message


@dataclass
class ConversationSession:
    """Owns the conversation history used for turn execution."""

    system_message_factory: Callable[[], Message]
    few_shot_factory: Callable[[], list[Message]]
    messages: list[Message] = field(default_factory=list)

    def build_request_messages(self) -> list[Message]:
        """Build the full request transcript for the backend."""

        request_messages = [self.system_message_factory()]
        if len(self.messages) <= 2:
            request_messages.extend(self.few_shot_factory())
        request_messages.extend(self.messages)
        return request_messages

    def append(self, message: Message) -> None:
        """Append one message to the session."""

        self.messages.append(message)

    def extend(self, messages: list[Message]) -> None:
        """Append many messages to the session."""

        self.messages.extend(messages)

    def clear(self) -> None:
        """Reset the session history."""

        self.messages.clear()
