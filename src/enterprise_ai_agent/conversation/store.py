"""Conversation persistence abstractions and in-memory implementation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .exceptions import (
    ConversationAlreadyExistsError,
    ConversationNotFoundError,
    InvalidMessageError,
)
from .models import Conversation, Message


class ConversationStore(ABC):
    """Replaceable interface for conversation persistence."""

    @abstractmethod
    def create(self, conversation: Conversation) -> None:
        """Store a new conversation."""

    @abstractmethod
    def get(self, conversation_id: str) -> Conversation | None:
        """Return a conversation when present."""

    @abstractmethod
    def append_messages(
        self,
        conversation_id: str,
        messages: Sequence[Message],
    ) -> Conversation:
        """Append messages atomically and return the updated conversation."""


class InMemoryConversationStore(ConversationStore):
    """Process-local conversation store."""

    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}

    def create(self, conversation: Conversation) -> None:
        """Store a new conversation without replacing an existing one."""

        if conversation.id in self._conversations:
            raise ConversationAlreadyExistsError(f"conversation '{conversation.id}' already exists")
        self._conversations[conversation.id] = conversation

    def get(self, conversation_id: str) -> Conversation | None:
        """Return a stored conversation or None."""

        return self._conversations.get(conversation_id)

    def append_messages(
        self,
        conversation_id: str,
        messages: Sequence[Message],
    ) -> Conversation:
        """Append messages by replacing the immutable conversation value."""

        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"conversation '{conversation_id}' was not found")

        message_tuple = tuple(messages)
        if not message_tuple:
            raise InvalidMessageError("at least one message is required")
        if any(message.conversation_id != conversation_id for message in message_tuple):
            raise InvalidMessageError("message conversation_id must match the conversation")

        updated_at = max(
            conversation.updated_at,
            message_tuple[-1].created_at,
        )
        updated = Conversation(
            id=conversation.id,
            created_at=conversation.created_at,
            updated_at=updated_at,
            messages=conversation.messages + message_tuple,
        )
        self._conversations[conversation_id] = updated
        return updated
