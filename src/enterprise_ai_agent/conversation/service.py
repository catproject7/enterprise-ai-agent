"""Conversation orchestration with an injected Agent and store."""

from datetime import UTC, datetime
from uuid import uuid4

from enterprise_ai_agent.agent import Agent

from .exceptions import ConversationNotFoundError, InvalidMessageError
from .models import Conversation, Message, MessageRole
from .store import ConversationStore


class ConversationService:
    """Create conversations and append successful Agent exchanges."""

    def __init__(
        self,
        store: ConversationStore,
        agent: Agent[str],
    ) -> None:
        self._store = store
        self._agent = agent

    def create_conversation(self) -> Conversation:
        """Create and persist an empty conversation."""

        now = datetime.now(UTC)
        conversation = Conversation(
            id=uuid4().hex,
            created_at=now,
            updated_at=now,
        )
        self._store.create(conversation)
        return conversation

    def get_conversation(self, conversation_id: str) -> Conversation:
        """Return a conversation or raise when it does not exist."""

        conversation = self._store.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(f"conversation '{conversation_id}' was not found")
        return conversation

    def send_message(
        self,
        conversation_id: str,
        content: str,
    ) -> Conversation:
        """Run the Agent and persist user and assistant messages on success."""

        conversation = self.get_conversation(conversation_id)
        if not content.strip():
            raise InvalidMessageError("message content must not be empty")

        agent_input = self._build_agent_input(conversation, content)
        result = self._agent.run(agent_input)
        now = datetime.now(UTC)
        user_message = Message(
            id=uuid4().hex,
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=content,
            created_at=now,
        )
        assistant_message = Message(
            id=uuid4().hex,
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=result.output,
            created_at=now,
        )
        return self._store.append_messages(
            conversation.id,
            (user_message, assistant_message),
        )

    @staticmethod
    def _build_agent_input(conversation: Conversation, content: str) -> str:
        lines = [
            f"{message.role.value.title()}: {message.content}" for message in conversation.messages
        ]
        lines.append(f"User: {content}")
        return "\n".join(lines)
