"""Tests for conversation models, persistence, and orchestration."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.agent import Agent, AgentResult
from enterprise_ai_agent.conversation import (
    Conversation,
    ConversationAlreadyExistsError,
    ConversationNotFoundError,
    ConversationService,
    InMemoryConversationStore,
    InvalidMessageError,
    Message,
    MessageRole,
)


class FakeAgent(Agent[str]):
    """Recording Agent used by conversation tests."""

    def __init__(
        self,
        output: str = "answer",
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []

    def run(self, input: str) -> AgentResult[str]:
        self.calls.append(input)
        if self.error is not None:
            raise self.error
        return AgentResult(output=self.output)


def _now() -> datetime:
    return datetime.now(UTC)


def _make_conversation() -> Conversation:
    now = _now()
    return Conversation(
        id=uuid4().hex,
        created_at=now,
        updated_at=now,
    )


def _make_message(
    conversation_id: str,
    *,
    role: MessageRole = MessageRole.USER,
    content: str = "question",
    created_at: datetime | None = None,
) -> Message:
    return Message(
        id=uuid4().hex,
        conversation_id=conversation_id,
        role=role,
        content=content,
        created_at=created_at if created_at is not None else _now(),
    )


def test_message_is_frozen() -> None:
    message = _make_message(uuid4().hex)

    with pytest.raises(ValidationError):
        message.content = "changed"


def test_conversation_is_frozen() -> None:
    conversation = _make_conversation()

    with pytest.raises(ValidationError):
        conversation.messages = ()


@pytest.mark.parametrize("content", ["", " ", "\n\t", "\r\n"])
def test_message_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValidationError, match="content"):
        _make_message(uuid4().hex, content=content)


def test_message_rejects_invalid_role() -> None:
    with pytest.raises(ValidationError):
        Message(
            id=uuid4().hex,
            conversation_id=uuid4().hex,
            role="tool",
            content="question",
            created_at=_now(),
        )


def test_message_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        _make_message(uuid4().hex, created_at=datetime.now())


def test_store_create_and_get() -> None:
    store = InMemoryConversationStore()
    conversation = _make_conversation()

    store.create(conversation)

    assert store.get(conversation.id) is conversation


def test_store_get_missing_returns_none() -> None:
    store = InMemoryConversationStore()

    assert store.get(uuid4().hex) is None


def test_store_rejects_duplicate_create() -> None:
    store = InMemoryConversationStore()
    conversation = _make_conversation()
    store.create(conversation)

    with pytest.raises(ConversationAlreadyExistsError, match=conversation.id):
        store.create(conversation)


def test_store_append_preserves_order_and_replaces_immutably() -> None:
    store = InMemoryConversationStore()
    conversation = _make_conversation()
    store.create(conversation)
    first = _make_message(conversation.id, content="first")
    second = _make_message(
        conversation.id,
        role=MessageRole.ASSISTANT,
        content="second",
        created_at=first.created_at + timedelta(seconds=1),
    )

    updated = store.append_messages(conversation.id, (first, second))

    assert updated is not conversation
    assert conversation.messages == ()
    assert updated.messages == (first, second)
    assert updated.updated_at == second.created_at


def test_store_rejects_append_for_missing_conversation() -> None:
    store = InMemoryConversationStore()
    conversation_id = uuid4().hex

    with pytest.raises(ConversationNotFoundError, match=conversation_id):
        store.append_messages(conversation_id, (_make_message(conversation_id),))


def test_store_rejects_empty_append() -> None:
    store = InMemoryConversationStore()
    conversation = _make_conversation()
    store.create(conversation)

    with pytest.raises(InvalidMessageError, match="at least one"):
        store.append_messages(conversation.id, ())


def test_store_keeps_conversations_isolated() -> None:
    store = InMemoryConversationStore()
    first = _make_conversation()
    second = _make_conversation()
    store.create(first)
    store.create(second)
    first_message = _make_message(first.id, content="first")
    second_message = _make_message(second.id, content="second")

    store.append_messages(first.id, (first_message,))
    store.append_messages(second.id, (second_message,))

    assert store.get(first.id).messages == (first_message,)
    assert store.get(second.id).messages == (second_message,)


def test_service_creates_and_gets_conversation() -> None:
    store = InMemoryConversationStore()
    service = ConversationService(store, FakeAgent())

    conversation = service.create_conversation()

    assert service.get_conversation(conversation.id) == conversation
    assert conversation.messages == ()


def test_service_get_missing_conversation() -> None:
    service = ConversationService(InMemoryConversationStore(), FakeAgent())
    conversation_id = uuid4().hex

    with pytest.raises(ConversationNotFoundError, match=conversation_id):
        service.get_conversation(conversation_id)


def test_service_sends_first_message_to_agent() -> None:
    agent = FakeAgent("answer")
    service = ConversationService(InMemoryConversationStore(), agent)
    conversation = service.create_conversation()

    updated = service.send_message(conversation.id, "question")

    assert agent.calls == ["User: question"]
    assert [message.role for message in updated.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [message.content for message in updated.messages] == ["question", "answer"]


def test_service_builds_history_for_second_message() -> None:
    agent = FakeAgent("second answer")
    service = ConversationService(InMemoryConversationStore(), agent)
    conversation = service.create_conversation()
    service.send_message(conversation.id, "first question")

    service.send_message(conversation.id, "second question")

    assert agent.calls[1] == (
        "User: first question\nAssistant: second answer\nUser: second question"
    )


def test_service_does_not_write_when_agent_fails() -> None:
    error = RuntimeError("agent failed")
    agent = FakeAgent(error=error)
    store = InMemoryConversationStore()
    service = ConversationService(store, agent)
    conversation = service.create_conversation()

    with pytest.raises(RuntimeError, match="agent failed") as captured:
        service.send_message(conversation.id, "question")

    assert captured.value is error
    assert store.get(conversation.id) == conversation


@pytest.mark.parametrize("content", ["", " ", "\n\t"])
def test_service_rejects_empty_content(content: str) -> None:
    service = ConversationService(InMemoryConversationStore(), FakeAgent())
    conversation = service.create_conversation()

    with pytest.raises(InvalidMessageError, match="content"):
        service.send_message(conversation.id, content)
