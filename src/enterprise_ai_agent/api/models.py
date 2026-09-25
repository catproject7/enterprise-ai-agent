"""HTTP request and response models."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from enterprise_ai_agent.conversation import Conversation, Message
from enterprise_ai_agent.rag import Citation


class AgentRunRequest(BaseModel):
    """One text input for the configured Agent."""

    model_config = ConfigDict(frozen=True)

    input: str

    @field_validator("input")
    @classmethod
    def validate_input(cls, input: str) -> str:
        """Reject empty Agent input."""

        if not input.strip():
            raise ValueError("input must not be empty")
        return input


class AgentRunResponse(BaseModel):
    """One final Agent output."""

    model_config = ConfigDict(frozen=True)

    output: str


class HealthResponse(BaseModel):
    """Application health status."""

    model_config = ConfigDict(frozen=True)

    status: str


class ErrorResponse(BaseModel):
    """Stable API error response."""

    model_config = ConfigDict(frozen=True)

    detail: str


class MessageResponse(BaseModel):
    """One conversation message."""

    model_config = ConfigDict(frozen=True)

    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime

    @classmethod
    def from_domain(cls, message: Message) -> "MessageResponse":
        """Map a domain message to its API representation."""

        return cls(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role.value,
            content=message.content,
            created_at=message.created_at,
        )


class ConversationResponse(BaseModel):
    """One conversation and its ordered messages."""

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[MessageResponse, ...] = ()

    @classmethod
    def from_domain(cls, conversation: Conversation) -> "ConversationResponse":
        """Map a domain conversation to its API representation."""

        return cls(
            id=conversation.id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            messages=tuple(
                MessageResponse.from_domain(message) for message in conversation.messages
            ),
        )


class CreateConversationResponse(ConversationResponse):
    """Response returned when a conversation is created."""


class SendMessageRequest(BaseModel):
    """One user message sent to a conversation."""

    model_config = ConfigDict(frozen=True)

    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, content: str) -> str:
        """Reject empty message content."""

        if not content.strip():
            raise ValueError("content must not be empty")
        return content


class SendMessageResponse(BaseModel):
    """The successful exchange appended to a conversation."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    user_message: MessageResponse
    assistant_message: MessageResponse


class RagQueryRequest(BaseModel):
    """One question sent directly to the RAG pipeline."""

    model_config = ConfigDict(frozen=True)

    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        """Reject empty questions."""

        if not question.strip():
            raise ValueError("question must not be empty")
        return question


class RagQueryResponse(BaseModel):
    """A direct RAG answer with its supporting citations."""

    model_config = ConfigDict(frozen=True)

    answer: str
    citations: tuple[Citation, ...] = ()
