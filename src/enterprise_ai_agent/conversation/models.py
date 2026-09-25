"""Conversation and message domain models."""

from datetime import datetime, timedelta
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class MessageRole(StrEnum):
    """Supported conversation message roles."""

    USER = "user"
    ASSISTANT = "assistant"


def _validate_uuid_hex(value: str, field_name: str) -> str:
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a UUID hex string") from error

    if value != parsed.hex:
        raise ValueError(f"{field_name} must be a UUID hex string")
    return value


def _validate_utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC")
    return value


class Message(BaseModel):
    """One immutable user or assistant message."""

    model_config = ConfigDict(frozen=True)

    id: str
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Require a UUID hex message identifier."""

        return _validate_uuid_hex(value, "id")

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value: str) -> str:
        """Require a UUID hex conversation identifier."""

        return _validate_uuid_hex(value, "conversation_id")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Reject empty message content."""

        if not value.strip():
            raise ValueError("message content must not be empty")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        """Require a UTC-aware timestamp."""

        return _validate_utc_datetime(value, "created_at")


class Conversation(BaseModel):
    """One immutable conversation and its ordered messages."""

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[Message, ...] = ()

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        """Require a UUID hex conversation identifier."""

        return _validate_uuid_hex(value, "id")

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamps(cls, value: datetime) -> datetime:
        """Require UTC-aware timestamps."""

        return _validate_utc_datetime(value, "timestamp")

    @model_validator(mode="after")
    def validate_conversation(self) -> "Conversation":
        """Enforce timestamp and message ownership invariants."""

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        if any(message.conversation_id != self.id for message in self.messages):
            raise ValueError("message conversation_id must match conversation id")
        return self
