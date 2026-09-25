"""Exceptions raised by conversation persistence and orchestration."""


class ConversationError(Exception):
    """Base exception for conversation failures."""


class ConversationNotFoundError(ConversationError):
    """Raised when a requested conversation does not exist."""


class ConversationAlreadyExistsError(ConversationError):
    """Raised when a conversation ID is already stored."""


class InvalidMessageError(ConversationError):
    """Raised when a message cannot be persisted or sent."""
