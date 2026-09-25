"""Exceptions raised by agent-facing tools."""


class ToolError(Exception):
    """Base exception for tool registration and invocation failures."""


class DuplicateToolNameError(ToolError):
    """Raised when two tools are registered with the same name."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""


class ToolArgumentError(ToolError):
    """Raised when a tool call has invalid arguments."""
