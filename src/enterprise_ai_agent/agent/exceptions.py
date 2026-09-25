"""Exceptions raised during agent execution."""


class AgentError(Exception):
    """Base exception for agent failures."""


class MaxToolStepsExceededError(AgentError):
    """Raised when the agent requests more tool calls than it supports."""
