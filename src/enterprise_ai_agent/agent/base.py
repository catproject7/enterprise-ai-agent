"""Agent abstraction."""

from abc import ABC, abstractmethod

from .models import AgentResult


class Agent[OutputT](ABC):
    """Replaceable interface for agent execution."""

    @abstractmethod
    def run(self, input: str) -> AgentResult[OutputT]:
        """Run the agent for one text input."""
