"""LLM service abstraction."""

from abc import ABC, abstractmethod


class LLMService(ABC):
    """Replaceable interface for generating text from a prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate one synchronous response for a prompt."""
