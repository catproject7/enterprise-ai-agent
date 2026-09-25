"""Tool abstraction."""

from abc import ABC, abstractmethod


class Tool[InputT, OutputT](ABC):
    """Replaceable interface for a synchronous agent-facing capability."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable tool name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Describe when the tool should be used."""

    @abstractmethod
    def run(self, input: InputT) -> OutputT:
        """Execute the tool for one typed input."""
