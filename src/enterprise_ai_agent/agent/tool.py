"""Deterministic single-tool agent."""

from enterprise_ai_agent.tools import Tool

from .base import Agent
from .models import AgentResult


class ToolAgent[OutputT](Agent[OutputT]):
    """Execute one injected tool for each input."""

    def __init__(self, tool: Tool[str, OutputT]) -> None:
        self._tool = tool

    def run(self, input: str) -> AgentResult[OutputT]:
        """Run the injected tool and wrap its output."""

        return AgentResult(output=self._tool.run(input))
