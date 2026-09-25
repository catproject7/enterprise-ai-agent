"""LLM-driven agent with one tool-calling round."""

from enterprise_ai_agent.llm import ToolCallingLLM
from enterprise_ai_agent.tools import ToolRegistry

from .base import Agent
from .exceptions import MaxToolStepsExceededError
from .models import AgentResult


class LLMAgent(Agent[str]):
    """Use an LLM to optionally execute one tool and produce a final answer."""

    def __init__(
        self,
        llm: ToolCallingLLM,
        registry: ToolRegistry,
    ) -> None:
        self._llm = llm
        self._registry = registry

    def run(self, input: str) -> AgentResult[str]:
        """Run one LLM decision and at most one tool execution."""

        tools = self._registry.specs()
        response = self._llm.generate_with_tools(input, tools)
        if response.tool_call is None:
            return AgentResult(output=response.text)

        tool_result = self._registry.invoke(response.tool_call)
        final_response = self._llm.generate_with_tools(
            input,
            tools,
            (tool_result,),
        )
        if final_response.tool_call is not None:
            raise MaxToolStepsExceededError("agent supports at most one tool call")

        return AgentResult(output=final_response.text)
