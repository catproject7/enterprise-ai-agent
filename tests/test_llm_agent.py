"""Tests for the LLM agent with one supported tool-call round."""

import pytest

from enterprise_ai_agent.agent import (
    Agent,
    AgentResult,
    LLMAgent,
    MaxToolStepsExceededError,
)
from enterprise_ai_agent.llm import (
    ToolCall,
    ToolCallingLLM,
    ToolCallingResponse,
    ToolResult,
    ToolSpec,
)
from enterprise_ai_agent.tools import Tool, ToolNotFoundError, ToolRegistry


class FakeTool(Tool[str, str]):
    """Recording tool used by the LLM agent tests."""

    name = "fake"
    description = "Return the configured output."

    def __init__(
        self,
        output: str = "tool output",
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []

    def run(self, input: str) -> str:
        self.calls.append(input)
        if self.error is not None:
            raise self.error
        return self.output


class FakeToolCallingLLM(ToolCallingLLM):
    """Scripted tool-calling LLM."""

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, tuple[ToolSpec, ...], tuple[ToolResult, ...]]] = []

    def generate_with_tools(
        self,
        prompt: str,
        tools: tuple[ToolSpec, ...],
        tool_results: tuple[ToolResult, ...] = (),
    ) -> ToolCallingResponse:
        self.calls.append((prompt, tools, tool_results))
        return self.responses.pop(0)


def _make_call(*, name: str = "fake") -> ToolCall:
    return ToolCall(
        call_id="call-1",
        name=name,
        arguments='{"query": "hello"}',
    )


def test_llm_agent_returns_direct_text_without_tools() -> None:
    llm = FakeToolCallingLLM([ToolCallingResponse(text="answer")])
    agent = LLMAgent(llm, ToolRegistry())

    result = agent.run("question")

    assert result == AgentResult(output="answer")
    assert llm.calls == [("question", (), ())]


def test_llm_agent_executes_one_tool_and_returns_final_text() -> None:
    tool = FakeTool()
    registry = ToolRegistry()
    registry.register(tool, argument_name="query")
    call = _make_call()
    llm = FakeToolCallingLLM(
        [
            ToolCallingResponse(tool_call=call),
            ToolCallingResponse(text="final answer"),
        ]
    )
    agent = LLMAgent(llm, registry)

    result = agent.run("question")

    assert result == AgentResult(output="final answer")
    assert tool.calls == ["hello"]
    assert len(llm.calls) == 2
    assert llm.calls[1][2] == (ToolResult(call=call, output="tool output"),)


def test_llm_agent_rejects_unknown_tool() -> None:
    llm = FakeToolCallingLLM([ToolCallingResponse(tool_call=_make_call(name="missing"))])
    agent = LLMAgent(llm, ToolRegistry())

    with pytest.raises(ToolNotFoundError, match="missing"):
        agent.run("question")

    assert len(llm.calls) == 1


def test_llm_agent_propagates_tool_error_unchanged() -> None:
    error = RuntimeError("tool failed")
    registry = ToolRegistry()
    registry.register(FakeTool(error=error), argument_name="query")
    llm = FakeToolCallingLLM([ToolCallingResponse(tool_call=_make_call())])
    agent = LLMAgent(llm, registry)

    with pytest.raises(RuntimeError, match="tool failed") as captured:
        agent.run("question")

    assert captured.value is error


def test_llm_agent_rejects_second_tool_call() -> None:
    registry = ToolRegistry()
    tool = FakeTool()
    registry.register(tool, argument_name="query")
    llm = FakeToolCallingLLM(
        [
            ToolCallingResponse(tool_call=_make_call()),
            ToolCallingResponse(tool_call=_make_call()),
        ]
    )
    agent = LLMAgent(llm, registry)

    with pytest.raises(MaxToolStepsExceededError, match="one tool call"):
        agent.run("question")

    assert tool.calls == ["hello"]
    assert len(llm.calls) == 2


def test_llm_agent_implements_agent_interface() -> None:
    agent: Agent[str] = LLMAgent(
        FakeToolCallingLLM([ToolCallingResponse(text="answer")]),
        ToolRegistry(),
    )

    assert isinstance(agent, Agent)
    assert agent.run("question") == AgentResult(output="answer")
