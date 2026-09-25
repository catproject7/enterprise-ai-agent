"""Tests for the minimal agent foundation."""

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.agent import Agent, AgentResult, ToolAgent
from enterprise_ai_agent.tools import Tool


class FakeTool(Tool[str, str]):
    """Recording in-memory tool used to exercise the agent boundary."""

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


class IncompleteAgent(Agent[str]):
    """Agent implementation missing the required run method."""


def test_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


def test_agent_requires_run_implementation() -> None:
    with pytest.raises(TypeError):
        IncompleteAgent()  # type: ignore[abstract]


def test_tool_agent_uses_injected_tool() -> None:
    tool = FakeTool()
    agent = ToolAgent(tool)

    result = agent.run("hello")

    assert result.output == "tool output"
    assert tool.calls == ["hello"]


def test_tool_agent_passes_input_unchanged() -> None:
    tool = FakeTool()
    agent = ToolAgent(tool)
    text = "  hello  "

    agent.run(text)

    assert tool.calls == [text]


@pytest.mark.parametrize("text", ["", " ", "\n\t", "\r\n"])
def test_tool_agent_does_not_validate_input(text: str) -> None:
    tool = FakeTool()
    agent = ToolAgent(tool)

    result = agent.run(text)

    assert result.output == tool.output
    assert tool.calls == [text]


def test_tool_output_is_wrapped_in_agent_result() -> None:
    output = "  exact output  "
    agent = ToolAgent(FakeTool(output))

    result = agent.run("hello")

    assert isinstance(result, AgentResult)
    assert result.output is output


def test_tool_agent_propagates_tool_error_unchanged() -> None:
    error = RuntimeError("tool failed")
    agent = ToolAgent(FakeTool(error=error))

    with pytest.raises(RuntimeError, match="tool failed") as captured:
        agent.run("hello")

    assert captured.value is error


def test_tool_agent_is_deterministic() -> None:
    tool = FakeTool("stable output")
    agent = ToolAgent(tool)

    first = agent.run("hello")
    second = agent.run("hello")

    assert first == second
    assert first.output == second.output == "stable output"
    assert tool.calls == ["hello", "hello"]


def test_agent_result_preserves_output() -> None:
    result = AgentResult(output="answer")

    assert result.output == "answer"


def test_agent_result_validates_output_type() -> None:
    result = AgentResult[int](output="1")

    assert result.output == 1

    with pytest.raises(ValidationError):
        AgentResult[int](output="not-an-integer")


def test_agent_result_is_frozen() -> None:
    result = AgentResult(output="answer")

    with pytest.raises(ValidationError):
        result.output = "changed"
