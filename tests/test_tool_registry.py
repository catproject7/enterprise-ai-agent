"""Tests for tool registration and string-input invocation."""

import pytest
from pydantic import BaseModel, ConfigDict

from enterprise_ai_agent.llm import ToolCall
from enterprise_ai_agent.tools import (
    DuplicateToolNameError,
    Tool,
    ToolArgumentError,
    ToolNotFoundError,
    ToolRegistry,
)


class FakeTool(Tool[str, str]):
    """Recording string-input tool."""

    def __init__(
        self,
        name: str = "fake",
        output: str = "tool output",
        *,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._description = "Return the configured output."
        self.output = output
        self.error = error
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def run(self, input: str) -> str:
        self.calls.append(input)
        if self.error is not None:
            raise self.error
        return self.output


class FakeOutput(BaseModel):
    """Deterministic Pydantic output used for serialization tests."""

    model_config = ConfigDict(frozen=True)

    value: str
    count: int


class PydanticOutputTool(Tool[str, FakeOutput]):
    """Tool returning a Pydantic model."""

    name = "structured"
    description = "Return structured output."

    def run(self, input: str) -> FakeOutput:
        return FakeOutput(value=input, count=1)


def _make_call(
    *,
    name: str = "fake",
    arguments: str = '{"query": "hello"}',
) -> ToolCall:
    return ToolCall(
        call_id="call-1",
        name=name,
        arguments=arguments,
    )


def test_register_and_get_tool() -> None:
    registry = ToolRegistry()
    tool = FakeTool()

    registry.register(tool, argument_name="query")

    assert registry.get("fake") is tool


def test_specs_preserve_registration_order() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool("first"), argument_name="query")
    registry.register(FakeTool("second"), argument_name="input")

    specs = registry.specs()

    assert [spec.name for spec in specs] == ["first", "second"]


def test_specs_generate_string_query_schema() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(), argument_name="query")

    spec = registry.specs()[0]

    assert spec.name == "fake"
    assert spec.description == "Return the configured output."
    assert spec.input_schema == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }


def test_register_rejects_duplicate_tool_name() -> None:
    registry = ToolRegistry()
    registry.register(FakeTool(), argument_name="query")

    with pytest.raises(DuplicateToolNameError, match="fake"):
        registry.register(FakeTool(), argument_name="input")


def test_get_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError, match="missing"):
        registry.get("missing")


def test_invoke_passes_query_to_tool_and_wraps_output() -> None:
    registry = ToolRegistry()
    tool = FakeTool(output="result")
    registry.register(tool, argument_name="query")
    call = _make_call()

    result = registry.invoke(call)

    assert result.call == call
    assert result.output == "result"
    assert tool.calls == ["hello"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ('{"input": "hello"}', "query"),
        ('{"query": "hello", "extra": "value"}', "query"),
        ('{"query": 1}', "query"),
        ("not-json", "valid JSON"),
        ("[]", "JSON object"),
    ],
)
def test_invoke_rejects_invalid_arguments(arguments: str, message: str) -> None:
    registry = ToolRegistry()
    tool = FakeTool()
    registry.register(tool, argument_name="query")

    with pytest.raises(ToolArgumentError, match=message):
        registry.invoke(_make_call(arguments=arguments))

    assert tool.calls == []


def test_invoke_propagates_tool_error_unchanged() -> None:
    error = RuntimeError("tool failed")
    registry = ToolRegistry()
    registry.register(FakeTool(error=error), argument_name="query")

    with pytest.raises(RuntimeError, match="tool failed") as captured:
        registry.invoke(_make_call())

    assert captured.value is error


def test_invoke_serializes_pydantic_output_deterministically() -> None:
    registry = ToolRegistry()
    registry.register(PydanticOutputTool(), argument_name="query")

    result = registry.invoke(_make_call(name="structured"))

    assert result.output == '{"value":"hello","count":1}'


def test_invoke_serializes_json_output_deterministically() -> None:
    registry = ToolRegistry()

    class JsonOutputTool(Tool[str, dict[str, int]]):
        name = "json"
        description = "Return JSON-compatible output."

        def run(self, input: str) -> dict[str, int]:
            return {"second": 2, "first": 1}

    registry.register(JsonOutputTool(), argument_name="input")

    result = registry.invoke(_make_call(name="json", arguments='{"input": "hello"}'))

    assert result.output == '{"first":1,"second":2}'
