"""Tests for provider-neutral and OpenAI-compatible tool-calling models."""

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.llm import (
    OpenAICompatibleToolCallingLLM,
    ToolCall,
    ToolCallingLLM,
    ToolCallingResponse,
    ToolResult,
    ToolSpec,
    UnsupportedToolCallsError,
)


class FakeFunctionCall:
    """Minimal stand-in for a Responses API function call item."""

    type = "function_call"

    def __init__(
        self,
        *,
        call_id: str = "call-1",
        name: str = "fake",
        arguments: str = '{"query": "hello"}',
    ) -> None:
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class FakeResponse:
    """Minimal Responses API response."""

    def __init__(
        self,
        *,
        output: list[object] | None = None,
        output_text: str = "",
    ) -> None:
        self.output = output if output is not None else []
        self.output_text = output_text


class FakeResponses:
    """Recording stand-in for the client's responses namespace."""

    def __init__(
        self,
        response: FakeResponse | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response if response is not None else FakeResponse(output_text="hello")
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    """Recording stand-in for an injected OpenAI-compatible client."""

    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


def _make_llm(
    *,
    response: FakeResponse | None = None,
    error: Exception | None = None,
) -> tuple[OpenAICompatibleToolCallingLLM, FakeResponses]:
    responses = FakeResponses(response, error=error)
    return OpenAICompatibleToolCallingLLM(FakeClient(responses), model="test-model"), responses


def _make_spec(*, name: str = "fake") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Test tool.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def test_tool_calling_llm_is_abstract() -> None:
    with pytest.raises(TypeError):
        ToolCallingLLM()  # type: ignore[abstract]


def test_adapter_implements_tool_calling_llm() -> None:
    llm, _ = _make_llm()

    assert isinstance(llm, ToolCallingLLM)


def test_response_returns_text_when_no_tool_is_requested() -> None:
    llm, _ = _make_llm(response=FakeResponse(output_text="answer"))

    response = llm.generate_with_tools("question", ())

    assert response == ToolCallingResponse(text="answer")


def test_response_returns_one_tool_call() -> None:
    function_call = FakeFunctionCall()
    llm, _ = _make_llm(response=FakeResponse(output=[function_call]))

    response = llm.generate_with_tools("question", (_make_spec(),))

    assert response.tool_call == ToolCall(
        call_id="call-1",
        name="fake",
        arguments='{"query": "hello"}',
    )


def test_multiple_tool_calls_are_rejected() -> None:
    llm, _ = _make_llm(
        response=FakeResponse(
            output=[FakeFunctionCall(call_id="call-1"), FakeFunctionCall(call_id="call-2")]
        )
    )

    with pytest.raises(UnsupportedToolCallsError, match="parallel"):
        llm.generate_with_tools("question", (_make_spec(),))


def test_tool_specs_are_sent_to_provider() -> None:
    llm, responses = _make_llm()
    spec = _make_spec()

    llm.generate_with_tools("question", (spec,))

    call = responses.calls[0]
    assert call["model"] == "test-model"
    assert call["input"] == "question"
    assert call["tool_choice"] == "auto"
    assert call["parallel_tool_calls"] is False
    assert call["tools"] == [
        {
            "type": "function",
            "name": "fake",
            "description": "Test tool.",
            "parameters": spec.input_schema,
            "strict": True,
        }
    ]


def test_tool_results_are_sent_as_function_call_output() -> None:
    llm, responses = _make_llm(response=FakeResponse(output_text="final"))
    call = ToolCall(
        call_id="call-1",
        name="fake",
        arguments='{"query": "hello"}',
    )
    result = ToolResult(call=call, output="tool output")

    llm.generate_with_tools("question", (_make_spec(),), (result,))

    assert responses.calls[0]["input"] == [
        {"role": "user", "content": "question"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "fake",
            "arguments": '{"query": "hello"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "tool output",
        },
    ]


def test_provider_error_is_propagated_unchanged() -> None:
    error = RuntimeError("provider failed")
    llm, _ = _make_llm(error=error)

    with pytest.raises(RuntimeError, match="provider failed") as captured:
        llm.generate_with_tools("question", ())

    assert captured.value is error


def test_tool_calling_response_requires_exactly_one_result() -> None:
    with pytest.raises(ValidationError):
        ToolCallingResponse()
    with pytest.raises(ValidationError):
        ToolCallingResponse(
            text="answer",
            tool_call=ToolCall(call_id="call-1", name="fake", arguments='{"query": "hello"}'),
        )
