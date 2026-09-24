"""Tests for the LLM service abstraction and the OpenAI-compatible adapter."""

import pytest

from enterprise_ai_agent.llm import LLMService, OpenAICompatibleLLMService


class FakeResponse:
    """Minimal stand-in for an OpenAI Responses API response object."""

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeResponses:
    """Recording stand-in for the client's ``responses`` namespace."""

    def __init__(
        self,
        *,
        output_text: str = "hello",
        error: Exception | None = None,
    ) -> None:
        self.output_text = output_text
        self.error = error
        self.calls: list[dict[str, str]] = []

    def create(self, *, model: str, input: str) -> FakeResponse:
        self.calls.append({"model": model, "input": input})
        if self.error is not None:
            raise self.error
        return FakeResponse(self.output_text)


class FakeClient:
    """Recording stand-in for an injected OpenAI-compatible client."""

    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses


class FakeLLMService(LLMService):
    """Simple in-memory implementation used to exercise the interface."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _make_service(
    *,
    output_text: str = "hello",
    model: str = "test-model",
    error: Exception | None = None,
) -> tuple[OpenAICompatibleLLMService, FakeResponses]:
    responses = FakeResponses(output_text=output_text, error=error)
    service = OpenAICompatibleLLMService(FakeClient(responses), model=model)
    return service, responses


def test_llm_service_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMService()  # type: ignore[abstract]


def test_fake_implementation_satisfies_interface() -> None:
    service: LLMService = FakeLLMService("stub response")

    assert isinstance(service, LLMService)
    assert service.generate("test prompt") == "stub response"


def test_adapter_implements_llm_service() -> None:
    service, _ = _make_service()

    assert isinstance(service, LLMService)


def test_generate_returns_provider_text() -> None:
    service, _ = _make_service(output_text="hello")

    assert service.generate("test prompt") == "hello"


def test_generate_calls_provider_once() -> None:
    service, responses = _make_service()

    service.generate("test prompt")

    assert len(responses.calls) == 1


def test_generate_preserves_prompt_whitespace() -> None:
    service, responses = _make_service()

    service.generate("  hello  ")

    assert responses.calls == [{"model": "test-model", "input": "  hello  "}]


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t", "\r\n"])
def test_generate_rejects_empty_prompt(prompt: str) -> None:
    service, responses = _make_service()

    with pytest.raises(ValueError, match="prompt must not be empty"):
        service.generate(prompt)

    assert responses.calls == []


@pytest.mark.parametrize("model", ["", "   "])
def test_init_rejects_empty_model(model: str) -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        OpenAICompatibleLLMService(FakeClient(FakeResponses()), model=model)


def test_generate_sends_configured_model() -> None:
    service, responses = _make_service(model="gpt-test")

    service.generate("test prompt")

    assert responses.calls[0]["model"] == "gpt-test"


def test_generate_returns_response_text_unchanged() -> None:
    service, _ = _make_service(output_text="  answer with spaces  ")

    assert service.generate("test prompt") == "  answer with spaces  "


def test_generate_propagates_provider_error() -> None:
    error = RuntimeError("provider failed")
    service, _ = _make_service(error=error)

    with pytest.raises(RuntimeError, match="provider failed") as captured:
        service.generate("test prompt")

    assert captured.value is error


def test_generate_does_not_swallow_provider_error() -> None:
    error = ValueError("upstream failure")
    service, _ = _make_service(error=error)

    with pytest.raises(ValueError) as captured:
        service.generate("test prompt")

    assert captured.value is error
    assert str(captured.value) != ""
