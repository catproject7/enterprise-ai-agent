"""Tests for request context, structured logs, and trace wrappers."""

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from enterprise_ai_agent.agent import Agent, AgentResult
from enterprise_ai_agent.api import create_app
from enterprise_ai_agent.llm import (
    ToolCall,
    ToolCallingLLM,
    ToolCallingResponse,
    ToolResult,
    ToolSpec,
)
from enterprise_ai_agent.observability import (
    OBSERVABILITY_LOGGER_NAME,
    RequestObservabilityMiddleware,
    StructuredFormatter,
    TracedAgent,
    TracedToolCallingLLM,
    TracedToolRegistry,
    get_request_id,
    request_context,
    resolve_request_id,
)
from enterprise_ai_agent.tools import Tool, ToolRegistry


class FakeAgent(Agent[str]):
    """Fake Agent used by instrumentation tests."""

    def __init__(
        self,
        output: str = "answer",
        *,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []

    def run(self, input: str) -> AgentResult[str]:
        self.calls.append(input)
        if self.error is not None:
            raise self.error
        return AgentResult(output=self.output)


class FakeToolCallingLLM(ToolCallingLLM):
    """Fake tool-calling LLM used by instrumentation tests."""

    def __init__(
        self,
        response: ToolCallingResponse,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, tuple[ToolSpec, ...], tuple[ToolResult, ...]]] = []

    def generate_with_tools(
        self,
        prompt: str,
        tools: tuple[ToolSpec, ...],
        tool_results: tuple[ToolResult, ...] = (),
    ) -> ToolCallingResponse:
        self.calls.append((prompt, tools, tool_results))
        if self.error is not None:
            raise self.error
        return self.response


class FakeTool(Tool[str, str]):
    """Fake Tool used by registry instrumentation tests."""

    name = "fake"
    description = "Return the supplied input."

    def run(self, input: str) -> str:
        return input


def _events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if hasattr(record, "event")]


def test_request_id_generation_returns_safe_random_value() -> None:
    request_id = resolve_request_id()

    assert len(request_id) == 32
    assert all(character in "0123456789abcdef" for character in request_id)


def test_request_id_uses_safe_client_value() -> None:
    assert resolve_request_id("client-request-123") == "client-request-123"


def test_request_id_rejects_unsafe_client_value() -> None:
    request_id = resolve_request_id("unsafe request id")

    assert request_id != "unsafe request id"
    assert len(request_id) == 32


def test_request_context_supports_nesting_and_reset() -> None:
    assert get_request_id() is None

    with request_context("outer") as outer:
        assert outer == "outer"
        assert get_request_id() == "outer"
        with request_context("inner") as inner:
            assert inner == "inner"
            assert get_request_id() == "inner"
        assert get_request_id() == "outer"

    assert get_request_id() is None


def test_structured_formatter_emits_json_fields() -> None:
    record = logging.LogRecord(
        name=OBSERVABILITY_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request.completed",
        args=(),
        exc_info=None,
    )
    record.event = "request.completed"
    record.request_id = "request-1"
    record.component = "api"
    record.duration_ms = 12.5
    record.status = 200
    record.method = "GET"
    record.path = "/health"

    payload = json.loads(StructuredFormatter().format(record))

    assert payload["event"] == "request.completed"
    assert payload["request_id"] == "request-1"
    assert payload["component"] == "api"
    assert payload["duration_ms"] == 12.5
    assert payload["status"] == 200
    assert payload["method"] == "GET"
    assert payload["path"] == "/health"
    assert "tool_name" not in payload


def test_middleware_generates_request_id_and_logs_health_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32
    events = _events(caplog)
    started = next(record for record in events if record.event == "request.started")
    completed = next(record for record in events if record.event == "request.completed")
    assert started.request_id == request_id
    assert started.component == "api"
    assert started.method == "GET"
    assert started.path == "/health"
    assert completed.status == 200
    assert completed.duration_ms >= 0


def test_middleware_reuses_client_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    client = TestClient(create_app())

    response = client.get("/health", headers={"X-Request-ID": "client-request-id"})

    assert response.headers["X-Request-ID"] == "client-request-id"
    assert all(record.request_id == "client-request-id" for record in _events(caplog))


def test_middleware_logs_unhandled_request_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger=OBSERVABILITY_LOGGER_NAME)
    app = FastAPI()
    app.add_middleware(RequestObservabilityMiddleware)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("boom")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    failed = next(record for record in _events(caplog) if record.event == "request.failed")
    assert failed.method == "GET"
    assert failed.path == "/boom"
    assert failed.status == 500
    assert failed.error_type == "RuntimeError"
    assert failed.duration_ms >= 0


def test_traced_agent_preserves_result_and_logs_lifecycle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    agent = FakeAgent()
    traced = TracedAgent(agent)

    result = traced.run("question")

    assert result == AgentResult(output="answer")
    assert agent.calls == ["question"]
    assert [record.event for record in _events(caplog)] == [
        "agent.started",
        "agent.completed",
    ]
    assert _events(caplog)[1].duration_ms >= 0


def test_traced_agent_propagates_error_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    error = RuntimeError("agent failed")
    traced = TracedAgent(FakeAgent(error=error))

    with pytest.raises(RuntimeError, match="agent failed") as captured:
        traced.run("question")

    assert captured.value is error
    assert _events(caplog)[-1].event == "agent.failed"
    assert _events(caplog)[-1].error_type == "RuntimeError"


def test_traced_llm_preserves_response_and_logs_tool_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    tool_call = ToolCall(
        call_id="call-1",
        name="rag",
        arguments='{"query": "question"}',
    )
    response = ToolCallingResponse(tool_call=tool_call)
    traced = TracedToolCallingLLM(FakeToolCallingLLM(response))

    result = traced.generate_with_tools("question", ())

    assert result is response
    completed = _events(caplog)[-1]
    assert completed.event == "llm.completed"
    assert completed.tool_name == "rag"
    assert completed.call_id == "call-1"
    assert completed.duration_ms >= 0


def test_traced_llm_propagates_error_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    error = RuntimeError("llm failed")
    response = ToolCallingResponse(text="answer")
    traced = TracedToolCallingLLM(FakeToolCallingLLM(response, error=error))

    with pytest.raises(RuntimeError, match="llm failed") as captured:
        traced.generate_with_tools("question", ())

    assert captured.value is error
    assert _events(caplog)[-1].event == "llm.failed"


def test_traced_tool_registry_preserves_result_and_logs_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    registry = ToolRegistry()
    registry.register(FakeTool(), argument_name="query")
    traced = TracedToolRegistry(registry)
    call = ToolCall(
        call_id="call-1",
        name="fake",
        arguments='{"query": "hello"}',
    )

    result = traced.invoke(call)

    assert result.output == "hello"
    assert [record.event for record in _events(caplog)] == [
        "tool.started",
        "tool.completed",
    ]
    assert _events(caplog)[-1].tool_name == "fake"
    assert _events(caplog)[-1].call_id == "call-1"
    assert _events(caplog)[-1].duration_ms >= 0


def test_traced_tool_registry_propagates_error_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    traced = TracedToolRegistry(ToolRegistry())
    call = ToolCall(
        call_id="call-1",
        name="missing",
        arguments='{"query": "hello"}',
    )

    with pytest.raises(Exception) as captured:
        traced.invoke(call)

    assert type(captured.value).__name__ == "ToolNotFoundError"
    assert _events(caplog)[-1].event == "tool.failed"
    assert _events(caplog)[-1].error_type == "ToolNotFoundError"


def test_nested_wrappers_share_request_id(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    traced_agent = TracedAgent(FakeAgent())

    with request_context("request-1"):
        traced_agent.run("question")

    assert all(record.request_id == "request-1" for record in _events(caplog))


def test_logs_do_not_include_prompt_or_tool_output(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=OBSERVABILITY_LOGGER_NAME)
    prompt = "SECRET_PROMPT"
    output = "SECRET_OUTPUT"
    traced = TracedAgent(FakeAgent(output=output))

    traced.run(prompt)

    payload = "\n".join(str(vars(record)) for record in _events(caplog))
    assert prompt not in payload
    assert output not in payload
    assert "Traceback" not in payload
