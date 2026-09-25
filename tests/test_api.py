"""Tests for the FastAPI application foundation."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from enterprise_ai_agent.agent import (
    Agent,
    AgentResult,
    MaxToolStepsExceededError,
)
from enterprise_ai_agent.api import create_app
from enterprise_ai_agent.api.app import app as default_app
from enterprise_ai_agent.llm import UnsupportedToolCallsError
from enterprise_ai_agent.tools import ToolArgumentError, ToolNotFoundError


class FakeAgent(Agent[str]):
    """Recording Agent used to exercise the HTTP boundary."""

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


def test_create_app_returns_fastapi_application() -> None:
    app = create_app(FakeAgent())

    assert isinstance(app, FastAPI)


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_default_app_health_endpoint_returns_ok() -> None:
    client = TestClient(default_app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_agent_endpoint_returns_503_without_configured_agent() -> None:
    client = TestClient(create_app())

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Agent is not configured"}


def test_agent_endpoint_returns_fake_agent_output() -> None:
    agent = FakeAgent("answer")
    client = TestClient(create_app(agent))

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 200
    assert response.json() == {"output": "answer"}


def test_agent_endpoint_passes_input_unchanged() -> None:
    agent = FakeAgent()
    client = TestClient(create_app(agent))
    input_text = "  question with spaces  "

    client.post("/agent/run", json={"input": input_text})

    assert agent.calls == [input_text]


def test_agent_endpoint_maps_agent_result_output() -> None:
    agent = FakeAgent("  exact output  ")
    client = TestClient(create_app(agent))

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 200
    assert response.json() == {"output": "  exact output  "}


def test_agent_endpoint_rejects_empty_input() -> None:
    client = TestClient(create_app(FakeAgent()))

    response = client.post("/agent/run", json={"input": "   "})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}


def test_agent_endpoint_rejects_missing_input() -> None:
    client = TestClient(create_app(FakeAgent()))

    response = client.post("/agent/run", json={})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}


def test_agent_endpoint_maps_max_tool_steps_error() -> None:
    client = TestClient(create_app(FakeAgent(error=MaxToolStepsExceededError("tool limit"))))

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent tool call limit exceeded"}


def test_agent_endpoint_maps_tool_not_found_error() -> None:
    client = TestClient(create_app(FakeAgent(error=ToolNotFoundError("missing"))))

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent requested an unavailable tool"}


def test_agent_endpoint_maps_tool_argument_error() -> None:
    client = TestClient(create_app(FakeAgent(error=ToolArgumentError("invalid"))))

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent produced invalid tool arguments"}


def test_agent_endpoint_maps_unsupported_tool_calls_error() -> None:
    client = TestClient(create_app(FakeAgent(error=UnsupportedToolCallsError("parallel"))))

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent produced unsupported tool calls"}


def test_agent_endpoint_does_not_leak_unexpected_exception_details() -> None:
    client = TestClient(
        create_app(FakeAgent(error=RuntimeError("secret path"))),
        raise_server_exceptions=False,
    )

    response = client.post("/agent/run", json={"input": "question"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "secret path" not in response.text
    assert "Traceback" not in response.text


def test_openapi_describes_request_and_response_models() -> None:
    client = TestClient(create_app(FakeAgent()))

    schema = client.get("/openapi.json").json()
    agent_run = schema["paths"]["/agent/run"]["post"]

    assert agent_run["requestBody"]["content"]["application/json"]["schema"]["$ref"] == (
        "#/components/schemas/AgentRunRequest"
    )
    assert agent_run["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] == (
        "#/components/schemas/AgentRunResponse"
    )
    assert "AgentRunRequest" in schema["components"]["schemas"]
    assert "AgentRunResponse" in schema["components"]["schemas"]
    assert (
        schema["components"]["schemas"]["ErrorResponse"]["properties"]["detail"]["type"] == "string"
    )
