"""Tests for the agent-facing tool abstraction and RAG tool."""

import pytest

from enterprise_ai_agent.rag import Answer, RAGPipeline, RAGResponse
from enterprise_ai_agent.tools import RAGTool, Tool


class FakeTool(Tool[str, str]):
    """Minimal in-memory tool used to exercise the interface."""

    name = "fake"
    description = "Return the supplied input."

    def run(self, input: str) -> str:
        return input


class FakePipeline(RAGPipeline):
    """Recording RAG pipeline that never calls production infrastructure."""

    def __init__(
        self,
        response: RAGResponse,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[str] = []

    def run(self, question: str) -> RAGResponse:
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        return self.response


def _make_response(text: str = "answer") -> RAGResponse:
    return RAGResponse(answer=Answer(text=text))


def test_tool_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tool()  # type: ignore[abstract]


def test_fake_tool_satisfies_interface() -> None:
    tool: Tool[str, str] = FakeTool()

    assert isinstance(tool, Tool)
    assert tool.name == "fake"
    assert tool.description == "Return the supplied input."
    assert tool.run("hello") == "hello"


def test_rag_tool_exposes_stable_name_and_description() -> None:
    tool = RAGTool(FakePipeline(_make_response()))

    assert tool.name == "rag"
    assert tool.description == "Answer a user query using the knowledge-base RAG pipeline."


def test_rag_tool_uses_injected_pipeline() -> None:
    response = _make_response()
    pipeline = FakePipeline(response)
    tool = RAGTool(pipeline)

    result = tool.run("question")

    assert result is response
    assert pipeline.calls == ["question"]


def test_rag_tool_passes_query_unchanged() -> None:
    pipeline = FakePipeline(_make_response())
    tool = RAGTool(pipeline)
    query = "  What is alpha?  "

    tool.run(query)

    assert pipeline.calls == [query]


@pytest.mark.parametrize("query", ["", " ", "\n\t", "\r\n"])
def test_rag_tool_does_not_reimplement_query_validation(query: str) -> None:
    response = _make_response()
    pipeline = FakePipeline(response)
    tool = RAGTool(pipeline)

    result = tool.run(query)

    assert result is response
    assert pipeline.calls == [query]


def test_rag_tool_returns_pipeline_response_unchanged() -> None:
    response = _make_response("  answer with spaces  ")
    tool = RAGTool(FakePipeline(response))

    result = tool.run("question")

    assert result is response


def test_rag_tool_propagates_pipeline_error_unchanged() -> None:
    error = RuntimeError("pipeline failed")
    tool = RAGTool(FakePipeline(_make_response(), error=error))

    with pytest.raises(RuntimeError, match="pipeline failed") as captured:
        tool.run("question")

    assert captured.value is error


def test_rag_tool_is_deterministic() -> None:
    response = _make_response()
    pipeline = FakePipeline(response)
    tool = RAGTool(pipeline)

    first = tool.run("question")
    second = tool.run("question")

    assert first is response
    assert second is response
    assert pipeline.calls == ["question", "question"]


def test_rag_tool_does_not_modify_injected_pipeline_or_response() -> None:
    response = _make_response()
    pipeline = FakePipeline(response)
    original_response = response.model_dump()
    tool = RAGTool(pipeline)

    tool.run("question")

    assert pipeline.response is response
    assert response.model_dump() == original_response
