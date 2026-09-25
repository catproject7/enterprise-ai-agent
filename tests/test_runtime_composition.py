"""Tests for runtime composition without external infrastructure."""

from collections.abc import Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from enterprise_ai_agent.agent import Agent, AgentResult, LLMAgent
from enterprise_ai_agent.chunking import Chunk
from enterprise_ai_agent.core.config import AppSettings
from enterprise_ai_agent.embeddings import EmbeddedChunk, EmbeddingService
from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.llm import (
    LLMService,
    ToolCall,
    ToolCallingLLM,
    ToolCallingResponse,
    ToolResult,
    ToolSpec,
)
from enterprise_ai_agent.observability import TracedAgent
from enterprise_ai_agent.runtime import (
    RuntimeConfigurationError,
    assemble_agent,
    composition,
    create_runtime,
    create_runtime_app,
)
from enterprise_ai_agent.vector_store import SearchResult, VectorStore


class FakeEmbeddingService(EmbeddingService):
    """Deterministic embedding service."""

    def __init__(self, dimension: int = 2) -> None:
        super().__init__(dimension)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        self.calls.append(text_list)
        return [[1.0, 0.0] for _ in text_list]


class FakeVectorStore(VectorStore):
    """Recording vector store that returns configured results."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results if results is not None else []
        self.ensured_dimensions: list[int] = []
        self.searches: list[tuple[list[float], int]] = []

    def ensure_collection(self, dimension: int) -> None:
        self.ensured_dimensions.append(dimension)

    def recreate_collection(self, dimension: int) -> None:
        self.ensured_dimensions.append(dimension)

    def upsert(self, embedded_chunks: Sequence[EmbeddedChunk]) -> int:
        return len(embedded_chunks)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        self.searches.append((list(query_vector), limit))
        return self.results


class FakeLLMService(LLMService):
    """Deterministic RAG text generation service."""

    def __init__(self, response: str = "rag answer") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeToolCallingLLM(ToolCallingLLM):
    """Scripted tool-calling LLM."""

    def __init__(self, responses: list[ToolCallingResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, tuple[ToolSpec, ...], tuple[ToolResult, ...]]] = []

    def generate_with_tools(
        self,
        prompt: str,
        tools: Sequence[ToolSpec],
        tool_results: Sequence[ToolResult] = (),
    ) -> ToolCallingResponse:
        self.calls.append((prompt, tuple(tools), tuple(tool_results)))
        return self.responses.pop(0)


class FakeAgent(Agent[str]):
    """Simple fake Agent used by runtime app tests."""

    def run(self, input: str) -> AgentResult[str]:
        return AgentResult(output=f"answer:{input}")


def _make_result(content: str) -> SearchResult:
    return SearchResult(
        id="result-1",
        score=0.9,
        chunk=Chunk(
            content=content,
            metadata=DocumentMetadata(
                source="/documents/sample.txt",
                file_name="sample.txt",
                file_type=DocumentType.TEXT,
                file_size=100,
            ),
            chunk_index=0,
            start_offset=0,
            end_offset=len(content),
        ),
    )


def _make_settings(**overrides: object) -> AppSettings:
    values: dict[str, object] = {
        "llm_model": "test-model",
        "llm_api_key": "test-key",
        "embedding_dimension": 2,
        "qdrant_collection_name": "test_collection",
    }
    values.update(overrides)
    return AppSettings(_env_file=None, **values)


def test_assemble_agent_executes_rag_tool_through_llm_agent() -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore([_make_result("alpha")])
    llm_service = FakeLLMService("rag answer")
    tool_call = ToolCall(
        call_id="call-1",
        name="rag",
        arguments='{"query": "question"}',
    )
    tool_calling_llm = FakeToolCallingLLM(
        [
            ToolCallingResponse(tool_call=tool_call),
            ToolCallingResponse(text="final answer"),
        ]
    )

    agent = assemble_agent(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        tool_calling_llm=tool_calling_llm,
        instrument=False,
    )

    result = agent.run("question")

    assert isinstance(agent, LLMAgent)
    assert result == AgentResult(output="final answer")
    assert embedding_service.calls == [["question"]]
    assert vector_store.searches == [([1.0, 0.0], 5)]
    assert tool_calling_llm.calls[0][1][0].name == "rag"
    assert tool_calling_llm.calls[1][2][0].call == tool_call
    assert "rag answer" in tool_calling_llm.calls[1][2][0].output


def test_assemble_agent_can_disable_instrumentation() -> None:
    agent = assemble_agent(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        llm_service=FakeLLMService(),
        tool_calling_llm=FakeToolCallingLLM([ToolCallingResponse(text="answer")]),
        instrument=False,
    )

    assert type(agent) is LLMAgent


def test_assemble_agent_can_enable_instrumentation() -> None:
    agent = assemble_agent(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(),
        llm_service=FakeLLMService(),
        tool_calling_llm=FakeToolCallingLLM([ToolCallingResponse(text="answer")]),
        instrument=True,
    )

    assert isinstance(agent, TracedAgent)


def test_create_runtime_builds_shared_clients_and_initializes_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    qdrant_client = object()
    openai_client = object()
    llm_service = FakeLLMService()
    tool_calling_llm = FakeToolCallingLLM([ToolCallingResponse(text="answer")])
    expected_agent = FakeAgent()
    captured: dict[str, object] = {}
    qdrant_calls: list[dict[str, object]] = []
    openai_calls: list[dict[str, object]] = []
    llm_clients: list[object] = []
    tool_llm_clients: list[object] = []

    monkeypatch.setattr(
        composition,
        "FastEmbedEmbeddingService",
        lambda **kwargs: embedding_service,
    )
    monkeypatch.setattr(
        composition,
        "QdrantClient",
        lambda **kwargs: qdrant_calls.append(kwargs) or qdrant_client,
    )
    monkeypatch.setattr(
        composition,
        "QdrantVectorStore",
        lambda client, *, collection_name: vector_store,
    )
    monkeypatch.setattr(
        composition,
        "OpenAI",
        lambda **kwargs: openai_calls.append(kwargs) or openai_client,
    )
    monkeypatch.setattr(
        composition,
        "OpenAICompatibleLLMService",
        lambda client, *, model: llm_clients.append(client) or llm_service,
    )
    monkeypatch.setattr(
        composition,
        "OpenAICompatibleToolCallingLLM",
        lambda client, *, model: tool_llm_clients.append(client) or tool_calling_llm,
    )
    monkeypatch.setattr(
        composition,
        "assemble_agent",
        lambda **kwargs: captured.update(kwargs) or expected_agent,
    )

    agent = create_runtime(_make_settings())

    assert agent is expected_agent
    assert qdrant_calls == [{"location": ":memory:"}]
    assert openai_calls == [{"api_key": "test-key"}]
    assert vector_store.ensured_dimensions == [2]
    assert captured == {
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "llm_service": llm_service,
        "tool_calling_llm": tool_calling_llm,
    }
    assert llm_clients == [openai_client]
    assert tool_llm_clients == [openai_client]


def test_create_runtime_requires_llm_api_key() -> None:
    settings = AppSettings(
        _env_file=None,
        llm_api_key=None,
        llm_model="test-model",
    )

    with pytest.raises(RuntimeConfigurationError, match="LLM_API_KEY"):
        create_runtime(settings)


def test_create_runtime_app_injects_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeAgent()
    monkeypatch.setattr(composition, "create_runtime", lambda settings: agent)
    monkeypatch.setattr(
        composition,
        "configure_observability_logging",
        lambda level: None,
    )

    app = create_runtime_app(_make_settings())
    client = TestClient(app)
    response = client.post("/agent/run", json={"input": "hello"})

    assert isinstance(app, FastAPI)
    assert response.status_code == 200
    assert response.json() == {"output": "answer:hello"}


def test_settings_include_runtime_defaults() -> None:
    settings = AppSettings(_env_file=None)

    assert settings.llm_model == "gpt-4o-mini"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dimension == 384
    assert settings.qdrant_url == ":memory:"
    assert settings.qdrant_collection_name == "document_chunks"
    assert settings.llm_api_key is None
