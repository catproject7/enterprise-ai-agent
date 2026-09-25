"""Tests for the citation-aware RAG query API."""

from fastapi.testclient import TestClient

from enterprise_ai_agent.api import create_app
from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.rag import Answer, Citation, RAGPipeline, RAGResponse


class FakeRAGPipeline(RAGPipeline):
    """Recording RAG pipeline that returns a deterministic response."""

    def __init__(self, response: RAGResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    def run(self, question: str) -> RAGResponse:
        self.calls.append(question)
        return self.response


def _make_response() -> RAGResponse:
    citation = Citation(
        result_id="result-1",
        metadata=DocumentMetadata(
            source="/documents/example.md",
            file_name="example.md",
            file_type=DocumentType.MARKDOWN,
            file_size=100,
        ),
        chunk_index=0,
        start_offset=0,
        end_offset=10,
    )
    return RAGResponse(
        answer=Answer(text="Rotate API keys every 90 days."),
        citations=(citation,),
    )


def test_rag_query_returns_answer_and_citations() -> None:
    pipeline = FakeRAGPipeline(_make_response())
    client = TestClient(create_app(rag_pipeline=pipeline))

    response = client.post(
        "/rag/query",
        json={"question": "How often should API keys be rotated?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Rotate API keys every 90 days."
    assert body["citations"][0]["result_id"] == "result-1"
    assert body["citations"][0]["metadata"]["file_name"] == "example.md"
    assert pipeline.calls == ["How often should API keys be rotated?"]


def test_rag_query_returns_503_without_pipeline() -> None:
    client = TestClient(create_app())

    response = client.post("/rag/query", json={"question": "question"})

    assert response.status_code == 503
    assert response.json() == {"detail": "RAG pipeline is not configured"}


def test_rag_query_rejects_empty_question() -> None:
    client = TestClient(create_app(rag_pipeline=FakeRAGPipeline(_make_response())))

    response = client.post("/rag/query", json={"question": "   "})

    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}
