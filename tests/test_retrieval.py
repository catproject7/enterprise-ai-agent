"""Tests for the semantic retrieval pipeline."""

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from enterprise_ai_agent.chunking import Chunk
from enterprise_ai_agent.embeddings import EmbeddedChunk, EmbeddingService
from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.retrieval import Retriever
from enterprise_ai_agent.vector_store import (
    QdrantVectorStore,
    SearchResult,
    VectorStore,
)


def _make_chunk(content: str, *, chunk_index: int = 0) -> Chunk:
    return Chunk(
        content=content,
        metadata=DocumentMetadata(
            source="/documents/sample.txt",
            file_name="sample.txt",
            file_type=DocumentType.TEXT,
            file_size=len(content.encode("utf-8")),
        ),
        chunk_index=chunk_index,
        start_offset=0,
        end_offset=len(content),
    )


def _search_result(content: str, *, score: float = 1.0) -> SearchResult:
    return SearchResult(
        id=f"id-{content}",
        score=score,
        chunk=_make_chunk(content),
    )


class FakeEmbeddingService(EmbeddingService):
    """Recording embedding service that never downloads a model."""

    def __init__(
        self,
        vector: Sequence[float],
        *,
        event_log: list[str] | None = None,
    ) -> None:
        super().__init__(dimension=len(vector))
        self.vector = list(vector)
        self.calls: list[list[str]] = []
        self.event_log = event_log

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        self.calls.append(text_list)
        if self.event_log is not None:
            self.event_log.append("embed_text")
        return [self.vector for _ in text_list]


class FailingEmbeddingService(EmbeddingService):
    """Embedding service that raises a controlled error."""

    def __init__(self, error: ValueError) -> None:
        super().__init__(dimension=2)
        self.error = error

    def embed_text(self, text: str) -> list[float]:
        raise self.error

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("embed_texts should not be called")


class FakeVectorStore(VectorStore):
    """Recording vector store with configurable results and errors."""

    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        error: ValueError | None = None,
        event_log: list[str] | None = None,
    ) -> None:
        self.results = results if results is not None else []
        self.error = error
        self.calls: list[tuple[list[float], int]] = []
        self.event_log = event_log

    def ensure_collection(self, dimension: int) -> None:
        raise AssertionError("ensure_collection should not be called")

    def recreate_collection(self, dimension: int) -> None:
        raise AssertionError("recreate_collection should not be called")

    def upsert(self, embedded_chunks: Sequence[EmbeddedChunk]) -> int:
        raise AssertionError("upsert should not be called")

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((list(query_vector), limit))
        if self.event_log is not None:
            self.event_log.append("search")
        if self.error is not None:
            raise self.error
        return self.results


def test_retrieve_calls_embedding_with_original_query() -> None:
    embedding_service = FakeEmbeddingService([1.0, 2.0])
    vector_store = FakeVectorStore([_search_result("alpha")])
    retriever = Retriever(embedding_service, vector_store)
    query = "  alpha query  "

    retriever.retrieve(query)

    assert embedding_service.calls == [[query]]


def test_retrieve_passes_embedding_vector_to_vector_store() -> None:
    embedding_service = FakeEmbeddingService([1.0, 2.0, 3.0])
    vector_store = FakeVectorStore([_search_result("alpha")])
    retriever = Retriever(embedding_service, vector_store)

    retriever.retrieve("alpha")

    assert vector_store.calls == [([1.0, 2.0, 3.0], 5)]


def test_retrieve_uses_default_limit() -> None:
    embedding_service = FakeEmbeddingService([1.0, 0.0])
    vector_store = FakeVectorStore()
    retriever = Retriever(embedding_service, vector_store)

    retriever.retrieve("alpha")

    assert vector_store.calls[0][1] == 5


def test_retrieve_uses_custom_limit() -> None:
    embedding_service = FakeEmbeddingService([1.0, 0.0])
    vector_store = FakeVectorStore()
    retriever = Retriever(embedding_service, vector_store)

    retriever.retrieve("alpha", limit=2)

    assert vector_store.calls[0][1] == 2


def test_retrieve_returns_vector_store_results_without_changes() -> None:
    results = [
        _search_result("alpha", score=0.9),
        _search_result("beta", score=0.8),
    ]
    embedding_service = FakeEmbeddingService([1.0, 0.0])
    vector_store = FakeVectorStore(results)
    retriever = Retriever(embedding_service, vector_store)

    retrieved = retriever.retrieve("alpha")

    assert retrieved is results
    assert [result.chunk.content for result in retrieved] == ["alpha", "beta"]


@pytest.mark.parametrize("query", ["", " ", "\n\t", "\r\n"])
def test_retrieve_rejects_empty_or_whitespace_query(query: str) -> None:
    embedding_service = FakeEmbeddingService([1.0, 0.0])
    vector_store = FakeVectorStore()
    retriever = Retriever(embedding_service, vector_store)

    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(query)

    assert embedding_service.calls == []
    assert vector_store.calls == []


@pytest.mark.parametrize("limit", [0, -1])
def test_retrieve_rejects_invalid_limit_before_embedding(limit: int) -> None:
    embedding_service = FakeEmbeddingService([1.0, 0.0])
    vector_store = FakeVectorStore()
    retriever = Retriever(embedding_service, vector_store)

    with pytest.raises(ValueError, match="limit"):
        retriever.retrieve("alpha", limit=limit)

    assert embedding_service.calls == []
    assert vector_store.calls == []


def test_retrieve_propagates_embedding_error() -> None:
    error = ValueError("embedding failed")
    embedding_service = FailingEmbeddingService(error)
    vector_store = FakeVectorStore()
    retriever = Retriever(embedding_service, vector_store)

    with pytest.raises(ValueError, match="embedding failed") as captured:
        retriever.retrieve("alpha")

    assert captured.value is error
    assert vector_store.calls == []


def test_retrieve_propagates_vector_store_error() -> None:
    error = ValueError("search failed")
    embedding_service = FakeEmbeddingService([1.0, 0.0])
    vector_store = FakeVectorStore(error=error)
    retriever = Retriever(embedding_service, vector_store)

    with pytest.raises(ValueError, match="search failed") as captured:
        retriever.retrieve("alpha")

    assert captured.value is error


def test_retrieve_calls_embedding_before_vector_search() -> None:
    event_log: list[str] = []
    embedding_service = FakeEmbeddingService([1.0, 0.0], event_log=event_log)
    vector_store = FakeVectorStore(event_log=event_log)
    retriever = Retriever(embedding_service, vector_store)

    retriever.retrieve("alpha")

    assert event_log == ["embed_text", "search"]


class MappingEmbeddingService(EmbeddingService):
    """Deterministic embedding service used for Qdrant integration."""

    def __init__(self) -> None:
        super().__init__(dimension=2)
        self.vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_retrieve_with_in_memory_qdrant_round_trip() -> None:
    client = QdrantClient(location=":memory:")
    try:
        vector_store = QdrantVectorStore(client, collection_name="retrieval_test")
        embedding_service = MappingEmbeddingService()
        vector_store.ensure_collection(embedding_service.dimension)
        vector_store.upsert(
            [
                EmbeddedChunk(
                    chunk=_make_chunk("alpha", chunk_index=0),
                    vector=(1.0, 0.0),
                ),
                EmbeddedChunk(
                    chunk=_make_chunk("beta", chunk_index=1),
                    vector=(0.0, 1.0),
                ),
            ]
        )
        retriever = Retriever(embedding_service, vector_store)

        results = retriever.retrieve("alpha", limit=1)

        assert len(results) == 1
        assert results[0].chunk.content == "alpha"
        assert results[0].chunk.metadata.file_name == "sample.txt"
    finally:
        client.close()
