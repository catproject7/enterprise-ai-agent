"""Tests for the Qdrant vector store adapter."""

from collections.abc import Sequence

import pytest
from qdrant_client import QdrantClient

from enterprise_ai_agent.chunking import Chunk
from enterprise_ai_agent.embeddings import EmbeddedChunk, EmbeddingService
from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.vector_store import QdrantVectorStore


def _make_chunk(
    content: str,
    *,
    chunk_index: int = 0,
    file_name: str = "sample.txt",
    start_offset: int = 0,
) -> Chunk:
    return Chunk(
        content=content,
        metadata=DocumentMetadata(
            source=f"/documents/{file_name}",
            file_name=file_name,
            file_type=DocumentType.TEXT,
            file_size=len(content.encode("utf-8")),
        ),
        chunk_index=chunk_index,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
    )


def _embedded_chunk(
    content: str,
    vector: Sequence[float],
    *,
    chunk_index: int = 0,
    file_name: str = "sample.txt",
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=_make_chunk(
            content,
            chunk_index=chunk_index,
            file_name=file_name,
        ),
        vector=tuple(vector),
    )


@pytest.fixture
def client() -> QdrantClient:
    qdrant_client = QdrantClient(location=":memory:")
    yield qdrant_client
    qdrant_client.close()


@pytest.fixture
def store(client: QdrantClient) -> QdrantVectorStore:
    return QdrantVectorStore(client, collection_name="test_chunks")


def test_ensure_collection_creates_collection_idempotently(
    client: QdrantClient,
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(2)
    store.ensure_collection(2)

    assert client.collection_exists("test_chunks")


def test_ensure_collection_rejects_dimension_mismatch(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(2)

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.ensure_collection(3)


def test_recreate_collection_removes_existing_points(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(2)
    store.upsert([_embedded_chunk("alpha", [1.0, 0.0])])

    store.recreate_collection(2)

    assert store.search([1.0, 0.0]) == []


def test_search_orders_results_by_cosine_similarity(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(2)
    store.upsert(
        [
            _embedded_chunk("alpha", [1.0, 0.0]),
            _embedded_chunk("beta", [0.0, 1.0]),
            _embedded_chunk("gamma", [0.8, 0.2]),
        ]
    )

    results = store.search([1.0, 0.0], limit=3)

    assert [result.chunk.content for result in results] == ["alpha", "gamma", "beta"]
    assert [result.score for result in results] == sorted(
        [result.score for result in results],
        reverse=True,
    )


def test_search_round_trips_complete_chunk_metadata(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(2)
    original = _embedded_chunk(
        "metadata content",
        [1.0, 0.0],
        chunk_index=4,
        file_name="metadata.md",
    )
    store.upsert([original])

    result = store.search([1.0, 0.0])[0]

    assert result.chunk == original.chunk
    assert result.chunk.metadata.source == "/documents/metadata.md"
    assert result.chunk.metadata.file_name == "metadata.md"
    assert result.chunk.metadata.file_type == DocumentType.TEXT
    assert result.chunk.metadata.file_size == len("metadata content")
    assert result.chunk.chunk_index == 4
    assert result.chunk.start_offset == 0
    assert result.chunk.end_offset == len("metadata content")


def test_duplicate_upsert_is_idempotent(
    client: QdrantClient,
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(2)
    embedded = _embedded_chunk("alpha", [1.0, 0.0])

    assert store.upsert([embedded]) == 1
    assert store.upsert([embedded]) == 1
    assert client.count("test_chunks").count == 1


def test_upsert_rejects_vector_dimension_mismatch(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(3)

    with pytest.raises(ValueError, match="embedding vector dimension mismatch"):
        store.upsert([_embedded_chunk("alpha", [1.0, 0.0])])


def test_search_rejects_query_vector_dimension_mismatch(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(3)

    with pytest.raises(ValueError, match="query vector dimension mismatch"):
        store.search([1.0, 0.0])


@pytest.mark.parametrize("limit", [0, -1])
def test_search_rejects_invalid_limit(
    store: QdrantVectorStore,
    limit: int,
) -> None:
    store.ensure_collection(2)

    with pytest.raises(ValueError, match="limit"):
        store.search([1.0, 0.0], limit=limit)


def test_search_returns_empty_list_for_empty_collection(
    store: QdrantVectorStore,
) -> None:
    store.ensure_collection(2)

    assert store.search([1.0, 0.0]) == []


def test_empty_upsert_returns_zero(store: QdrantVectorStore) -> None:
    assert store.upsert([]) == 0


class MappingEmbeddingService(EmbeddingService):
    """Small embedding service used for the end-to-end test."""

    def __init__(self) -> None:
        super().__init__(dimension=2)
        self.vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_chunk_to_embedding_to_qdrant_round_trip(
    store: QdrantVectorStore,
) -> None:
    embedding_service = MappingEmbeddingService()
    chunks = [
        _make_chunk("alpha", chunk_index=0),
        _make_chunk("beta", chunk_index=1),
    ]

    embedded_chunks = embedding_service.embed_chunks(chunks)
    store.ensure_collection(embedding_service.dimension)
    assert store.upsert(embedded_chunks) == 2

    results = store.search([1.0, 0.0], limit=1)

    assert len(results) == 1
    assert results[0].chunk == chunks[0]
    assert results[0].chunk.metadata == chunks[0].metadata
