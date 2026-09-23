"""Tests for embedding service primitives."""

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

import enterprise_ai_agent.embeddings.fastembed as fastembed_module
from enterprise_ai_agent.chunking import Chunk
from enterprise_ai_agent.embeddings import (
    EmbeddedChunk,
    EmbeddingService,
    FastEmbedEmbeddingService,
)
from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType


def _make_chunk(
    content: str,
    *,
    chunk_index: int = 0,
    file_name: str = "sample.txt",
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
        start_offset=0,
        end_offset=len(content),
    )


class FakeEmbeddingService(EmbeddingService):
    """Deterministic local embedding service for tests."""

    def __init__(self, dimension: int = 3) -> None:
        super().__init__(dimension)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        self.calls.append(text_list)

        vectors: list[list[float]] = []
        for index, text in enumerate(text_list):
            vector = [float(len(text)), float(index)]
            vector.extend([0.0] * max(self.dimension - len(vector), 0))
            vectors.append(vector[: self.dimension])
        return vectors


class WrongDimensionEmbeddingService(EmbeddingService):
    """Service that intentionally returns vectors with the wrong dimension."""

    def __init__(self) -> None:
        super().__init__(dimension=3)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


def test_embed_texts_returns_vectors_in_input_order() -> None:
    service = FakeEmbeddingService(dimension=3)

    vectors = service.embed_texts(["a", "bb", "ccc"])

    assert vectors == [
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 2.0, 0.0],
    ]


def test_embed_text_reuses_batch_interface() -> None:
    service = FakeEmbeddingService(dimension=2)

    vector = service.embed_text("abc")

    assert vector == [3.0, 0.0]
    assert service.calls == [["abc"]]


def test_embed_chunks_preserves_order_and_metadata() -> None:
    service = FakeEmbeddingService(dimension=2)
    first = _make_chunk("alpha", chunk_index=0, file_name="first.txt")
    second = _make_chunk("beta", chunk_index=1, file_name="second.txt")

    embedded_chunks = service.embed_chunks([first, second])

    assert [embedded.chunk for embedded in embedded_chunks] == [first, second]
    assert [embedded.vector for embedded in embedded_chunks] == [
        (5.0, 0.0),
        (4.0, 1.0),
    ]
    assert [embedded.chunk.metadata for embedded in embedded_chunks] == [
        first.metadata,
        second.metadata,
    ]


def test_empty_batches_return_empty_lists() -> None:
    service = FakeEmbeddingService()

    assert service.embed_texts([]) == []
    assert service.embed_chunks([]) == []


def test_embedding_service_rejects_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FakeEmbeddingService(dimension=0)


def test_embedding_service_rejects_wrong_vector_dimension() -> None:
    service = WrongDimensionEmbeddingService()

    with pytest.raises(ValueError, match="dimension mismatch"):
        service.embed_text("content")


def test_embedded_chunk_rejects_empty_vector() -> None:
    with pytest.raises(ValidationError):
        EmbeddedChunk(chunk=_make_chunk("content"), vector=())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_embedded_chunk_rejects_non_finite_vector_values(value: float) -> None:
    with pytest.raises(ValidationError):
        EmbeddedChunk(chunk=_make_chunk("content"), vector=(1.0, value))


class FakeTextEmbedding:
    """FastEmbed replacement that never downloads a model."""

    instances: list["FakeTextEmbedding"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[list[str], int]] = []
        self.__class__.instances.append(self)

    def embed(
        self,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        text_list = list(documents)
        self.calls.append((text_list, batch_size))
        return [[float(len(text)), float(index)] for index, text in enumerate(text_list)]


def test_fastembed_service_is_lazy_and_uses_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTextEmbedding.instances.clear()
    monkeypatch.setattr(fastembed_module, "TextEmbedding", FakeTextEmbedding)
    service = FastEmbedEmbeddingService(
        model_name="test/model",
        dimension=2,
        batch_size=7,
        cache_dir="/tmp/models",
    )

    assert FakeTextEmbedding.instances == []

    vectors = service.embed_texts(["a", "bb"])
    model = FakeTextEmbedding.instances[0]

    assert vectors == [[1.0, 0.0], [2.0, 1.0]]
    assert model.kwargs == {
        "model_name": "test/model",
        "cache_dir": "/tmp/models",
        "lazy_load": True,
    }
    assert model.calls == [(["a", "bb"], 7)]


def test_fastembed_empty_batch_does_not_initialize_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeTextEmbedding.instances.clear()
    monkeypatch.setattr(fastembed_module, "TextEmbedding", FakeTextEmbedding)
    service = FastEmbedEmbeddingService(dimension=2)

    assert service.embed_texts([]) == []
    assert FakeTextEmbedding.instances == []


def test_fastembed_service_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch size"):
        FastEmbedEmbeddingService(batch_size=0)


def test_fastembed_service_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model name"):
        FastEmbedEmbeddingService(model_name="")
