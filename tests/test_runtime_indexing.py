"""Tests for the runtime document indexing entry point."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from enterprise_ai_agent.core.config import AppSettings
from enterprise_ai_agent.embeddings import EmbeddedChunk, EmbeddingService
from enterprise_ai_agent.runtime import indexing
from enterprise_ai_agent.vector_store import SearchResult, VectorStore


class FakeEmbeddingService(EmbeddingService):
    """Deterministic embedding service."""

    def __init__(self) -> None:
        super().__init__(dimension=2)
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = list(texts)
        self.calls.append(text_list)
        return [[1.0, 0.0] for _ in text_list]


class FakeVectorStore(VectorStore):
    """Recording vector store."""

    def __init__(self) -> None:
        self.upserted: list[EmbeddedChunk] = []

    def ensure_collection(self, dimension: int) -> None:
        pass

    def recreate_collection(self, dimension: int) -> None:
        pass

    def upsert(self, embedded_chunks: Sequence[EmbeddedChunk]) -> int:
        chunk_list = list(embedded_chunks)
        self.upserted.extend(chunk_list)
        return len(chunk_list)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        return []


def _make_settings() -> AppSettings:
    return AppSettings(
        _env_file=None,
        llm_api_key="test-key",
        llm_model="test-model",
        embedding_dimension=2,
    )


def test_index_directory_loads_supported_files_and_upserts_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "first.md").write_text("# First\n\nAlpha content.", encoding="utf-8")
    (tmp_path / "second.txt").write_text("Beta content.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("unsupported", encoding="utf-8")
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    monkeypatch.setattr(
        indexing,
        "_build_embedding_service",
        lambda settings: embedding_service,
    )
    monkeypatch.setattr(
        indexing,
        "_build_vector_store",
        lambda settings, embedding: vector_store,
    )

    indexed = indexing.index_directory(tmp_path, settings=_make_settings())

    assert indexed == 2
    assert embedding_service.calls == [["# First\n\nAlpha content."], ["Beta content."]]
    assert len(vector_store.upserted) == 2


def test_index_files_indexes_explicit_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    monkeypatch.setattr(
        indexing,
        "_build_embedding_service",
        lambda settings: embedding_service,
    )
    monkeypatch.setattr(
        indexing,
        "_build_vector_store",
        lambda settings, embedding: vector_store,
    )

    indexed = indexing.index_files((second, first), settings=_make_settings())

    assert indexed == 2
    assert embedding_service.calls == [["second"], ["first"]]


def test_index_directory_rejects_missing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        indexing,
        "_build_embedding_service",
        lambda settings: FakeEmbeddingService(),
    )
    monkeypatch.setattr(
        indexing,
        "_build_vector_store",
        lambda settings, embedding: FakeVectorStore(),
    )

    with pytest.raises(FileNotFoundError):
        indexing.index_directory(tmp_path / "missing", settings=_make_settings())


def test_indexing_main_reports_chunk_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(indexing, "index_path", lambda path: 3)

    exit_code = indexing.main(["demo/documents"])

    assert exit_code == 0
    assert "Indexed 3 chunks from demo/documents" in capsys.readouterr().out
