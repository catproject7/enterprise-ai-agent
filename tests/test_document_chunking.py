"""Tests for document chunking."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.chunking import (
    Chunk,
    ChunkingConfig,
    chunk_document,
    chunk_documents,
)
from enterprise_ai_agent.ingestion import (
    Document,
    DocumentMetadata,
    DocumentType,
    load_document,
)


def _make_document(
    content: str,
    *,
    file_name: str = "sample.txt",
    file_type: DocumentType = DocumentType.TEXT,
) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source=f"/documents/{file_name}",
            file_name=file_name,
            file_type=file_type,
            file_size=len(content.encode("utf-8")),
        ),
    )


def test_chunk_document_short_text_returns_single_chunk() -> None:
    document = _make_document("short text")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=100, chunk_overlap=20),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "short text"
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len("short text")


def test_chunk_document_text_equal_to_chunk_size_returns_single_chunk() -> None:
    document = _make_document("abcde")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=5, chunk_overlap=2),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "abcde"
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == 5


def test_chunk_document_creates_fixed_windows() -> None:
    document = _make_document("abcdefghijkl")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=5, chunk_overlap=0),
    )

    assert [chunk.content for chunk in chunks] == ["abcde", "fghij", "kl"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 5),
        (5, 10),
        (10, 12),
    ]


def test_chunk_document_preserves_exact_overlap() -> None:
    document = _make_document("abcdefghij")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=5, chunk_overlap=2),
    )

    assert [chunk.content for chunk in chunks] == ["abcde", "defgh", "ghij"]
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.start_offset == previous.end_offset - 2


def test_chunk_document_supports_zero_overlap() -> None:
    document = _make_document("abcdefgh")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=3, chunk_overlap=0),
    )

    assert [chunk.content for chunk in chunks] == ["abc", "def", "gh"]
    assert chunks[1].start_offset == chunks[0].end_offset


def test_chunk_document_supports_chunk_size_one() -> None:
    document = _make_document("abc")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=1, chunk_overlap=0),
    )

    assert [chunk.content for chunk in chunks] == ["a", "b", "c"]
    assert [chunk.start_offset for chunk in chunks] == [0, 1, 2]


def test_chunk_document_supports_maximum_overlap() -> None:
    document = _make_document("abcdef")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=3, chunk_overlap=2),
    )

    assert [chunk.content for chunk in chunks] == ["abc", "bcd", "cde", "def"]
    assert all(
        current.start_offset == previous.end_offset - 2
        for previous, current in zip(chunks, chunks[1:], strict=False)
    )


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunking_config_rejects_non_positive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=chunk_size, chunk_overlap=0)


def test_chunking_config_rejects_negative_overlap() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=5, chunk_overlap=-1)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(3, 3), (3, 4)],
)
def test_chunking_config_rejects_overlap_not_smaller_than_size(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunk_document_empty_string_returns_empty_list() -> None:
    assert chunk_document(_make_document("")) == []


@pytest.mark.parametrize("content", [" ", "\n\t ", "\r\n"])
def test_chunk_document_whitespace_only_returns_empty_list(content: str) -> None:
    assert chunk_document(_make_document(content)) == []


def test_chunk_document_preserves_whitespace_windows_inside_document() -> None:
    document = _make_document("abc" + (" " * 6) + "def")

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=3, chunk_overlap=0),
    )

    assert [chunk.content for chunk in chunks] == ["abc", "   ", "   ", "def"]


def test_chunk_documents_empty_input_returns_empty_list() -> None:
    assert chunk_documents([]) == []


def test_chunk_documents_preserves_document_order_and_resets_indices() -> None:
    first = _make_document("abcdef", file_name="first.txt")
    second = _make_document("xyz", file_name="second.txt")

    chunks = chunk_documents(
        [first, second],
        config=ChunkingConfig(chunk_size=4, chunk_overlap=0),
    )

    assert [chunk.content for chunk in chunks] == ["abcd", "ef", "xyz"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 0]
    assert [chunk.metadata.file_name for chunk in chunks] == [
        "first.txt",
        "first.txt",
        "second.txt",
    ]


@pytest.mark.parametrize("file_type", list(DocumentType))
def test_chunk_document_inherits_metadata_and_document_type(
    file_type: DocumentType,
) -> None:
    document = _make_document(
        "content",
        file_name=f"sample.{file_type.value}",
        file_type=file_type,
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].metadata == document.metadata
    assert chunks[0].metadata.file_type == file_type


def test_chunk_document_does_not_modify_input_document() -> None:
    document = _make_document("abcdefghij")
    before = document.model_dump()

    chunk_document(
        document,
        config=ChunkingConfig(chunk_size=4, chunk_overlap=1),
    )

    assert document.model_dump() == before


def test_chunk_offsets_match_original_content() -> None:
    content = "0123456789"
    document = _make_document(content)

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=4, chunk_overlap=1),
    )

    for chunk in chunks:
        assert chunk.content == content[chunk.start_offset : chunk.end_offset]
        assert len(chunk.content) == chunk.end_offset - chunk.start_offset
        assert chunk.start_offset >= 0
        assert chunk.end_offset <= len(content)


def test_last_chunk_terminates_at_document_end_without_duplicate() -> None:
    content = "abcdefghij"
    document = _make_document(content)

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=4, chunk_overlap=1),
    )

    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]
    assert chunks[-1].content == "ghij"
    assert chunks[-1].end_offset == len(content)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": "abc", "chunk_index": -1, "start_offset": 0, "end_offset": 3},
        {"content": "abc", "chunk_index": 0, "start_offset": -1, "end_offset": 3},
        {"content": "", "chunk_index": 0, "start_offset": 5, "end_offset": 5},
        {"content": "ab", "chunk_index": 0, "start_offset": 0, "end_offset": 3},
    ],
)
def test_chunk_rejects_invalid_self_boundaries(kwargs: dict[str, object]) -> None:
    metadata = _make_document("abc").metadata

    with pytest.raises(ValidationError):
        Chunk(metadata=metadata, **kwargs)


def test_chunking_models_are_frozen() -> None:
    config = ChunkingConfig()
    chunk = chunk_document(_make_document("content"))[0]

    with pytest.raises(ValidationError):
        config.chunk_size = 10
    with pytest.raises(ValidationError):
        chunk.content = "changed"


def test_load_document_to_chunk_document_integration(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")
    document = load_document(path)

    chunks = chunk_document(
        document,
        config=ChunkingConfig(chunk_size=5, chunk_overlap=2),
    )

    assert [chunk.content for chunk in chunks] == ["hello", "lo wo", "world"]
    assert all(chunk.metadata == document.metadata for chunk in chunks)
