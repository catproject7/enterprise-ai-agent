"""Fixed-size document chunking."""

from collections.abc import Iterable, Iterator

from enterprise_ai_agent.ingestion.document import Document

from .models import Chunk, ChunkingConfig


def _iter_chunk_offsets(
    text_length: int,
    config: ChunkingConfig,
) -> Iterator[tuple[int, int]]:
    start = 0

    while start < text_length:
        end = min(start + config.chunk_size, text_length)
        yield start, end

        if end == text_length:
            break

        start = end - config.chunk_overlap


def _chunk_document(document: Document, config: ChunkingConfig) -> list[Chunk]:
    if not document.content.strip():
        return []

    return [
        Chunk(
            content=document.content[start:end],
            metadata=document.metadata,
            chunk_index=chunk_index,
            start_offset=start,
            end_offset=end,
        )
        for chunk_index, (start, end) in enumerate(
            _iter_chunk_offsets(len(document.content), config)
        )
    ]


def chunk_document(
    document: Document,
    *,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Split one document into fixed-size overlapping chunks."""

    return _chunk_document(document, config or ChunkingConfig())


def chunk_documents(
    documents: Iterable[Document],
    *,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Split documents in input order and return a flat chunk list."""

    settings = config or ChunkingConfig()
    return [chunk for document in documents for chunk in _chunk_document(document, settings)]
