"""Index local documents into the configured vector store."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from enterprise_ai_agent.chunking import ChunkingConfig, chunk_document
from enterprise_ai_agent.core.config import AppSettings, get_settings
from enterprise_ai_agent.embeddings import EmbeddingService
from enterprise_ai_agent.ingestion import DocumentIngestionError, load_document
from enterprise_ai_agent.vector_store import VectorStore

from .composition import (
    RuntimeConfigurationError,
    _build_embedding_service,
    _build_vector_store,
)

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".md", ".markdown", ".txt"})


def index_path(path: str | Path, *, settings: AppSettings | None = None) -> int:
    """Index one supported file or every supported file under a directory."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        files = (source,)
    elif source.is_dir():
        files = tuple(
            file
            for file in sorted(source.rglob("*"))
            if file.is_file() and file.suffix.lower() in _SUPPORTED_EXTENSIONS
        )
    else:
        raise ValueError(f"Unsupported indexing path: {source}")

    resolved = settings if settings is not None else get_settings()
    embedding_service = _build_embedding_service(resolved)
    vector_store = _build_vector_store(resolved, embedding_service)
    return _index_files(files, embedding_service, vector_store)


def index_directory(path: str | Path, *, settings: AppSettings | None = None) -> int:
    """Index supported files below a directory."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_dir():
        raise NotADirectoryError(source)
    return index_path(source, settings=settings)


def index_files(
    paths: Sequence[str | Path],
    *,
    settings: AppSettings | None = None,
) -> int:
    """Index an explicit ordered collection of files."""

    resolved = settings if settings is not None else get_settings()
    embedding_service = _build_embedding_service(resolved)
    vector_store = _build_vector_store(resolved, embedding_service)
    return _index_files(tuple(Path(path) for path in paths), embedding_service, vector_store)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the document indexing module from the command line."""

    parser = argparse.ArgumentParser(description="Index local documents into Qdrant.")
    parser.add_argument("path", help="File or directory containing PDF, Markdown, or TXT files")
    args = parser.parse_args(argv)

    try:
        indexed_chunks = index_path(args.path)
    except (
        DocumentIngestionError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeConfigurationError,
        ValueError,
    ) as error:
        print(f"Indexing failed: {error}", file=sys.stderr)
        return 1

    print(f"Indexed {indexed_chunks} chunks from {args.path}")
    return 0


def _index_files(
    files: Sequence[Path],
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
) -> int:
    chunking_config = ChunkingConfig()
    indexed_chunks = 0

    for file in files:
        document = load_document(file)
        chunks = chunk_document(document, config=chunking_config)
        embedded_chunks = embedding_service.embed_chunks(chunks)
        indexed_chunks += vector_store.upsert(embedded_chunks)

    return indexed_chunks


if __name__ == "__main__":
    raise SystemExit(main())
