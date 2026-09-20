"""Document chunking primitives."""

from .models import Chunk, ChunkingConfig
from .splitter import chunk_document, chunk_documents

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "chunk_document",
    "chunk_documents",
]
