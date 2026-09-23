"""Embedding service primitives."""

from .fastembed import FastEmbedEmbeddingService
from .models import EmbeddedChunk
from .service import EmbeddingService

__all__ = [
    "EmbeddedChunk",
    "EmbeddingService",
    "FastEmbedEmbeddingService",
]
