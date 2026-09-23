"""Embedding service abstraction and shared operations."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from math import isfinite

from enterprise_ai_agent.chunking.models import Chunk

from .models import EmbeddedChunk


class EmbeddingService(ABC):
    """Replaceable interface for producing text embedding vectors."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be greater than zero")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the number of values in each embedding vector."""

        return self._dimension

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts in input order."""

    def embed_text(self, text: str) -> list[float]:
        """Embed one text through the batch interface."""

        vectors = self.embed_texts([text])
        self._validate_vectors(vectors, expected_count=1)
        return vectors[0]

    def embed_chunks(self, chunks: Iterable[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks while preserving their input order and metadata."""

        chunk_list = list(chunks)
        if not chunk_list:
            return []

        vectors = self.embed_texts([chunk.content for chunk in chunk_list])
        self._validate_vectors(vectors, expected_count=len(chunk_list))

        return [
            EmbeddedChunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunk_list, vectors, strict=True)
        ]

    def _validate_vectors(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        expected_count: int,
    ) -> None:
        if len(vectors) != expected_count:
            raise ValueError(
                f"embedding service returned {len(vectors)} vectors for {expected_count} texts"
            )

        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )
            if not all(isfinite(float(value)) for value in vector):
                raise ValueError("embedding vector must contain only finite values")
