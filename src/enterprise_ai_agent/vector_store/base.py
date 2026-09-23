"""Vector store abstraction."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from enterprise_ai_agent.embeddings.models import EmbeddedChunk

from .models import SearchResult


class VectorStore(ABC):
    """Replaceable interface for vector persistence and similarity search."""

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection when missing and validate its vector dimension."""

    @abstractmethod
    def recreate_collection(self, dimension: int) -> None:
        """Delete and recreate the collection with the requested dimension."""

    @abstractmethod
    def upsert(self, embedded_chunks: Sequence[EmbeddedChunk]) -> int:
        """Insert or replace embedded chunks and return the number written."""

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Return the nearest chunks for a query vector."""
