"""Semantic retrieval pipeline."""

from enterprise_ai_agent.embeddings import EmbeddingService
from enterprise_ai_agent.vector_store import SearchResult, VectorStore


class Retriever:
    """Compose query embedding and vector similarity search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Embed a text query and return its nearest indexed chunks."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_vector = self._embedding_service.embed_text(query)
        return self._vector_store.search(query_vector, limit=limit)
