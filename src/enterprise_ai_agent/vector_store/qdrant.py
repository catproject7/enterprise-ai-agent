"""Qdrant vector store implementation."""

from collections.abc import Sequence
from math import isfinite
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from enterprise_ai_agent.chunking.models import Chunk
from enterprise_ai_agent.embeddings.models import EmbeddedChunk

from .base import VectorStore
from .models import SearchResult


def _validate_dimension(dimension: int) -> None:
    if dimension <= 0:
        raise ValueError("vector dimension must be greater than zero")


def _validate_vector(
    vector: Sequence[float],
    *,
    dimension: int,
    name: str,
) -> None:
    if len(vector) != dimension:
        raise ValueError(f"{name} dimension mismatch: expected {dimension}, got {len(vector)}")
    if not all(isfinite(float(value)) for value in vector):
        raise ValueError(f"{name} must contain only finite values")


def _chunk_id(chunk: Chunk) -> str:
    return str(uuid5(NAMESPACE_URL, chunk.model_dump_json()))


class QdrantVectorStore(VectorStore):
    """Qdrant-backed storage for embedded document chunks."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = "document_chunks",
    ) -> None:
        if not collection_name:
            raise ValueError("Qdrant collection name must not be empty")
        self._client = client
        self._collection_name = collection_name

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection when missing and validate an existing dimension."""

        _validate_dimension(dimension)

        if self._client.collection_exists(self._collection_name):
            existing_dimension = self._collection_dimension()
            if existing_dimension != dimension:
                raise ValueError(
                    "Qdrant collection dimension mismatch: "
                    f"expected {dimension}, got {existing_dimension}"
                )
            return

        self._create_collection(dimension)

    def recreate_collection(self, dimension: int) -> None:
        """Replace the collection, removing all existing points."""

        _validate_dimension(dimension)

        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)
        self._create_collection(dimension)

    def upsert(self, embedded_chunks: Sequence[EmbeddedChunk]) -> int:
        """Insert or replace embedded chunks using deterministic point IDs."""

        if not embedded_chunks:
            return 0

        dimension = self._collection_dimension()
        points: list[PointStruct] = []

        for embedded_chunk in embedded_chunks:
            _validate_vector(
                embedded_chunk.vector,
                dimension=dimension,
                name="embedding vector",
            )
            points.append(
                PointStruct(
                    id=_chunk_id(embedded_chunk.chunk),
                    vector=[float(value) for value in embedded_chunk.vector],
                    payload={"chunk": embedded_chunk.chunk.model_dump(mode="json")},
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Return the nearest chunks for a query vector."""

        if limit <= 0:
            raise ValueError("search limit must be greater than zero")

        dimension = self._collection_dimension()
        _validate_vector(query_vector, dimension=dimension, name="query vector")

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=[float(value) for value in query_vector],
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            raw_chunk = payload.get("chunk")
            if not isinstance(raw_chunk, dict):
                raise ValueError("Qdrant point payload does not contain a valid chunk")

            results.append(
                SearchResult(
                    id=str(point.id),
                    score=float(point.score),
                    chunk=Chunk.model_validate(raw_chunk),
                )
            )

        return results

    def _create_collection(self, dimension: int) -> None:
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def _collection_dimension(self) -> int:
        if not self._client.collection_exists(self._collection_name):
            raise ValueError(f"Qdrant collection does not exist: {self._collection_name}")

        vectors_config = self._client.get_collection(self._collection_name).config.params.vectors
        if not isinstance(vectors_config, VectorParams):
            raise ValueError("named Qdrant vectors are not supported")
        return vectors_config.size
