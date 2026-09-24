"""Deterministic context construction for retrieval results."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from enterprise_ai_agent.ingestion.document import DocumentMetadata
from enterprise_ai_agent.vector_store.models import SearchResult


class ContextBuilderConfig(BaseModel):
    """Limits for deterministic RAG context construction."""

    model_config = ConfigDict(frozen=True)

    max_context_chunks: int = Field(default=5, gt=0)
    max_context_chars: int = Field(default=4000, gt=0)


class ContextSource(BaseModel):
    """Source information retained for a selected retrieval result."""

    model_config = ConfigDict(frozen=True)

    result_id: str
    score: float
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int


class Context(BaseModel):
    """Deterministic text context and its ordered source information."""

    model_config = ConfigDict(frozen=True)

    text: str
    sources: tuple[ContextSource, ...]


class ContextBuilder:
    """Convert ordered retrieval results into bounded context."""

    def __init__(self, config: ContextBuilderConfig | None = None) -> None:
        self._config = config if config is not None else ContextBuilderConfig()

    def build(self, results: Sequence[SearchResult]) -> Context:
        """Build context while preserving retrieval order and provenance."""

        text_parts: list[str] = []
        sources: list[ContextSource] = []
        current_length = 0

        for result in results:
            if len(sources) >= self._config.max_context_chunks:
                break

            remaining = self._config.max_context_chars - current_length
            if remaining <= 0:
                break

            separator = "\n\n" if text_parts else ""
            if len(separator) >= remaining:
                break

            available = remaining - len(separator)
            content = result.chunk.content[:available]
            if not content:
                break

            text_parts.append(content)
            sources.append(
                ContextSource(
                    result_id=result.id,
                    score=result.score,
                    metadata=result.chunk.metadata,
                    chunk_index=result.chunk.chunk_index,
                    start_offset=result.chunk.start_offset,
                    end_offset=result.chunk.end_offset,
                )
            )
            current_length += len(separator) + len(content)

            if len(content) < len(result.chunk.content):
                break

        return Context(text="\n\n".join(text_parts), sources=tuple(sources))
