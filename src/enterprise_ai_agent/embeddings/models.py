"""Embedding data models."""

from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator

from enterprise_ai_agent.chunking.models import Chunk


class EmbeddedChunk(BaseModel):
    """A source chunk paired with its embedding vector."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    vector: tuple[float, ...] = Field(min_length=1)

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, vector: tuple[float, ...]) -> tuple[float, ...]:
        """Reject vectors containing non-finite values."""

        if not all(isfinite(value) for value in vector):
            raise ValueError("embedding vector must contain only finite values")
        return vector
