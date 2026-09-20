"""Chunk data models and chunking configuration."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from enterprise_ai_agent.ingestion.document import DocumentMetadata


class ChunkingConfig(BaseModel):
    """Configuration for fixed-size document chunking."""

    model_config = ConfigDict(frozen=True)

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        """Ensure overlapping windows always move forward."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class Chunk(BaseModel):
    """A contiguous slice of an ingested document."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Validate invariants that do not require the source document."""

        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if len(self.content) != self.end_offset - self.start_offset:
            raise ValueError("content length must match the chunk offset range")
        return self
