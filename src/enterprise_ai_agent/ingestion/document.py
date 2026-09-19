"""Unified document and metadata models."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(StrEnum):
    """Supported document formats."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class DocumentMetadata(BaseModel):
    """Provenance metadata attached to an ingested document."""

    model_config = ConfigDict(frozen=True)

    source: str
    file_name: str
    file_type: DocumentType
    file_size: int = Field(ge=0)


class Document(BaseModel):
    """A normalized document ready for later processing."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: DocumentMetadata
