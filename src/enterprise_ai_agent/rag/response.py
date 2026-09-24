"""Final RAG answer and citation data models."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from enterprise_ai_agent.ingestion.document import DocumentMetadata


class Answer(BaseModel):
    """A natural-language answer produced for a RAG response."""

    model_config = ConfigDict(frozen=True)

    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, text: str) -> str:
        """Reject empty answer text."""

        if not text.strip():
            raise ValueError("answer text must not be empty")
        return text


class Citation(BaseModel):
    """A stable reference to the source chunk supporting an answer."""

    model_config = ConfigDict(frozen=True)

    result_id: str
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int

    @field_validator("result_id")
    @classmethod
    def validate_result_id(cls, result_id: str) -> str:
        """Reject empty result identifiers."""

        if not result_id.strip():
            raise ValueError("result_id must not be empty")
        return result_id

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Require a non-empty source offset range."""

        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class RAGResponse(BaseModel):
    """A final answer with its ordered supporting citations."""

    model_config = ConfigDict(frozen=True)

    answer: Answer
    citations: tuple[Citation, ...] = ()
