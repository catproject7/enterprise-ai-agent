"""RAG execution trace models."""

from pydantic import BaseModel, ConfigDict, field_validator

from enterprise_ai_agent.vector_store import SearchResult

from .response import RAGResponse


class RAGRun(BaseModel):
    """A complete RAG execution trace for one question."""

    model_config = ConfigDict(frozen=True)

    question: str
    retrieval_results: tuple[SearchResult, ...]
    response: RAGResponse

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        """Reject empty questions."""

        if not question.strip():
            raise ValueError("question must not be empty")
        return question
