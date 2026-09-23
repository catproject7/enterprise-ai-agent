"""Vector store result models."""

from math import isfinite

from pydantic import BaseModel, ConfigDict, field_validator

from enterprise_ai_agent.chunking.models import Chunk


class SearchResult(BaseModel):
    """A similarity search result containing its source chunk."""

    model_config = ConfigDict(frozen=True)

    id: str
    score: float
    chunk: Chunk

    @field_validator("score")
    @classmethod
    def validate_score(cls, score: float) -> float:
        """Reject non-finite similarity scores."""

        if not isfinite(score):
            raise ValueError("similarity score must be finite")
        return score
