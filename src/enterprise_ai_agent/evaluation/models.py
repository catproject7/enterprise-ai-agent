"""Pydantic models for RAG evaluation."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GroundTruthChunk(BaseModel):
    """Stable identity for an expected relevant source chunk."""

    model_config = ConfigDict(frozen=True)

    source: str
    chunk_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int

    @field_validator("source")
    @classmethod
    def validate_source(cls, source: str) -> str:
        """Reject empty source identities."""

        if not source.strip():
            raise ValueError("source must not be empty")
        return source

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Require a non-empty source offset range."""

        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class EvaluationCase(BaseModel):
    """One question and its expected retrieval and answer data."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    expected_answer: str
    relevant_chunks: tuple[GroundTruthChunk, ...] = Field(min_length=1)

    @field_validator("case_id", "question", "expected_answer")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject empty case, question, and expected answer values."""

        if not value.strip():
            raise ValueError("evaluation text fields must not be empty")
        return value


class EvaluationDatasetMetadata(BaseModel):
    """Portable metadata for a versioned evaluation dataset."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str | None = None

    @field_validator("name", "version")
    @classmethod
    def validate_non_empty_metadata(cls, value: str) -> str:
        """Reject empty dataset metadata values."""

        if not value.strip():
            raise ValueError("dataset metadata fields must not be empty")
        return value


class EvaluationDataset(BaseModel):
    """A collection of evaluation cases."""

    model_config = ConfigDict(frozen=True)

    cases: tuple[EvaluationCase, ...]
    metadata: EvaluationDatasetMetadata | None = None


class RetrievalMetrics(BaseModel):
    """Recall and precision metrics for one retrieval result set."""

    model_config = ConfigDict(frozen=True)

    recall_at_k: float = Field(ge=0.0, le=1.0)
    precision_at_k: float = Field(ge=0.0, le=1.0)
    matched_relevant: int = Field(ge=0)
    relevant_count: int = Field(ge=0)


class AnswerCorrectness(BaseModel):
    """Normalized exact-match result for a generated answer."""

    model_config = ConfigDict(frozen=True)

    exact_match: bool
    score: float

    @field_validator("score")
    @classmethod
    def validate_score(cls, score: float) -> float:
        """Require a binary correctness score."""

        if score not in (0.0, 1.0):
            raise ValueError("answer correctness score must be 0.0 or 1.0")
        return score


class CitationCoverage(BaseModel):
    """Coverage of expected relevant chunks by response citations."""

    model_config = ConfigDict(frozen=True)

    matched_citations: int = Field(ge=0)
    expected_citations: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    """Evaluation metrics for one case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    retrieval: RetrievalMetrics
    answer_correctness: AnswerCorrectness
    citation_coverage: CitationCoverage


class AggregateMetrics(BaseModel):
    """Aggregate metrics across an evaluation dataset."""

    model_config = ConfigDict(frozen=True)

    case_count: int = Field(gt=0)
    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    answer_accuracy: float = Field(ge=0.0, le=1.0)
    mean_citation_coverage: float = Field(ge=0.0, le=1.0)


class EvaluationReport(BaseModel):
    """Aggregate report for an evaluated dataset."""

    model_config = ConfigDict(frozen=True)

    total_cases: int = Field(ge=0)
    results: tuple[EvaluationResult, ...]
    aggregate: AggregateMetrics | None = None

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        """Keep total case count and aggregate state consistent."""

        if self.total_cases != len(self.results):
            raise ValueError("total_cases must match results length")
        if not self.results and self.aggregate is not None:
            raise ValueError("empty evaluations cannot have aggregate metrics")
        if self.results and self.aggregate is None:
            raise ValueError("non-empty evaluations require aggregate metrics")
        return self
