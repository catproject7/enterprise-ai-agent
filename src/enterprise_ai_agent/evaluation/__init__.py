"""RAG evaluation primitives."""

from .evaluator import Evaluator
from .metrics import (
    calculate_answer_correctness,
    calculate_citation_coverage,
    calculate_retrieval_metrics,
)
from .models import (
    AnswerCorrectness,
    CitationCoverage,
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    GroundTruthChunk,
    RetrievalMetrics,
)

__all__ = [
    "AnswerCorrectness",
    "CitationCoverage",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationResult",
    "Evaluator",
    "GroundTruthChunk",
    "RetrievalMetrics",
    "calculate_answer_correctness",
    "calculate_citation_coverage",
    "calculate_retrieval_metrics",
]
