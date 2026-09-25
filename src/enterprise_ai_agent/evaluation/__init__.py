"""RAG evaluation primitives."""

from .evaluator import Evaluator
from .metrics import (
    aggregate_results,
    calculate_answer_correctness,
    calculate_citation_coverage,
    calculate_retrieval_metrics,
)
from .models import (
    AggregateMetrics,
    AnswerCorrectness,
    CitationCoverage,
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
    EvaluationResult,
    GroundTruthChunk,
    RetrievalMetrics,
)
from .runner import EvaluationRunner

__all__ = [
    "AggregateMetrics",
    "AnswerCorrectness",
    "CitationCoverage",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationRunner",
    "Evaluator",
    "GroundTruthChunk",
    "RetrievalMetrics",
    "aggregate_results",
    "calculate_answer_correctness",
    "calculate_citation_coverage",
    "calculate_retrieval_metrics",
]
