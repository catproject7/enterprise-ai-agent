"""Evaluation orchestration for completed RAG outputs."""

from collections.abc import Sequence

from enterprise_ai_agent.rag import RAGResponse
from enterprise_ai_agent.vector_store import SearchResult

from .metrics import (
    calculate_answer_correctness,
    calculate_citation_coverage,
    calculate_retrieval_metrics,
)
from .models import EvaluationCase, EvaluationResult


class Evaluator:
    """Evaluate completed retrieval and RAG outputs for one case."""

    def evaluate(
        self,
        case: EvaluationCase,
        retrieval_results: Sequence[SearchResult],
        response: RAGResponse,
        k: int,
    ) -> EvaluationResult:
        """Calculate all first-version metrics for one case."""

        return EvaluationResult(
            case_id=case.case_id,
            retrieval=calculate_retrieval_metrics(
                case.relevant_chunks,
                retrieval_results,
                k=k,
            ),
            answer_correctness=calculate_answer_correctness(
                case.expected_answer,
                response.answer.text,
            ),
            citation_coverage=calculate_citation_coverage(
                case.relevant_chunks,
                response,
            ),
        )
