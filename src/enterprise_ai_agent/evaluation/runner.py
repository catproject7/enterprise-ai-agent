"""Batch evaluation orchestration."""

from enterprise_ai_agent.rag import RAGPipeline

from .evaluator import Evaluator
from .metrics import aggregate_results
from .models import EvaluationDataset, EvaluationReport


class EvaluationRunner:
    """Run an evaluation dataset through a RAG pipeline and evaluator."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        evaluator: Evaluator,
        *,
        k: int = 5,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        self._pipeline = pipeline
        self._evaluator = evaluator
        self._k = k

    def run(self, dataset: EvaluationDataset) -> EvaluationReport:
        """Evaluate all dataset cases in order."""

        results = []
        for case in dataset.cases:
            rag_run = self._pipeline.run_with_trace(case.question)
            results.append(
                self._evaluator.evaluate(
                    case,
                    rag_run.retrieval_results,
                    rag_run.response,
                    self._k,
                )
            )

        return EvaluationReport(
            total_cases=len(results),
            results=tuple(results),
            aggregate=aggregate_results(results),
        )
