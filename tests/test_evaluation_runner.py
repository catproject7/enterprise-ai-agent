"""Tests for batch RAG evaluation orchestration."""

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.evaluation import (
    AggregateMetrics,
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
    EvaluationResult,
    EvaluationRunner,
    Evaluator,
    GroundTruthChunk,
    RetrievalMetrics,
    aggregate_results,
)
from enterprise_ai_agent.rag import Answer, RAGPipeline, RAGResponse, RAGRun
from enterprise_ai_agent.vector_store import SearchResult


def _make_case(case_id: str) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=f"question-{case_id}",
        expected_answer=f"answer-{case_id}",
        relevant_chunks=(
            GroundTruthChunk(source=f"/{case_id}.txt", chunk_index=0, start_offset=0, end_offset=5),
        ),
    )


def _make_response(text: str) -> RAGResponse:
    return RAGResponse(answer=Answer(text=text))


def _make_run(question: str, *, result_id: str = "result-1") -> RAGRun:
    return RAGRun(
        question=question,
        retrieval_results=(),
        response=_make_response(f"answer-{question}"),
    )


def _make_evaluation_result(
    case_id: str,
    *,
    recall: float = 1.0,
    precision: float = 1.0,
    answer_score: float = 1.0,
    coverage: float = 1.0,
) -> EvaluationResult:
    return EvaluationResult(
        case_id=case_id,
        retrieval=RetrievalMetrics(
            recall_at_k=recall,
            precision_at_k=precision,
            matched_relevant=1,
            relevant_count=1,
        ),
        answer_correctness={"exact_match": answer_score == 1.0, "score": answer_score},
        citation_coverage={
            "matched_citations": 1 if coverage == 1.0 else 0,
            "expected_citations": 1,
            "coverage": coverage,
        },
    )


class FakePipeline(RAGPipeline):
    def __init__(
        self,
        runs: dict[str, RAGRun] | None = None,
        *,
        error: Exception | None = None,
        error_question: str | None = None,
        event_log: list[str] | None = None,
    ) -> None:
        self.runs = runs if runs is not None else {}
        self.error = error
        self.error_question = error_question
        self.calls: list[str] = []
        self.event_log = event_log

    def run_with_trace(self, question: str) -> RAGRun:
        self.calls.append(question)
        if self.event_log is not None:
            self.event_log.append(f"pipeline:{question}")
        if self.error is not None and (
            self.error_question is None or self.error_question == question
        ):
            raise self.error
        return self.runs.get(question, _make_run(question))


class FakeEvaluator(Evaluator):
    def __init__(
        self,
        results: dict[str, EvaluationResult],
        *,
        error: Exception | None = None,
        event_log: list[str] | None = None,
    ) -> None:
        self.results = results
        self.error = error
        self.calls: list[tuple[str, tuple[SearchResult, ...], RAGResponse, int]] = []
        self.event_log = event_log

    def evaluate(
        self,
        case: EvaluationCase,
        retrieval_results,
        response: RAGResponse,
        k: int,
    ) -> EvaluationResult:
        self.calls.append((case.case_id, tuple(retrieval_results), response, k))
        if self.event_log is not None:
            self.event_log.append(f"evaluator:{case.case_id}")
        if self.error is not None:
            raise self.error
        return self.results[case.case_id]


def test_runner_empty_dataset_returns_empty_report() -> None:
    pipeline = FakePipeline()
    evaluator = FakeEvaluator({})
    runner = EvaluationRunner(pipeline, evaluator)

    report = runner.run(EvaluationDataset(cases=()))

    assert report == EvaluationReport(total_cases=0, results=(), aggregate=None)
    assert pipeline.calls == []
    assert evaluator.calls == []


def test_runner_executes_single_case_with_original_question_and_dependencies() -> None:
    case = _make_case("case-1")
    rag_run = _make_run(case.question)
    pipeline = FakePipeline({case.question: rag_run})
    evaluator = FakeEvaluator({"case-1": _make_evaluation_result("case-1")})
    runner = EvaluationRunner(pipeline, evaluator, k=3)

    report = runner.run(EvaluationDataset(cases=(case,)))

    assert report.total_cases == 1
    assert report.aggregate is not None
    assert pipeline.calls == [case.question]
    assert evaluator.calls == [
        ("case-1", rag_run.retrieval_results, rag_run.response, 3),
    ]


def test_runner_preserves_case_order_and_aggregates_results() -> None:
    first = _make_case("case-1")
    second = _make_case("case-2")
    pipeline = FakePipeline(
        {
            first.question: _make_run(first.question),
            second.question: _make_run(second.question),
        }
    )
    evaluator = FakeEvaluator(
        {
            "case-1": _make_evaluation_result("case-1", recall=1.0, precision=1.0),
            "case-2": _make_evaluation_result(
                "case-2",
                recall=0.5,
                precision=0.5,
                answer_score=0.0,
                coverage=0.5,
            ),
        }
    )
    runner = EvaluationRunner(pipeline, evaluator)

    report = runner.run(EvaluationDataset(cases=(first, second)))

    assert report.total_cases == 2
    assert [result.case_id for result in report.results] == ["case-1", "case-2"]
    assert report.aggregate == AggregateMetrics(
        case_count=2,
        mean_recall_at_k=0.75,
        mean_precision_at_k=0.75,
        answer_accuracy=0.5,
        mean_citation_coverage=0.75,
    )


@pytest.mark.parametrize("k", [0, -1])
def test_runner_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        EvaluationRunner(FakePipeline(), FakeEvaluator({}), k=k)


def test_runner_is_deterministic() -> None:
    case = _make_case("case-1")
    pipeline = FakePipeline({case.question: _make_run(case.question)})
    evaluator = FakeEvaluator({"case-1": _make_evaluation_result("case-1")})
    runner = EvaluationRunner(pipeline, evaluator)

    assert runner.run(EvaluationDataset(cases=(case,))) == runner.run(
        EvaluationDataset(cases=(case,))
    )


def test_runner_does_not_modify_inputs() -> None:
    case = _make_case("case-1")
    dataset = EvaluationDataset(cases=(case,))
    rag_run = _make_run(case.question)
    before_case = case.model_dump()
    before_run = rag_run.model_dump()

    runner = EvaluationRunner(
        FakePipeline({case.question: rag_run}),
        FakeEvaluator({"case-1": _make_evaluation_result("case-1")}),
    )

    runner.run(dataset)

    assert case.model_dump() == before_case
    assert rag_run.model_dump() == before_run


def test_runner_propagates_pipeline_error_and_stops() -> None:
    first = _make_case("case-1")
    second = _make_case("case-2")
    error = RuntimeError("pipeline failed")
    pipeline = FakePipeline(error=error, error_question=second.question)
    evaluator = FakeEvaluator({"case-1": _make_evaluation_result("case-1")})
    runner = EvaluationRunner(pipeline, evaluator)

    with pytest.raises(RuntimeError, match="pipeline failed") as captured:
        runner.run(EvaluationDataset(cases=(first, second)))

    assert captured.value is error
    assert evaluator.calls[0][0] == "case-1"
    assert len(evaluator.calls) == 1


def test_runner_propagates_evaluator_error() -> None:
    case = _make_case("case-1")
    error = RuntimeError("evaluation failed")
    runner = EvaluationRunner(
        FakePipeline({case.question: _make_run(case.question)}),
        FakeEvaluator({}, error=error),
    )

    with pytest.raises(RuntimeError, match="evaluation failed") as captured:
        runner.run(EvaluationDataset(cases=(case,)))

    assert captured.value is error


def test_runner_calls_components_in_order() -> None:
    case = _make_case("case-1")
    event_log: list[str] = []
    runner = EvaluationRunner(
        FakePipeline({case.question: _make_run(case.question)}, event_log=event_log),
        FakeEvaluator({"case-1": _make_evaluation_result("case-1")}, event_log=event_log),
    )

    runner.run(EvaluationDataset(cases=(case,)))

    assert event_log == [f"pipeline:{case.question}", "evaluator:case-1"]


def test_aggregate_results_returns_none_for_empty_results() -> None:
    assert aggregate_results([]) is None


def test_evaluation_report_rejects_inconsistent_total_cases() -> None:
    with pytest.raises(ValidationError):
        EvaluationReport(total_cases=1, results=(), aggregate=None)


def test_evaluation_report_rejects_invalid_aggregate_state() -> None:
    aggregate = AggregateMetrics(
        case_count=1,
        mean_recall_at_k=1.0,
        mean_precision_at_k=1.0,
        answer_accuracy=1.0,
        mean_citation_coverage=1.0,
    )
    with pytest.raises(ValidationError):
        EvaluationReport(total_cases=0, results=(), aggregate=aggregate)
    with pytest.raises(ValidationError):
        EvaluationReport(
            total_cases=1,
            results=(_make_evaluation_result("case-1"),),
            aggregate=None,
        )


def test_evaluation_report_is_frozen() -> None:
    report = EvaluationReport(total_cases=0, results=(), aggregate=None)

    with pytest.raises(ValidationError):
        report.total_cases = 1
