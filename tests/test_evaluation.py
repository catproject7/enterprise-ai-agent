"""Tests for the RAG evaluation foundation."""

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.chunking import Chunk
from enterprise_ai_agent.evaluation import (
    AnswerCorrectness,
    CitationCoverage,
    EvaluationCase,
    EvaluationDataset,
    Evaluator,
    GroundTruthChunk,
    RetrievalMetrics,
    calculate_answer_correctness,
    calculate_citation_coverage,
    calculate_retrieval_metrics,
)
from enterprise_ai_agent.ingestion import (
    DocumentMetadata,
    DocumentType,
)
from enterprise_ai_agent.rag import Answer, Citation, RAGResponse
from enterprise_ai_agent.vector_store import SearchResult


def _make_metadata(source: str = "/documents/sample.txt") -> DocumentMetadata:
    return DocumentMetadata(
        source=source,
        file_name=source.rsplit("/", 1)[-1],
        file_type=DocumentType.TEXT,
        file_size=100,
    )


def _make_ground_truth(
    *,
    source: str = "/documents/sample.txt",
    chunk_index: int = 0,
    start_offset: int = 0,
    end_offset: int = 5,
) -> GroundTruthChunk:
    return GroundTruthChunk(
        source=source,
        chunk_index=chunk_index,
        start_offset=start_offset,
        end_offset=end_offset,
    )


def _make_result(
    ground_truth: GroundTruthChunk,
    *,
    result_id: str = "result-1",
    score: float = 0.9,
) -> SearchResult:
    start = ground_truth.start_offset
    end = ground_truth.end_offset
    return SearchResult(
        id=result_id,
        score=score,
        chunk=Chunk(
            content="x" * (end - start),
            metadata=_make_metadata(ground_truth.source),
            chunk_index=ground_truth.chunk_index,
            start_offset=start,
            end_offset=end,
        ),
    )


def _make_citation(ground_truth: GroundTruthChunk) -> Citation:
    return Citation(
        result_id="citation",
        metadata=_make_metadata(ground_truth.source),
        chunk_index=ground_truth.chunk_index,
        start_offset=ground_truth.start_offset,
        end_offset=ground_truth.end_offset,
    )


def _make_case(
    *,
    relevant_chunks: tuple[GroundTruthChunk, ...] | None = None,
    expected_answer: str = "answer",
) -> EvaluationCase:
    return EvaluationCase(
        case_id="case-1",
        question="What is the answer?",
        expected_answer=expected_answer,
        relevant_chunks=relevant_chunks or (_make_ground_truth(),),
    )


def test_ground_truth_chunk_accepts_valid_identity() -> None:
    ground_truth = _make_ground_truth(chunk_index=2, start_offset=3, end_offset=8)

    assert ground_truth.source == "/documents/sample.txt"
    assert ground_truth.chunk_index == 2
    assert ground_truth.start_offset == 3
    assert ground_truth.end_offset == 8


@pytest.mark.parametrize("source", ["", " ", "\n\t"])
def test_ground_truth_chunk_rejects_empty_source(source: str) -> None:
    with pytest.raises(ValidationError):
        _make_ground_truth(source=source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_index": -1},
        {"start_offset": -1},
        {"start_offset": 5, "end_offset": 5},
        {"start_offset": 5, "end_offset": 4},
    ],
)
def test_ground_truth_chunk_rejects_invalid_identity(kwargs: dict[str, int]) -> None:
    values: dict[str, object] = {
        "source": "/documents/sample.txt",
        "chunk_index": 0,
        "start_offset": 0,
        "end_offset": 5,
    }
    values.update(kwargs)

    with pytest.raises(ValidationError):
        GroundTruthChunk(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", ""),
        ("question", " "),
        ("expected_answer", "\n\t"),
    ],
)
def test_evaluation_case_rejects_empty_text(field: str, value: str) -> None:
    values: dict[str, object] = {
        "case_id": "case-1",
        "question": "question",
        "expected_answer": "answer",
        "relevant_chunks": (_make_ground_truth(),),
    }
    values[field] = value

    with pytest.raises(ValidationError):
        EvaluationCase(**values)


def test_evaluation_case_requires_relevant_chunks() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            case_id="case-1",
            question="question",
            expected_answer="answer",
            relevant_chunks=(),
        )


def test_evaluation_dataset_wraps_cases() -> None:
    dataset = EvaluationDataset(cases=(_make_case(),))

    assert len(dataset.cases) == 1


def test_metric_models_validate_types_and_ranges() -> None:
    with pytest.raises(ValidationError):
        RetrievalMetrics(
            recall_at_k=1.1,
            precision_at_k=0.0,
            matched_relevant=0,
            relevant_count=1,
        )
    with pytest.raises(ValidationError):
        RetrievalMetrics(
            recall_at_k=0.0,
            precision_at_k=0.0,
            matched_relevant=-1,
            relevant_count=1,
        )
    with pytest.raises(ValidationError):
        AnswerCorrectness(exact_match=True, score=0.5)
    with pytest.raises(ValidationError):
        CitationCoverage(matched_citations=0, expected_citations=1, coverage=2.0)


def test_evaluation_models_are_frozen() -> None:
    case = _make_case()

    with pytest.raises(ValidationError):
        case.case_id = "changed"


def test_retrieval_metrics_empty_results() -> None:
    relevant = (_make_ground_truth(),)

    metrics = calculate_retrieval_metrics(relevant, [], k=5)

    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.matched_relevant == 0


def test_retrieval_metrics_use_k_as_precision_denominator() -> None:
    relevant = (
        _make_ground_truth(chunk_index=0, start_offset=0, end_offset=5),
        _make_ground_truth(chunk_index=1, start_offset=5, end_offset=10),
    )
    results = [_make_result(chunk) for chunk in relevant]

    metrics = calculate_retrieval_metrics(relevant, results, k=5)

    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 0.4


def test_retrieval_metrics_handle_exact_k() -> None:
    relevant = (
        _make_ground_truth(chunk_index=0, start_offset=0, end_offset=5),
        _make_ground_truth(chunk_index=1, start_offset=5, end_offset=10),
    )
    results = [_make_result(chunk) for chunk in relevant]

    metrics = calculate_retrieval_metrics(relevant, results, k=2)

    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 1.0


def test_retrieval_metrics_ignore_results_beyond_k() -> None:
    first = _make_ground_truth(chunk_index=0, start_offset=0, end_offset=5)
    second = _make_ground_truth(chunk_index=1, start_offset=5, end_offset=10)
    results = [_make_result(first), _make_result(second, result_id="result-2")]

    metrics = calculate_retrieval_metrics((first, second), results, k=1)

    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 1.0


def test_retrieval_metrics_do_not_double_count_duplicates() -> None:
    relevant = (_make_ground_truth(),)
    result = _make_result(relevant[0])

    metrics = calculate_retrieval_metrics(relevant, [result, result], k=2)

    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 0.5
    assert metrics.matched_relevant == 1


def test_retrieval_metrics_report_missing_relevant_chunk() -> None:
    relevant = (
        _make_ground_truth(chunk_index=0, start_offset=0, end_offset=5),
        _make_ground_truth(chunk_index=1, start_offset=5, end_offset=10),
    )
    metrics = calculate_retrieval_metrics(relevant, [_make_result(relevant[0])], k=2)

    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 0.5


@pytest.mark.parametrize("k", [0, -1])
def test_retrieval_metrics_reject_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        calculate_retrieval_metrics((_make_ground_truth(),), [], k=k)


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        ("Hello world", "Hello world"),
        ("Hello world", "hello world"),
        ("  Hello   world  ", "hello world"),
        ("HELLO\nWORLD", "hello world"),
    ],
)
def test_answer_correctness_normalizes_exact_match(expected: str, actual: str) -> None:
    result = calculate_answer_correctness(expected, actual)

    assert result.exact_match is True
    assert result.score == 1.0


def test_answer_correctness_rejects_mismatch() -> None:
    result = calculate_answer_correctness("expected", "actual")

    assert result.exact_match is False
    assert result.score == 0.0


def test_answer_model_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Answer(text="")


def test_citation_coverage_full_match() -> None:
    relevant = (_make_ground_truth(),)
    response = RAGResponse(
        answer=Answer(text="answer"),
        citations=(_make_citation(relevant[0]),),
    )

    result = calculate_citation_coverage(relevant, response)

    assert result.matched_citations == 1
    assert result.expected_citations == 1
    assert result.coverage == 1.0


def test_citation_coverage_partial_match() -> None:
    relevant = (
        _make_ground_truth(chunk_index=0, start_offset=0, end_offset=5),
        _make_ground_truth(chunk_index=1, start_offset=5, end_offset=10),
    )
    response = RAGResponse(
        answer=Answer(text="answer"),
        citations=(_make_citation(relevant[1]),),
    )

    result = calculate_citation_coverage(relevant, response)

    assert result.matched_citations == 1
    assert result.expected_citations == 2
    assert result.coverage == 0.5


def test_citation_coverage_without_citations() -> None:
    result = calculate_citation_coverage(
        (_make_ground_truth(),),
        RAGResponse(answer=Answer(text="answer")),
    )

    assert result.coverage == 0.0


def test_citation_coverage_ignores_order_and_duplicates() -> None:
    relevant = (
        _make_ground_truth(chunk_index=0, start_offset=0, end_offset=5),
        _make_ground_truth(chunk_index=1, start_offset=5, end_offset=10),
    )
    response = RAGResponse(
        answer=Answer(text="answer"),
        citations=(
            _make_citation(relevant[1]),
            _make_citation(relevant[0]),
            _make_citation(relevant[0]),
        ),
    )

    result = calculate_citation_coverage(relevant, response)

    assert result.matched_citations == 2
    assert result.coverage == 1.0


def test_citation_coverage_ignores_extra_citations() -> None:
    relevant = (_make_ground_truth(),)
    extra = _make_ground_truth(chunk_index=9, start_offset=50, end_offset=55)
    response = RAGResponse(
        answer=Answer(text="answer"),
        citations=(_make_citation(relevant[0]), _make_citation(extra)),
    )

    result = calculate_citation_coverage(relevant, response)

    assert result.coverage == 1.0


def test_evaluator_orchestrates_all_metrics() -> None:
    relevant = (_make_ground_truth(),)
    case = _make_case(relevant_chunks=relevant, expected_answer="  Expected  Answer ")
    results = [_make_result(relevant[0])]
    response = RAGResponse(
        answer=Answer(text="expected answer"),
        citations=(_make_citation(relevant[0]),),
    )

    result = Evaluator().evaluate(case, results, response, 1)

    assert result.case_id == "case-1"
    assert result.retrieval.recall_at_k == 1.0
    assert result.retrieval.precision_at_k == 1.0
    assert result.answer_correctness.exact_match is True
    assert result.citation_coverage.coverage == 1.0


def test_evaluator_is_deterministic() -> None:
    relevant = (_make_ground_truth(),)
    case = _make_case(relevant_chunks=relevant)
    results = [_make_result(relevant[0])]
    response = RAGResponse(
        answer=Answer(text="answer"),
        citations=(_make_citation(relevant[0]),),
    )
    evaluator = Evaluator()

    assert evaluator.evaluate(case, results, response, 1) == evaluator.evaluate(
        case,
        results,
        response,
        1,
    )


def test_evaluator_does_not_modify_inputs() -> None:
    relevant = (_make_ground_truth(),)
    case = _make_case(relevant_chunks=relevant)
    results = [_make_result(relevant[0])]
    response = RAGResponse(
        answer=Answer(text="answer"),
        citations=(_make_citation(relevant[0]),),
    )
    before = (case.model_dump(), results[0].model_dump(), response.model_dump())

    Evaluator().evaluate(case, results, response, 1)

    after = (case.model_dump(), results[0].model_dump(), response.model_dump())
    assert after == before


def test_evaluator_propagates_metric_errors() -> None:
    relevant = (_make_ground_truth(),)
    case = _make_case(relevant_chunks=relevant)
    response = RAGResponse(answer=Answer(text="answer"))

    with pytest.raises(ValueError, match="k"):
        Evaluator().evaluate(case, [], response, 0)
