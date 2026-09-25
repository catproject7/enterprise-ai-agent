"""Tests for deterministic evaluation report export."""

import json

from enterprise_ai_agent.evaluation import (
    AggregateMetrics,
    AnswerCorrectness,
    CitationCoverage,
    EvaluationReport,
    EvaluationResult,
    RetrievalMetrics,
)
from enterprise_ai_agent.evaluation.export import serialize_evaluation_report


def _make_result() -> EvaluationResult:
    return EvaluationResult(
        case_id="case-1",
        retrieval=RetrievalMetrics(
            recall_at_k=1.0,
            precision_at_k=0.5,
            matched_relevant=1,
            relevant_count=1,
        ),
        answer_correctness=AnswerCorrectness(exact_match=True, score=1.0),
        citation_coverage=CitationCoverage(
            matched_citations=1,
            expected_citations=1,
            coverage=1.0,
        ),
    )


def _make_report() -> EvaluationReport:
    return EvaluationReport(
        total_cases=1,
        results=(_make_result(),),
        aggregate=AggregateMetrics(
            case_count=1,
            mean_recall_at_k=1.0,
            mean_precision_at_k=0.5,
            answer_accuracy=1.0,
            mean_citation_coverage=1.0,
        ),
    )


def test_serialize_evaluation_report_round_trips() -> None:
    report = _make_report()

    serialized = serialize_evaluation_report(report)

    assert EvaluationReport.model_validate_json(serialized) == report


def test_serialize_empty_evaluation_report() -> None:
    report = EvaluationReport(total_cases=0, results=(), aggregate=None)

    serialized = serialize_evaluation_report(report)

    assert EvaluationReport.model_validate_json(serialized) == report
    assert json.loads(serialized)["aggregate"] is None


def test_serialize_evaluation_report_is_deterministic() -> None:
    report = _make_report()

    assert serialize_evaluation_report(report) == serialize_evaluation_report(report)


def test_serialize_evaluation_report_respects_indent() -> None:
    report = _make_report()

    compact = serialize_evaluation_report(report, indent=0)
    pretty = serialize_evaluation_report(report, indent=2)

    assert "\n" in compact
    assert '\n  "total_cases"' in pretty
    assert compact != pretty
