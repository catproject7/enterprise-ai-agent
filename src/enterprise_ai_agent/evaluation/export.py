"""Deterministic serialization for evaluation reports."""

from .models import EvaluationReport


def serialize_evaluation_report(
    report: EvaluationReport,
    *,
    indent: int = 2,
) -> str:
    """Serialize an evaluation report as deterministic JSON."""

    return report.model_dump_json(indent=indent)
