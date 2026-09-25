"""Evaluation dataset JSON loading."""

import json
from pathlib import Path

from .models import EvaluationDataset


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    """Load and validate an evaluation dataset from a UTF-8 JSON file."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(payload)
