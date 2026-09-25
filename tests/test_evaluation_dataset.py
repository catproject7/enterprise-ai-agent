"""Tests for evaluation dataset metadata and JSON loading."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationDatasetMetadata,
    GroundTruthChunk,
    load_evaluation_dataset,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "evaluation" / "smoke_dataset.json"


def _case_payload(case_id: str = "case-1") -> dict[str, object]:
    return {
        "case_id": case_id,
        "question": f"question-{case_id}",
        "expected_answer": f"answer-{case_id}",
        "relevant_chunks": [
            {
                "source": "sample.txt",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 10,
            }
        ],
    }


def _write_dataset(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_valid_dataset_fixture() -> None:
    dataset = load_evaluation_dataset(FIXTURE_PATH)

    assert dataset.metadata == EvaluationDatasetMetadata(
        name="rag-smoke",
        version="1",
        description="Small deterministic evaluation dataset",
    )
    assert [case.case_id for case in dataset.cases] == ["case-1", "case-2"]
    assert [case.question for case in dataset.cases] == [
        "What is alpha?",
        "What is beta?",
    ]


def test_dataset_metadata_description_is_optional() -> None:
    metadata = EvaluationDatasetMetadata(name="name", version="version")

    assert metadata.description is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("name", " "),
        ("version", ""),
        ("version", "\n\t"),
    ],
)
def test_dataset_metadata_rejects_empty_fields(field: str, value: str) -> None:
    values = {"name": "name", "version": "version", field: value}

    with pytest.raises(ValidationError):
        EvaluationDatasetMetadata(**values)


def test_dataset_metadata_is_frozen() -> None:
    metadata = EvaluationDatasetMetadata(name="name", version="1")

    with pytest.raises(ValidationError):
        metadata.name = "changed"


def test_load_dataset_without_metadata(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, {"cases": [_case_payload()]})

    dataset = load_evaluation_dataset(path)

    assert dataset.metadata is None
    assert len(dataset.cases) == 1


def test_load_dataset_preserves_case_and_ground_truth_order(tmp_path: Path) -> None:
    payload = {
        "cases": [
            {
                **_case_payload("case-1"),
                "relevant_chunks": [
                    {"source": "first.txt", "chunk_index": 2, "start_offset": 20, "end_offset": 30},
                    {
                        "source": "second.txt",
                        "chunk_index": 1,
                        "start_offset": 10,
                        "end_offset": 20,
                    },
                ],
            },
            _case_payload("case-2"),
        ]
    }
    path = _write_dataset(tmp_path, payload)

    dataset = load_evaluation_dataset(path)

    assert [case.case_id for case in dataset.cases] == ["case-1", "case-2"]
    assert [chunk.source for chunk in dataset.cases[0].relevant_chunks] == [
        "first.txt",
        "second.txt",
    ]
    assert dataset.cases[0].relevant_chunks[0].start_offset == 20
    assert dataset.cases[0].relevant_chunks[0].end_offset == 30


def test_dataset_constructor_remains_backward_compatible() -> None:
    case = EvaluationCase(
        case_id="case-1",
        question="question",
        expected_answer="answer",
        relevant_chunks=(
            GroundTruthChunk(source="sample.txt", chunk_index=0, start_offset=0, end_offset=10),
        ),
    )

    dataset = EvaluationDataset(cases=(case,))

    assert dataset.cases == (case,)
    assert dataset.metadata is None


def test_evaluation_dataset_is_frozen() -> None:
    dataset = EvaluationDataset(cases=())

    with pytest.raises(ValidationError):
        dataset.cases = ()


def test_load_dataset_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_evaluation_dataset("missing-dataset.json")


def test_load_dataset_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_evaluation_dataset(path)


def test_load_dataset_missing_required_field(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path, {"cases": [{"case_id": "case-1"}]})

    with pytest.raises(ValidationError):
        load_evaluation_dataset(path)


def test_load_dataset_invalid_offsets(tmp_path: Path) -> None:
    case = _case_payload()
    case["relevant_chunks"] = [
        {"source": "sample.txt", "chunk_index": 0, "start_offset": 10, "end_offset": 10}
    ]
    path = _write_dataset(tmp_path, {"cases": [case]})

    with pytest.raises(ValidationError):
        load_evaluation_dataset(path)


@pytest.mark.parametrize("field", ["question", "expected_answer"])
def test_load_dataset_rejects_empty_case_fields(tmp_path: Path, field: str) -> None:
    case = _case_payload()
    case[field] = " "
    path = _write_dataset(tmp_path, {"cases": [case]})

    with pytest.raises(ValidationError):
        load_evaluation_dataset(path)


def test_load_dataset_is_deterministic() -> None:
    first = load_evaluation_dataset(FIXTURE_PATH)
    second = load_evaluation_dataset(FIXTURE_PATH)

    assert first == second
