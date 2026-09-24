"""Tests for final RAG answer and citation data models."""

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.rag import Answer, Citation, RAGResponse


def _make_metadata(
    *,
    file_name: str = "sample.txt",
) -> DocumentMetadata:
    return DocumentMetadata(
        source=f"/documents/{file_name}",
        file_name=file_name,
        file_type=DocumentType.TEXT,
        file_size=100,
    )


def _make_citation(
    *,
    result_id: str = "result-1",
    file_name: str = "sample.txt",
    chunk_index: int = 0,
    start_offset: int = 0,
    end_offset: int = 10,
) -> Citation:
    return Citation(
        result_id=result_id,
        metadata=_make_metadata(file_name=file_name),
        chunk_index=chunk_index,
        start_offset=start_offset,
        end_offset=end_offset,
    )


def test_answer_accepts_normal_text() -> None:
    answer = Answer(text="The answer is alpha.")

    assert answer.text == "The answer is alpha."


@pytest.mark.parametrize("text", ["", " ", "\n\t", "\r\n"])
def test_answer_rejects_empty_text(text: str) -> None:
    with pytest.raises(ValidationError, match="answer text must not be empty"):
        Answer(text=text)


def test_answer_is_frozen() -> None:
    answer = Answer(text="answer")

    with pytest.raises(ValidationError):
        answer.text = "changed"


def test_citation_preserves_source_and_chunk_identity() -> None:
    citation = _make_citation(
        result_id="result-4",
        file_name="metadata.md",
        chunk_index=2,
        start_offset=5,
        end_offset=12,
    )

    assert citation.result_id == "result-4"
    assert citation.metadata.file_name == "metadata.md"
    assert citation.metadata.source == "/documents/metadata.md"
    assert citation.chunk_index == 2
    assert citation.start_offset == 5
    assert citation.end_offset == 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_index": -1},
        {"start_offset": -1},
        {"start_offset": 5, "end_offset": 5},
        {"start_offset": 5, "end_offset": 4},
    ],
)
def test_citation_rejects_invalid_identity_fields(kwargs: dict[str, int]) -> None:
    values = {
        "result_id": "result-1",
        "metadata": _make_metadata(),
        "chunk_index": 0,
        "start_offset": 0,
        "end_offset": 10,
    }
    values.update(kwargs)

    with pytest.raises(ValidationError):
        Citation(**values)


@pytest.mark.parametrize("result_id", ["", " ", "\n\t"])
def test_citation_rejects_empty_result_id(result_id: str) -> None:
    with pytest.raises(ValidationError, match="result_id must not be empty"):
        _make_citation(result_id=result_id)


def test_citation_is_frozen() -> None:
    citation = _make_citation()

    with pytest.raises(ValidationError):
        citation.chunk_index = 2


def test_rag_response_supports_answer_and_multiple_citations() -> None:
    answer = Answer(text="answer")
    citations = (
        _make_citation(result_id="first", chunk_index=0),
        _make_citation(result_id="second", chunk_index=1),
    )

    response = RAGResponse(answer=answer, citations=citations)

    assert response.answer == answer
    assert response.citations == citations
    assert [citation.result_id for citation in response.citations] == ["first", "second"]


def test_rag_response_allows_empty_citations() -> None:
    response = RAGResponse(answer=Answer(text="answer"))

    assert response.citations == ()


def test_rag_response_is_frozen() -> None:
    response = RAGResponse(answer=Answer(text="answer"))

    with pytest.raises(ValidationError):
        response.answer = Answer(text="changed")


def test_rag_response_does_not_modify_inputs() -> None:
    answer = Answer(text="answer")
    citation = _make_citation()
    before_answer = answer.model_dump()
    before_citation = citation.model_dump()

    RAGResponse(answer=answer, citations=(citation,))

    assert answer.model_dump() == before_answer
    assert citation.model_dump() == before_citation


def test_rag_response_is_deterministic() -> None:
    response = RAGResponse(
        answer=Answer(text="answer"),
        citations=(_make_citation(),),
    )

    assert response == RAGResponse(
        answer=Answer(text="answer"),
        citations=(_make_citation(),),
    )
