"""Tests for deterministic RAG context construction."""

import pytest
from pydantic import ValidationError

from enterprise_ai_agent.chunking import Chunk
from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.rag import ContextBuilder, ContextBuilderConfig
from enterprise_ai_agent.vector_store import SearchResult


def _make_result(
    content: str,
    *,
    result_id: str,
    score: float,
    chunk_index: int = 0,
    file_name: str = "sample.txt",
    start_offset: int = 0,
) -> SearchResult:
    return SearchResult(
        id=result_id,
        score=score,
        chunk=Chunk(
            content=content,
            metadata=DocumentMetadata(
                source=f"/documents/{file_name}",
                file_name=file_name,
                file_type=DocumentType.TEXT,
                file_size=len(content.encode("utf-8")),
            ),
            chunk_index=chunk_index,
            start_offset=start_offset,
            end_offset=start_offset + len(content),
        ),
    )


def test_empty_results_return_empty_context() -> None:
    context = ContextBuilder().build([])

    assert context.text == ""
    assert context.sources == ()


def test_single_result_is_converted_to_context() -> None:
    result = _make_result(
        "alpha",
        result_id="result-1",
        score=0.9,
        chunk_index=2,
        start_offset=4,
    )

    context = ContextBuilder().build([result])

    assert context.text == "alpha"
    assert len(context.sources) == 1
    source = context.sources[0]
    assert source.result_id == "result-1"
    assert source.score == 0.9
    assert source.metadata == result.chunk.metadata
    assert source.chunk_index == 2
    assert source.start_offset == 4
    assert source.end_offset == 9


def test_multiple_results_preserve_order_and_separators() -> None:
    results = [
        _make_result("alpha", result_id="result-1", score=0.9),
        _make_result("beta", result_id="result-2", score=0.8),
    ]

    context = ContextBuilder().build(results)

    assert context.text == "alpha\n\nbeta"
    assert [source.result_id for source in context.sources] == ["result-1", "result-2"]


def test_max_context_chunks_limits_sources() -> None:
    results = [
        _make_result("alpha", result_id="result-1", score=0.9),
        _make_result("beta", result_id="result-2", score=0.8),
    ]

    context = ContextBuilder(ContextBuilderConfig(max_context_chunks=1)).build(results)

    assert context.text == "alpha"
    assert [source.result_id for source in context.sources] == ["result-1"]


def test_max_context_chars_truncates_last_included_result() -> None:
    results = [
        _make_result("abc", result_id="result-1", score=0.9),
        _make_result("def", result_id="result-2", score=0.8),
    ]

    context = ContextBuilder(ContextBuilderConfig(max_context_chars=7)).build(results)

    assert context.text == "abc\n\nde"
    assert [source.result_id for source in context.sources] == ["result-1", "result-2"]


def test_max_context_chars_stops_when_separator_does_not_fit() -> None:
    results = [
        _make_result("abc", result_id="result-1", score=0.9),
        _make_result("def", result_id="result-2", score=0.8),
    ]

    context = ContextBuilder(ContextBuilderConfig(max_context_chars=5)).build(results)

    assert context.text == "abc"
    assert [source.result_id for source in context.sources] == ["result-1"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_context_chunks", 0),
        ("max_context_chunks", -1),
        ("max_context_chars", 0),
        ("max_context_chars", -1),
    ],
)
def test_context_builder_config_rejects_invalid_limits(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ContextBuilderConfig(**{field: value})


def test_context_builder_does_not_modify_input_results() -> None:
    results = [
        _make_result("alpha", result_id="result-1", score=0.9),
        _make_result("beta", result_id="result-2", score=0.8),
    ]
    before = [result.model_dump() for result in results]

    ContextBuilder().build(results)

    assert [result.model_dump() for result in results] == before


def test_context_builder_is_deterministic() -> None:
    results = [
        _make_result("alpha", result_id="result-1", score=0.9),
        _make_result("beta", result_id="result-2", score=0.8),
    ]
    builder = ContextBuilder()

    assert builder.build(results) == builder.build(results)
