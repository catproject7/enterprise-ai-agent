"""Pure metric calculations for RAG evaluation."""

import re
from collections.abc import Sequence

from enterprise_ai_agent.rag import Citation, RAGResponse
from enterprise_ai_agent.vector_store import SearchResult

from .models import (
    AnswerCorrectness,
    CitationCoverage,
    GroundTruthChunk,
    RetrievalMetrics,
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def _ground_truth_identity(chunk: GroundTruthChunk) -> tuple[str, int, int, int]:
    return (chunk.source, chunk.chunk_index, chunk.start_offset, chunk.end_offset)


def _search_result_identity(result: SearchResult) -> tuple[str, int, int, int]:
    chunk = result.chunk
    return (
        chunk.metadata.source,
        chunk.chunk_index,
        chunk.start_offset,
        chunk.end_offset,
    )


def _citation_identity(citation: Citation) -> tuple[str, int, int, int]:
    return (
        citation.metadata.source,
        citation.chunk_index,
        citation.start_offset,
        citation.end_offset,
    )


def calculate_retrieval_metrics(
    relevant_chunks: Sequence[GroundTruthChunk],
    actual_results: Sequence[SearchResult],
    *,
    k: int,
) -> RetrievalMetrics:
    """Calculate Recall@K and Precision@K for one retrieval result set."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    relevant_count = len(relevant_chunks)
    if relevant_count == 0:
        raise ValueError("relevant_chunks must not be empty")

    relevant_identities = {_ground_truth_identity(chunk) for chunk in relevant_chunks}
    retrieved_identities = {_search_result_identity(result) for result in actual_results[:k]}
    matched_relevant = len(relevant_identities & retrieved_identities)

    return RetrievalMetrics(
        recall_at_k=matched_relevant / relevant_count,
        precision_at_k=matched_relevant / k,
        matched_relevant=matched_relevant,
        relevant_count=relevant_count,
    )


def _normalize_answer(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text.strip().casefold())


def calculate_answer_correctness(
    expected_answer: str,
    actual_answer: str,
) -> AnswerCorrectness:
    """Calculate normalized exact match for an answer."""

    exact_match = _normalize_answer(expected_answer) == _normalize_answer(actual_answer)
    return AnswerCorrectness(
        exact_match=exact_match,
        score=1.0 if exact_match else 0.0,
    )


def calculate_citation_coverage(
    relevant_chunks: Sequence[GroundTruthChunk],
    response: RAGResponse,
) -> CitationCoverage:
    """Calculate how many expected sources are present in response citations."""

    relevant_identities = {_ground_truth_identity(chunk) for chunk in relevant_chunks}
    citation_identities = {_citation_identity(citation) for citation in response.citations}
    matched_citations = len(relevant_identities & citation_identities)
    expected_citations = len(relevant_chunks)

    coverage = matched_citations / expected_citations if expected_citations > 0 else 0.0
    return CitationCoverage(
        matched_citations=matched_citations,
        expected_citations=expected_citations,
        coverage=coverage,
    )
