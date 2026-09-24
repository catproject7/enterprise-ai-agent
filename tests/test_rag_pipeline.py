"""Tests for RAG pipeline orchestration."""

from collections.abc import Sequence

import pytest

from enterprise_ai_agent.chunking import Chunk
from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.llm import LLMService
from enterprise_ai_agent.rag import (
    Context,
    ContextBuilder,
    ContextSource,
    Prompt,
    PromptBuilder,
    RAGPipeline,
)
from enterprise_ai_agent.retrieval import Retriever
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
                file_size=100,
            ),
            chunk_index=chunk_index,
            start_offset=start_offset,
            end_offset=start_offset + len(content),
        ),
    )


def _make_context(
    *,
    text: str = "alpha context",
    sources: tuple[ContextSource, ...] = (),
) -> Context:
    return Context(text=text, sources=sources)


def _make_source(
    *,
    result_id: str = "result-1",
    file_name: str = "sample.txt",
    chunk_index: int = 0,
    start_offset: int = 0,
    end_offset: int = 10,
) -> ContextSource:
    return ContextSource(
        result_id=result_id,
        score=0.9,
        metadata=DocumentMetadata(
            source=f"/documents/{file_name}",
            file_name=file_name,
            file_type=DocumentType.TEXT,
            file_size=100,
        ),
        chunk_index=chunk_index,
        start_offset=start_offset,
        end_offset=end_offset,
    )


class FakeRetriever(Retriever):
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        error: Exception | None = None,
        event_log: list[str] | None = None,
    ) -> None:
        self.results = results if results is not None else []
        self.error = error
        self.calls: list[str] = []
        self.event_log = event_log

    def retrieve(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        self.calls.append(query)
        if self.event_log is not None:
            self.event_log.append("retrieve")
        if self.error is not None:
            raise self.error
        return self.results


class FakeContextBuilder(ContextBuilder):
    def __init__(
        self,
        context: Context,
        *,
        event_log: list[str] | None = None,
    ) -> None:
        self.context = context
        self.received_results: list[SearchResult] | None = None
        self.event_log = event_log

    def build(self, results: Sequence[SearchResult]) -> Context:
        self.received_results = list(results)
        if self.event_log is not None:
            self.event_log.append("context")
        return self.context


class FakePromptBuilder(PromptBuilder):
    def __init__(
        self,
        prompt: Prompt,
        *,
        event_log: list[str] | None = None,
    ) -> None:
        self.prompt = prompt
        self.received_question: str | None = None
        self.received_context: Context | None = None
        self.event_log = event_log

    def build(self, question: str, context: Context) -> Prompt:
        self.received_question = question
        self.received_context = context
        if self.event_log is not None:
            self.event_log.append("prompt")
        return self.prompt


class FakeLLMService(LLMService):
    def __init__(
        self,
        output: str = "generated answer",
        *,
        error: Exception | None = None,
        event_log: list[str] | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[str] = []
        self.event_log = event_log

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.event_log is not None:
            self.event_log.append("llm")
        if self.error is not None:
            raise self.error
        return self.output


def test_pipeline_happy_path_returns_rag_response() -> None:
    result = _make_result("alpha", result_id="result-1", score=0.9)
    pipeline = RAGPipeline(
        retriever=FakeRetriever([result]),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=FakeLLMService("generated answer"),
    )

    response = pipeline.run("What is alpha?")

    assert response.answer.text == "generated answer"
    assert len(response.citations) == 1
    assert response.citations[0].result_id == "result-1"


def test_pipeline_executes_components_in_order() -> None:
    event_log: list[str] = []
    pipeline = RAGPipeline(
        retriever=FakeRetriever([], event_log=event_log),
        context_builder=FakeContextBuilder(_make_context(), event_log=event_log),
        prompt_builder=FakePromptBuilder(
            Prompt(system="s", context="c", question="q"), event_log=event_log
        ),
        llm_service=FakeLLMService(event_log=event_log),
    )

    pipeline.run("question")

    assert event_log == ["retrieve", "context", "prompt", "llm"]


def test_pipeline_passes_original_question_to_retriever_and_prompt_builder() -> None:
    question = "  What is alpha?  "
    retriever = FakeRetriever([])
    prompt_builder = FakePromptBuilder(Prompt(system="s", context="c", question=question))
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=FakeContextBuilder(_make_context()),
        prompt_builder=prompt_builder,
        llm_service=FakeLLMService(),
    )

    pipeline.run(question)

    assert retriever.calls == [question]
    assert prompt_builder.received_question == question


def test_pipeline_passes_retrieval_results_to_context_builder() -> None:
    results = [_make_result("alpha", result_id="result-1", score=0.9)]
    context_builder = FakeContextBuilder(_make_context())
    pipeline = RAGPipeline(
        retriever=FakeRetriever(results),
        context_builder=context_builder,
        prompt_builder=FakePromptBuilder(Prompt(system="s", context="c", question="q")),
        llm_service=FakeLLMService(),
    )

    pipeline.run("question")

    assert context_builder.received_results == results


def test_pipeline_passes_rendered_prompt_to_llm_service() -> None:
    prompt = Prompt(system="system", context="context", question="question")
    llm_service = FakeLLMService()
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        context_builder=FakeContextBuilder(_make_context()),
        prompt_builder=FakePromptBuilder(prompt),
        llm_service=llm_service,
    )

    pipeline.run("question")

    assert llm_service.calls == [prompt.to_text()]


def test_pipeline_builds_citations_from_context_sources_in_order() -> None:
    sources = (
        _make_source(
            result_id="first", file_name="first.txt", chunk_index=1, start_offset=2, end_offset=12
        ),
        _make_source(
            result_id="second", file_name="second.txt", chunk_index=2, start_offset=5, end_offset=15
        ),
    )
    context_builder = FakeContextBuilder(_make_context(text="first\n\nsecond", sources=sources))
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        context_builder=context_builder,
        prompt_builder=FakePromptBuilder(Prompt(system="s", context="c", question="q")),
        llm_service=FakeLLMService("answer"),
    )

    response = pipeline.run("question")

    assert [citation.result_id for citation in response.citations] == ["first", "second"]
    assert [citation.metadata.file_name for citation in response.citations] == [
        "first.txt",
        "second.txt",
    ]
    assert [
        (citation.chunk_index, citation.start_offset, citation.end_offset)
        for citation in response.citations
    ] == [
        (1, 2, 12),
        (2, 5, 15),
    ]


def test_pipeline_empty_context_still_calls_llm_and_returns_no_citations() -> None:
    llm_service = FakeLLMService("cannot determine")
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=llm_service,
    )

    response = pipeline.run("question")

    assert response.answer.text == "cannot determine"
    assert response.citations == ()
    assert len(llm_service.calls) == 1


def test_pipeline_does_not_modify_retrieval_results() -> None:
    results = [_make_result("alpha", result_id="result-1", score=0.9)]
    before = [result.model_dump() for result in results]
    pipeline = RAGPipeline(
        retriever=FakeRetriever(results),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=FakeLLMService(),
    )

    pipeline.run("question")

    assert [result.model_dump() for result in results] == before


def test_pipeline_propagates_retriever_error() -> None:
    error = RuntimeError("retrieval failed")
    pipeline = RAGPipeline(
        retriever=FakeRetriever(error=error),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=FakeLLMService(),
    )

    with pytest.raises(RuntimeError, match="retrieval failed") as captured:
        pipeline.run("question")

    assert captured.value is error


def test_pipeline_propagates_llm_error() -> None:
    error = RuntimeError("generation failed")
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=FakeLLMService(error=error),
    )

    with pytest.raises(RuntimeError, match="generation failed") as captured:
        pipeline.run("question")

    assert captured.value is error


def test_pipeline_is_deterministic() -> None:
    pipeline = RAGPipeline(
        retriever=FakeRetriever([]),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=FakeLLMService("answer"),
    )

    assert pipeline.run("question") == pipeline.run("question")
