"""Tests for deterministic RAG prompt construction."""

import pytest

from enterprise_ai_agent.ingestion import DocumentMetadata, DocumentType
from enterprise_ai_agent.rag import Context, ContextSource, PromptBuilder


def _make_source(
    *,
    result_id: str = "result-1",
    score: float = 0.9,
    file_name: str = "sample.txt",
    chunk_index: int = 0,
) -> ContextSource:
    return ContextSource(
        result_id=result_id,
        score=score,
        metadata=DocumentMetadata(
            source=f"/documents/{file_name}",
            file_name=file_name,
            file_type=DocumentType.TEXT,
            file_size=100,
        ),
        chunk_index=chunk_index,
        start_offset=0,
        end_offset=100,
    )


def test_prompt_builder_creates_structured_prompt() -> None:
    context = Context(text="alpha context", sources=(_make_source(),))

    prompt = PromptBuilder().build("What is alpha?", context)

    assert "provided context" in prompt.system
    assert prompt.context == (
        "alpha context\n\nSources:\n[Source 1] file_name=sample.txt; chunk_index=0; score=0.9"
    )
    assert prompt.question == "What is alpha?"


def test_prompt_to_text_has_clear_sections() -> None:
    context = Context(text="alpha context", sources=(_make_source(),))

    text = PromptBuilder().build("What is alpha?", context).to_text()

    assert text == (
        "Answer the question using only the provided context. "
        "Do not invent facts that are not present in the context. "
        "If the context is insufficient, state that the answer cannot be determined "
        "from the provided context.\n\n"
        "Context:\n"
        "alpha context\n\n"
        "Sources:\n"
        "[Source 1] file_name=sample.txt; chunk_index=0; score=0.9\n\n"
        "Question:\n"
        "What is alpha?"
    )


def test_prompt_builder_preserves_source_order() -> None:
    context = Context(
        text="first\n\nsecond",
        sources=(
            _make_source(result_id="first", score=0.9, file_name="first.txt"),
            _make_source(result_id="second", score=0.8, file_name="second.txt"),
        ),
    )

    prompt = PromptBuilder().build("question", context)

    assert prompt.context.index("[Source 1] file_name=first.txt") < prompt.context.index(
        "[Source 2] file_name=second.txt"
    )
    assert "[Source 3]" not in prompt.context


def test_prompt_builder_handles_empty_context() -> None:
    prompt = PromptBuilder().build("What is missing?", Context(text="", sources=()))

    assert prompt.context == "No retrieved context is available."
    assert "No retrieved context is available." in prompt.to_text()


def test_prompt_builder_supports_custom_system_instruction() -> None:
    context = Context(text="alpha", sources=(_make_source(),))

    prompt = PromptBuilder("Use a concise style.").build("question", context)

    assert prompt.system == "Use a concise style."
    assert prompt.to_text().startswith("Use a concise style.\n\n")


@pytest.mark.parametrize("question", ["", " ", "\n\t", "\r\n"])
def test_prompt_builder_rejects_empty_question(question: str) -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        PromptBuilder().build(question, Context(text="", sources=()))


@pytest.mark.parametrize("instruction", ["", " ", "\n\t"])
def test_prompt_builder_rejects_empty_system_instruction(instruction: str) -> None:
    with pytest.raises(ValueError, match="system instruction must not be empty"):
        PromptBuilder(instruction)


def test_prompt_builder_preserves_question_whitespace() -> None:
    question = "  What is alpha?  "

    prompt = PromptBuilder().build(question, Context(text="", sources=()))

    assert prompt.question == question
    assert prompt.to_text().endswith(f"Question:\n{question}")


def test_prompt_builder_does_not_modify_context() -> None:
    context = Context(text="alpha", sources=(_make_source(),))
    before = context.model_dump()

    PromptBuilder().build("question", context)

    assert context.model_dump() == before


def test_prompt_builder_is_deterministic() -> None:
    context = Context(text="alpha", sources=(_make_source(),))
    builder = PromptBuilder()

    assert builder.build("question", context) == builder.build("question", context)
