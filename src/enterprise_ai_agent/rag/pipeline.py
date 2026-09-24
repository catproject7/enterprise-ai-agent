"""RAG pipeline orchestration."""

from enterprise_ai_agent.llm import LLMService
from enterprise_ai_agent.retrieval import Retriever

from .context import ContextBuilder
from .prompt import PromptBuilder
from .response import Answer, Citation, RAGResponse


class RAGPipeline:
    """Orchestrate retrieval, context, prompt, generation, and citations."""

    def __init__(
        self,
        retriever: Retriever,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
    ) -> None:
        self._retriever = retriever
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm_service = llm_service

    def run(self, question: str) -> RAGResponse:
        """Run the RAG flow for one question."""

        results = self._retriever.retrieve(question)
        context = self._context_builder.build(results)
        prompt = self._prompt_builder.build(question, context)
        answer_text = self._llm_service.generate(prompt.to_text())
        citations = tuple(
            Citation(
                result_id=source.result_id,
                metadata=source.metadata,
                chunk_index=source.chunk_index,
                start_offset=source.start_offset,
                end_offset=source.end_offset,
            )
            for source in context.sources
        )

        return RAGResponse(answer=Answer(text=answer_text), citations=citations)
