"""RAG pipeline tool."""

from enterprise_ai_agent.rag import RAGPipeline, RAGResponse

from .base import Tool


class RAGTool(Tool[str, RAGResponse]):
    """Expose the RAG pipeline through the agent-facing tool boundary."""

    name = "rag"
    description = "Answer a user query using the knowledge-base RAG pipeline."

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    def run(self, input: str) -> RAGResponse:
        """Run the injected RAG pipeline without modifying its behavior."""

        return self._pipeline.run(input)
