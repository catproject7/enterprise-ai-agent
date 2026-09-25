"""Compose existing components into a runnable Agent and FastAPI application."""

from fastapi import FastAPI
from openai import OpenAI
from qdrant_client import QdrantClient

from enterprise_ai_agent.agent import Agent, LLMAgent
from enterprise_ai_agent.api import create_app
from enterprise_ai_agent.core.config import AppSettings, get_settings
from enterprise_ai_agent.embeddings import (
    EmbeddingService,
    FastEmbedEmbeddingService,
)
from enterprise_ai_agent.llm import (
    LLMService,
    OpenAICompatibleLLMService,
    OpenAICompatibleToolCallingLLM,
    ToolCallingLLM,
)
from enterprise_ai_agent.observability import (
    TracedAgent,
    TracedToolCallingLLM,
    TracedToolRegistry,
    configure_observability_logging,
)
from enterprise_ai_agent.rag import ContextBuilder, PromptBuilder, RAGPipeline
from enterprise_ai_agent.retrieval import Retriever
from enterprise_ai_agent.tools import RAGTool, ToolRegistry
from enterprise_ai_agent.vector_store import QdrantVectorStore, VectorStore


class RuntimeConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


def assemble_agent(
    *,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    llm_service: LLMService,
    tool_calling_llm: ToolCallingLLM,
    instrument: bool = True,
) -> Agent[str]:
    """Assemble existing components into a RAG-capable LLM Agent."""

    retriever = Retriever(embedding_service, vector_store)
    pipeline = RAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=llm_service,
    )
    registry = ToolRegistry()
    registry.register(RAGTool(pipeline), argument_name="query")

    if instrument:
        registry = TracedToolRegistry(registry)
        tool_calling_llm = TracedToolCallingLLM(tool_calling_llm)

    agent = LLMAgent(tool_calling_llm, registry)
    if instrument:
        return TracedAgent(agent)
    return agent


def create_runtime(settings: AppSettings | None = None) -> Agent[str]:
    """Create a real Agent from application settings."""

    resolved = settings if settings is not None else get_settings()
    _validate_runtime_settings(resolved)

    embedding_service = FastEmbedEmbeddingService(
        model_name=resolved.embedding_model,
        dimension=resolved.embedding_dimension,
        batch_size=resolved.embedding_batch_size,
        cache_dir=resolved.embedding_cache_dir,
    )
    vector_store = QdrantVectorStore(
        _build_qdrant_client(resolved),
        collection_name=resolved.qdrant_collection_name,
    )
    vector_store.ensure_collection(embedding_service.dimension)

    openai_client = _build_openai_client(resolved)
    llm_service = OpenAICompatibleLLMService(
        openai_client,
        model=resolved.llm_model,
    )
    tool_calling_llm = OpenAICompatibleToolCallingLLM(
        openai_client,
        model=resolved.llm_model,
    )

    return assemble_agent(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        tool_calling_llm=tool_calling_llm,
    )


def create_runtime_app(settings: AppSettings | None = None) -> FastAPI:
    """Create FastAPI with a fully configured Agent."""

    resolved = settings if settings is not None else get_settings()
    configure_observability_logging(resolved.log_level)
    return create_app(agent=create_runtime(resolved))


def _validate_runtime_settings(settings: AppSettings) -> None:
    api_key = settings.llm_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise RuntimeConfigurationError(
            "ENTERPRISE_AI_AGENT_LLM_API_KEY is required to create the runtime"
        )


def _build_qdrant_client(settings: AppSettings) -> QdrantClient:
    if settings.qdrant_url == ":memory:":
        return QdrantClient(location=":memory:")

    client_kwargs: dict[str, object] = {"url": settings.qdrant_url}
    if settings.qdrant_api_key is not None:
        client_kwargs["api_key"] = settings.qdrant_api_key.get_secret_value()
    return QdrantClient(**client_kwargs)


def _build_openai_client(settings: AppSettings) -> OpenAI:
    api_key = settings.llm_api_key
    if api_key is None:
        raise RuntimeConfigurationError(
            "ENTERPRISE_AI_AGENT_LLM_API_KEY is required to create the runtime"
        )

    client_kwargs: dict[str, object] = {
        "api_key": api_key.get_secret_value(),
    }
    if settings.llm_base_url is not None:
        client_kwargs["base_url"] = settings.llm_base_url
    return OpenAI(**client_kwargs)
