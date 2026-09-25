"""Compose existing components into a runnable Agent and FastAPI application."""

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """Shared dependencies used by the Agent and RAG API."""

    agent: Agent[str]
    rag_pipeline: RAGPipeline


def assemble_agent(
    *,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    llm_service: LLMService,
    tool_calling_llm: ToolCallingLLM,
    instrument: bool = True,
) -> Agent[str]:
    """Assemble existing components into a RAG-capable LLM Agent."""

    pipeline = _build_rag_pipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
    )
    return _assemble_agent_from_pipeline(
        pipeline,
        tool_calling_llm,
        instrument=instrument,
    )


def create_runtime_components(
    settings: AppSettings | None = None,
) -> RuntimeComponents:
    """Create the shared Agent and RAG dependencies from application settings."""

    resolved = settings if settings is not None else get_settings()
    _validate_runtime_settings(resolved)

    embedding_service = _build_embedding_service(resolved)
    vector_store = _build_vector_store(resolved, embedding_service)

    openai_client = _build_openai_client(resolved)
    llm_service = OpenAICompatibleLLMService(
        openai_client,
        model=resolved.llm_model,
    )
    tool_calling_llm = OpenAICompatibleToolCallingLLM(
        openai_client,
        model=resolved.llm_model,
    )
    rag_pipeline = _build_rag_pipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
    )
    agent = _assemble_agent_from_pipeline(
        rag_pipeline,
        tool_calling_llm,
        instrument=True,
    )
    return RuntimeComponents(agent=agent, rag_pipeline=rag_pipeline)


def create_runtime(settings: AppSettings | None = None) -> Agent[str]:
    """Create a real Agent from application settings."""

    return create_runtime_components(settings).agent


def create_runtime_app(settings: AppSettings | None = None) -> FastAPI:
    """Create FastAPI with a fully configured Agent and RAG pipeline."""

    resolved = settings if settings is not None else get_settings()
    configure_observability_logging(resolved.log_level)
    components = create_runtime_components(resolved)
    return create_app(
        agent=components.agent,
        rag_pipeline=components.rag_pipeline,
    )


def _assemble_agent_from_pipeline(
    pipeline: RAGPipeline,
    tool_calling_llm: ToolCallingLLM,
    *,
    instrument: bool,
) -> Agent[str]:
    registry = ToolRegistry()
    registry.register(RAGTool(pipeline), argument_name="query")
    if instrument:
        registry = TracedToolRegistry(registry)
        tool_calling_llm = TracedToolCallingLLM(tool_calling_llm)

    agent = LLMAgent(tool_calling_llm, registry)
    if instrument:
        return TracedAgent(agent)
    return agent


def _build_rag_pipeline(
    *,
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    llm_service: LLMService,
) -> RAGPipeline:
    retriever = Retriever(embedding_service, vector_store)
    return RAGPipeline(
        retriever=retriever,
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(),
        llm_service=llm_service,
    )


def _build_embedding_service(settings: AppSettings) -> EmbeddingService:
    return FastEmbedEmbeddingService(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        cache_dir=settings.embedding_cache_dir,
    )


def _build_vector_store(
    settings: AppSettings,
    embedding_service: EmbeddingService,
) -> VectorStore:
    vector_store = QdrantVectorStore(
        _build_qdrant_client(settings),
        collection_name=settings.qdrant_collection_name,
    )
    vector_store.ensure_collection(embedding_service.dimension)
    return vector_store


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
