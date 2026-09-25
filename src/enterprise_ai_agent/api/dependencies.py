"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from enterprise_ai_agent.agent import Agent
from enterprise_ai_agent.conversation import (
    ConversationService,
    ConversationStore,
)
from enterprise_ai_agent.rag import RAGPipeline


def get_agent(request: Request) -> Agent[str]:
    """Return the configured Agent or fail closed."""

    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is not configured",
        )
    return agent


def get_conversation_store(request: Request) -> ConversationStore:
    """Return the configured conversation store or fail closed."""

    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation store is not configured",
        )
    return store


def get_conversation_service(
    agent: Annotated[Agent[str], Depends(get_agent)],
    store: Annotated[ConversationStore, Depends(get_conversation_store)],
) -> ConversationService:
    """Build the conversation service from injected dependencies."""

    return ConversationService(store, agent)


def get_rag_pipeline(request: Request) -> RAGPipeline:
    """Return the configured RAG pipeline or fail closed."""

    pipeline = getattr(request.app.state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not configured",
        )
    return pipeline
