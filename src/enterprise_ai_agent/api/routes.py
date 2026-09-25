"""HTTP routes for the Agent API."""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from enterprise_ai_agent.agent import Agent
from enterprise_ai_agent.conversation import ConversationService

from .dependencies import get_agent, get_conversation_service
from .models import (
    AgentRunRequest,
    AgentRunResponse,
    ConversationResponse,
    CreateConversationResponse,
    ErrorResponse,
    HealthResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Return application health without requiring an Agent."""

    return HealthResponse(status="ok")


@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        502: {"model": ErrorResponse, "description": "Agent execution failed"},
        503: {"model": ErrorResponse, "description": "Agent is not configured"},
    },
)
def run_agent(
    request: AgentRunRequest,
    agent: Annotated[Agent[str], Depends(get_agent)],
) -> AgentRunResponse:
    """Run the configured Agent for one text input."""

    result = agent.run(request.input)
    return AgentRunResponse(output=result.output)


@router.post(
    "/conversations",
    response_model=CreateConversationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Agent is not configured"},
    },
)
def create_conversation(
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> CreateConversationResponse:
    """Create an empty conversation."""

    conversation = service.create_conversation()
    return CreateConversationResponse.from_domain(conversation)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Conversation not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Agent is not configured"},
    },
)
def get_conversation(
    conversation_id: str,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ConversationResponse:
    """Return one conversation and its messages."""

    conversation = service.get_conversation(conversation_id)
    return ConversationResponse.from_domain(conversation)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=SendMessageResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Conversation not found"},
        422: {"model": ErrorResponse, "description": "Invalid message"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        502: {"model": ErrorResponse, "description": "Agent execution failed"},
        503: {"model": ErrorResponse, "description": "Agent is not configured"},
    },
)
def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> SendMessageResponse:
    """Run the Agent and append one user and assistant message pair."""

    conversation = service.send_message(conversation_id, request.content)
    user_message = conversation.messages[-2]
    assistant_message = conversation.messages[-1]
    return SendMessageResponse(
        conversation_id=conversation.id,
        user_message=MessageResponse.from_domain(user_message),
        assistant_message=MessageResponse.from_domain(assistant_message),
    )
