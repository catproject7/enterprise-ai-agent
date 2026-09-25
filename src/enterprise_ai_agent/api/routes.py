"""HTTP routes for the Agent API."""

from typing import Annotated

from fastapi import APIRouter, Depends

from enterprise_ai_agent.agent import Agent

from .dependencies import get_agent
from .models import AgentRunRequest, AgentRunResponse, ErrorResponse, HealthResponse

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
