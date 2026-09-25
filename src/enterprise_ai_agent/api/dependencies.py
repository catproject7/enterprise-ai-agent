"""FastAPI dependencies."""

from fastapi import HTTPException, Request, status

from enterprise_ai_agent.agent import Agent


def get_agent(request: Request) -> Agent[str]:
    """Return the configured Agent or fail closed."""

    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent is not configured",
        )
    return agent
