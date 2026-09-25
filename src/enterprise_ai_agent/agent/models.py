"""Agent execution models."""

from pydantic import BaseModel, ConfigDict


class AgentResult[OutputT](BaseModel):
    """The result produced by one agent execution."""

    model_config = ConfigDict(frozen=True)

    output: OutputT
