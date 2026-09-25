"""HTTP request and response models."""

from pydantic import BaseModel, ConfigDict, field_validator


class AgentRunRequest(BaseModel):
    """One text input for the configured Agent."""

    model_config = ConfigDict(frozen=True)

    input: str

    @field_validator("input")
    @classmethod
    def validate_input(cls, input: str) -> str:
        """Reject empty Agent input."""

        if not input.strip():
            raise ValueError("input must not be empty")
        return input


class AgentRunResponse(BaseModel):
    """One final Agent output."""

    model_config = ConfigDict(frozen=True)

    output: str


class HealthResponse(BaseModel):
    """Application health status."""

    model_config = ConfigDict(frozen=True)

    status: str


class ErrorResponse(BaseModel):
    """Stable API error response."""

    model_config = ConfigDict(frozen=True)

    detail: str
