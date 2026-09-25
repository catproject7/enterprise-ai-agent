"""Provider-neutral models and abstraction for LLM tool calling."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class ToolCallingError(Exception):
    """Base exception for tool-calling failures."""


class UnsupportedToolCallsError(ToolCallingError):
    """Raised when a provider returns multiple tool calls in one response."""


class ToolSpec(BaseModel):
    """Provider-neutral metadata describing one callable tool."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, object]

    @field_validator("name", "description")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject empty tool metadata."""

        if not value.strip():
            raise ValueError("tool metadata must not be empty")
        return value


class ToolCall(BaseModel):
    """One provider-neutral request to invoke a tool."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: str

    @field_validator("call_id", "name", "arguments")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject empty tool call fields."""

        if not value.strip():
            raise ValueError("tool call fields must not be empty")
        return value


class ToolResult(BaseModel):
    """The deterministic string output associated with one tool call."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    output: str


class ToolCallingResponse(BaseModel):
    """Either a final text response or one requested tool call."""

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def validate_response(self) -> "ToolCallingResponse":
        """Require exactly one of text or tool_call."""

        if (self.text is None) == (self.tool_call is None):
            raise ValueError("exactly one of text or tool_call must be provided")
        if self.text is not None and not self.text.strip():
            raise ValueError("response text must not be empty")
        return self


class ToolCallingLLM(ABC):
    """Replaceable interface for synchronous LLM tool-calling decisions."""

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: Sequence[ToolSpec],
        tool_results: Sequence[ToolResult] = (),
    ) -> ToolCallingResponse:
        """Generate text or request one tool call for the given tool context."""
