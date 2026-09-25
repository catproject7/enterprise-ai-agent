"""LLM service primitives."""

from .openai import OpenAICompatibleLLMService
from .openai_tool_calling import OpenAICompatibleToolCallingLLM
from .service import LLMService
from .tool_calling import (
    ToolCall,
    ToolCallingLLM,
    ToolCallingResponse,
    ToolResult,
    ToolSpec,
    UnsupportedToolCallsError,
)

__all__ = [
    "LLMService",
    "OpenAICompatibleToolCallingLLM",
    "OpenAICompatibleLLMService",
    "ToolCall",
    "ToolCallingLLM",
    "ToolCallingResponse",
    "ToolResult",
    "ToolSpec",
    "UnsupportedToolCallsError",
]
