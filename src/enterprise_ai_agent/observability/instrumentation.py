"""Transparent tracing wrappers for Agent, LLM, and Tool boundaries."""

import logging
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from enterprise_ai_agent.agent import Agent, AgentResult
from enterprise_ai_agent.llm import (
    ToolCall,
    ToolCallingLLM,
    ToolCallingResponse,
    ToolResult,
    ToolSpec,
)
from enterprise_ai_agent.tools import Tool, ToolRegistry

from .logging import log_event


class TracedAgent[OutputT](Agent[OutputT]):
    """Trace an Agent without changing its contract."""

    def __init__(self, agent: Agent[OutputT]) -> None:
        self._agent = agent

    def run(self, input: str) -> AgentResult[OutputT]:
        """Run the wrapped Agent and emit lifecycle events."""

        started = perf_counter()
        log_event("agent.started", component="agent")
        try:
            result = self._agent.run(input)
        except Exception as error:
            log_event(
                "agent.failed",
                component="agent",
                duration_ms=(perf_counter() - started) * 1000,
                status="error",
                error_type=type(error).__name__,
                level=logging.ERROR,
            )
            raise

        log_event(
            "agent.completed",
            component="agent",
            duration_ms=(perf_counter() - started) * 1000,
            status="ok",
        )
        return result


class TracedToolCallingLLM(ToolCallingLLM):
    """Trace a ToolCallingLLM without changing its contract."""

    def __init__(self, llm: ToolCallingLLM) -> None:
        self._llm = llm

    def generate_with_tools(
        self,
        prompt: str,
        tools: Sequence[ToolSpec],
        tool_results: Sequence[ToolResult] = (),
    ) -> ToolCallingResponse:
        """Generate a response and emit lifecycle events."""

        started = perf_counter()
        log_event("llm.started", component="llm")
        try:
            response = self._llm.generate_with_tools(prompt, tools, tool_results)
        except Exception as error:
            log_event(
                "llm.failed",
                component="llm",
                duration_ms=(perf_counter() - started) * 1000,
                status="error",
                error_type=type(error).__name__,
                level=logging.ERROR,
            )
            raise

        tool_call = response.tool_call
        log_event(
            "llm.completed",
            component="llm",
            duration_ms=(perf_counter() - started) * 1000,
            status="ok",
            tool_name=tool_call.name if tool_call is not None else None,
            call_id=tool_call.call_id if tool_call is not None else None,
        )
        return response


class TracedToolRegistry(ToolRegistry):
    """Trace ToolRegistry invocation without changing its behavior."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry

    def register(self, tool: Tool[str, Any], *, argument_name: str) -> None:
        """Register through the wrapped registry when present."""

        if self._registry is not None:
            self._registry.register(tool, argument_name=argument_name)
            return
        super().register(tool, argument_name=argument_name)

    def get(self, name: str) -> Tool[str, Any]:
        """Return a tool from the wrapped registry when present."""

        if self._registry is not None:
            return self._registry.get(name)
        return super().get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return specs from the wrapped registry when present."""

        if self._registry is not None:
            return self._registry.specs()
        return super().specs()

    def invoke(self, call: ToolCall) -> ToolResult:
        """Invoke the wrapped registry and emit lifecycle events."""

        started = perf_counter()
        log_event(
            "tool.started",
            component="tool",
            tool_name=call.name,
            call_id=call.call_id,
        )
        try:
            if self._registry is not None:
                result = self._registry.invoke(call)
            else:
                result = super().invoke(call)
        except Exception as error:
            log_event(
                "tool.failed",
                component="tool",
                duration_ms=(perf_counter() - started) * 1000,
                status="error",
                error_type=type(error).__name__,
                tool_name=call.name,
                call_id=call.call_id,
                level=logging.ERROR,
            )
            raise

        log_event(
            "tool.completed",
            component="tool",
            duration_ms=(perf_counter() - started) * 1000,
            status="ok",
            tool_name=call.name,
            call_id=call.call_id,
        )
        return result
