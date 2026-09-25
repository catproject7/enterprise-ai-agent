"""Tool registry and string-input invocation adapter."""

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from enterprise_ai_agent.llm.tool_calling import ToolCall, ToolResult, ToolSpec

from .base import Tool
from .exceptions import DuplicateToolNameError, ToolArgumentError, ToolNotFoundError


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    tool: Tool[str, Any]
    argument_name: str


class ToolRegistry:
    """Register named string-input tools and invoke them from tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def register(
        self,
        tool: Tool[str, Any],
        *,
        argument_name: str,
    ) -> None:
        """Register one tool and its canonical string argument name."""

        if not argument_name.strip():
            raise ValueError("argument_name must not be empty")
        if tool.name in self._tools:
            raise DuplicateToolNameError(f"tool '{tool.name}' is already registered")

        self._tools[tool.name] = _RegisteredTool(
            tool=tool,
            argument_name=argument_name,
        )

    def get(self, name: str) -> Tool[str, Any]:
        """Return a registered tool by name."""

        registration = self._tools.get(name)
        if registration is None:
            raise ToolNotFoundError(f"tool '{name}' is not registered")
        return registration.tool

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return provider-neutral tool metadata in registration order."""

        return tuple(
            ToolSpec(
                name=name,
                description=registration.tool.description,
                input_schema=self._input_schema(registration.argument_name),
            )
            for name, registration in self._tools.items()
        )

    def invoke(self, call: ToolCall) -> ToolResult:
        """Parse one tool call, execute its tool, and serialize the output."""

        registration = self._tools.get(call.name)
        if registration is None:
            raise ToolNotFoundError(f"tool '{call.name}' is not registered")

        arguments = self._parse_arguments(call.arguments)
        if set(arguments) != {registration.argument_name}:
            raise ToolArgumentError(
                f"tool arguments must contain only '{registration.argument_name}'"
            )

        value = arguments[registration.argument_name]
        if not isinstance(value, str):
            raise ToolArgumentError(
                f"tool argument '{registration.argument_name}' must be a string"
            )

        output = registration.tool.run(value)
        return ToolResult(call=call, output=self._serialize_output(output))

    @staticmethod
    def _input_schema(argument_name: str) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                argument_name: {
                    "type": "string",
                }
            },
            "required": [argument_name],
            "additionalProperties": False,
        }

    @staticmethod
    def _parse_arguments(arguments: str) -> dict[str, object]:
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ToolArgumentError("tool arguments must be valid JSON") from error

        if not isinstance(parsed, dict):
            raise ToolArgumentError("tool arguments must be a JSON object")
        return parsed

    @staticmethod
    def _serialize_output(output: object) -> str:
        if isinstance(output, str):
            return output
        if isinstance(output, BaseModel):
            return output.model_dump_json()
        return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
