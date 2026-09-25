"""OpenAI-compatible implementation of LLM tool calling."""

from collections.abc import Sequence

from openai import OpenAI
from openai.types.responses import FunctionToolParam

from .tool_calling import (
    ToolCall,
    ToolCallingLLM,
    ToolCallingResponse,
    ToolResult,
    ToolSpec,
    UnsupportedToolCallsError,
)


class OpenAICompatibleToolCallingLLM(ToolCallingLLM):
    """Tool-calling service backed by an injected OpenAI-compatible client."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        self._client = client
        self._model = model

    def generate_with_tools(
        self,
        prompt: str,
        tools: Sequence[ToolSpec],
        tool_results: Sequence[ToolResult] = (),
    ) -> ToolCallingResponse:
        """Generate one response through the injected client."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self._client.responses.create(
            model=self._model,
            input=self._build_input(prompt, tool_results),
            tools=[self._to_openai_tool(tool) for tool in tools],
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        tool_calls = [
            item for item in response.output if getattr(item, "type", None) == "function_call"
        ]
        if len(tool_calls) > 1:
            raise UnsupportedToolCallsError("parallel tool calls are not supported")
        if tool_calls:
            tool_call = tool_calls[0]
            return ToolCallingResponse(
                tool_call=ToolCall(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
            )

        return ToolCallingResponse(text=response.output_text)

    @staticmethod
    def _to_openai_tool(tool: ToolSpec) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
            strict=True,
        )

    @staticmethod
    def _build_input(
        prompt: str,
        tool_results: Sequence[ToolResult],
    ) -> str | list[dict[str, object]]:
        if not tool_results:
            return prompt

        input_items: list[dict[str, object]] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        for result in tool_results:
            input_items.extend(
                [
                    {
                        "type": "function_call",
                        "call_id": result.call.call_id,
                        "name": result.call.name,
                        "arguments": result.call.arguments,
                    },
                    {
                        "type": "function_call_output",
                        "call_id": result.call.call_id,
                        "output": result.output,
                    },
                ]
            )
        return input_items
