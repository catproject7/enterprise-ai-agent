"""OpenAI-compatible LLM service."""

from openai import OpenAI

from .service import LLMService


class OpenAICompatibleLLMService(LLMService):
    """LLM service backed by an injected OpenAI-compatible client."""

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

    def generate(self, prompt: str) -> str:
        """Generate one response through the injected client."""

        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = self._client.responses.create(
            model=self._model,
            input=prompt,
        )
        return response.output_text
