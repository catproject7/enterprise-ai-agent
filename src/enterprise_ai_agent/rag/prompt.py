"""Deterministic prompt construction from RAG context."""

from pydantic import BaseModel, ConfigDict

from .context import Context

_DEFAULT_SYSTEM_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Do not invent facts that are not present in the context. "
    "If the context is insufficient, state that the answer cannot be determined "
    "from the provided context."
)


class Prompt(BaseModel):
    """Structured prompt sections ready to render for an LLM."""

    model_config = ConfigDict(frozen=True)

    system: str
    context: str
    question: str

    def to_text(self) -> str:
        """Render the prompt as one deterministic string."""

        return f"{self.system}\n\nContext:\n{self.context}\n\nQuestion:\n{self.question}"


class PromptBuilder:
    """Build a deterministic prompt from a question and retrieved context."""

    def __init__(self, system_instruction: str | None = None) -> None:
        instruction = (
            system_instruction if system_instruction is not None else _DEFAULT_SYSTEM_INSTRUCTION
        )
        if not instruction.strip():
            raise ValueError("system instruction must not be empty")
        self._system_instruction = instruction

    def build(self, question: str, context: Context) -> Prompt:
        """Build a prompt without calling an LLM or mutating context."""

        if not question.strip():
            raise ValueError("question must not be empty")

        return Prompt(
            system=self._system_instruction,
            context=self._format_context(context),
            question=question,
        )

    @staticmethod
    def _format_context(context: Context) -> str:
        if not context.text and not context.sources:
            return "No retrieved context is available."

        sections: list[str] = []
        if context.text:
            sections.append(context.text)

        if context.sources:
            source_lines = [
                (
                    f"[Source {index}] file_name={source.metadata.file_name}; "
                    f"chunk_index={source.chunk_index}; score={source.score}"
                )
                for index, source in enumerate(context.sources, start=1)
            ]
            sections.append("Sources:\n" + "\n".join(source_lines))

        return "\n\n".join(sections)
