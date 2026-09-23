"""FastEmbed-backed embedding service."""

from collections.abc import Sequence

from fastembed import TextEmbedding

from .service import EmbeddingService


class FastEmbedEmbeddingService(EmbeddingService):
    """Embedding service backed by a lazily initialized FastEmbed model."""

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("embedding model name must not be empty")
        if batch_size <= 0:
            raise ValueError("embedding batch size must be greater than zero")

        super().__init__(dimension)
        self._model_name = model_name
        self._batch_size = batch_size
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts with FastEmbed in the requested order."""

        if not texts:
            return []

        vectors = [
            [float(value) for value in vector]
            for vector in self._get_model().embed(texts, batch_size=self._batch_size)
        ]
        self._validate_vectors(vectors, expected_count=len(texts))
        return vectors

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                lazy_load=True,
            )
        return self._model
