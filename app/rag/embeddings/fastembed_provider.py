"""Local ONNX-based embedding provider using FastEmbed."""

import asyncio

from fastembed import TextEmbedding

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings.base import BaseEmbeddingProvider

logger = get_logger(__name__)


class FastEmbedEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding generator using fast ONNX runtime models (zero API keys)."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._dimension = settings.EMBEDDING_DIMENSION
        self._model: TextEmbedding | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> TextEmbedding:
        """Lazy load the FastEmbed TextEmbedding model."""
        if self._model is None:
            logger.info("Initializing local FastEmbed model: %s", self._model_name)
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embedding inference using FastEmbed generator."""
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        res = await self.embed_batch([text])
        return res[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of text chunks asynchronously."""
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)
