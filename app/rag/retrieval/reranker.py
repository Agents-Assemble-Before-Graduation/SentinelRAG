"""Cross-Encoder Reranker interface and implementations (FastEmbed and Mock)."""

import os
from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retrieval.retriever import RetrievedChunk

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract interface for all Rerankers."""

    @abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Re-scores and re-orders a list of retrieved chunks in-place against a query.

        Args:
            query: The user query string.
            chunks: A list of RetrievedChunk objects.

        Returns:
            The sorted list of RetrievedChunk objects (highest score first).
        """
        pass


class MockReranker(BaseReranker):
    """Deterministic, offline-safe mock reranker for testing and fallbacks.

    Uses a simple term overlap metric to re-score candidate chunks.
    """

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        query_words = set(query.lower().split())
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            # Base score combined with a fraction of the overlap
            chunk.score = float(overlap) + 0.01 * chunk.score

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks


class FastEmbedReranker(BaseReranker):
    """Cross-Encoder Reranker using local ONNX runtimes via fastembed.rerank."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._encoder = None

    @property
    def encoder(self):
        """Lazy initializer for FastEmbed's TextCrossEncoder."""
        if self._encoder is None:
            from fastembed.rerank.cross_encoder.text_cross_encoder import (
                TextCrossEncoder,
            )

            # Map configured models to ones supported by FastEmbed Rerank
            # bge-reranker-large is not supported, so fallback to bge-reranker-base
            effective_model = self.model_name
            if "bge-reranker-large" in effective_model.lower():
                effective_model = "BAAI/bge-reranker-base"
            elif "/" not in effective_model and effective_model != "bge-reranker-base":
                # Ensure supported default model is selected
                effective_model = "Xenova/ms-marco-MiniLM-L-6-v2"

            try:
                self._encoder = TextCrossEncoder(model_name=effective_model)
            except Exception as e:
                logger.warning(
                    "Failed to initialize FastEmbed TextCrossEncoder for model %s: %s. "
                    "Reranking operations will fall back to MockReranker.",
                    effective_model,
                    str(e),
                )
                raise e
        return self._encoder

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        try:
            # fastembed rerank call: takes a query and an iterable of document content strings
            doc_texts = [c.content for c in chunks]
            scores = list(self.encoder.rerank(query, doc_texts))

            for chunk, score in zip(chunks, scores, strict=False):
                chunk.score = float(score)

            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks
        except Exception as e:
            logger.warning(
                "Error encountered during FastEmbed reranking: %s. Falling back to simple term overlap.",
                str(e),
            )
            # Fall back to mock reranker behavior
            mock = MockReranker()
            return await mock.rerank(query, chunks)


# Singleton cache for resolved reranker
_cached_reranker: Optional[BaseReranker] = None


def get_reranker(reranker_type: Optional[str] = None) -> BaseReranker:
    """Factory to retrieve or instantiate singleton Reranker.

    If in a testing environment or if initialization fails, falls back to MockReranker.
    """
    global _cached_reranker
    if _cached_reranker is not None and reranker_type is None:
        return _cached_reranker

    settings = get_settings()
    selected_type = (reranker_type or settings.RERANKER_PROVIDER).lower()

    # Force mock reranker in testing environment to prevent network connection errors
    if settings.is_testing or selected_type in {"mock", "test"}:
        reranker = MockReranker()
    else:
        try:
            reranker = FastEmbedReranker(model_name=settings.RERANKER_MODEL)
            # Try initializing to verify HF connectivity
            _ = reranker.encoder
        except Exception:
            logger.warning(
                "Hugging Face unavailable or connection failed. Using offline-safe MockReranker."
            )
            reranker = MockReranker()

    if reranker_type is None:
        _cached_reranker = reranker

    return reranker


def reset_reranker_cache() -> None:
    """Clear cached reranker singleton. For testing purposes."""
    global _cached_reranker
    _cached_reranker = None
