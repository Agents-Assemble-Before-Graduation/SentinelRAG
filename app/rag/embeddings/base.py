"""Abstract Base Class for Embedding Providers."""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Abstract interface for local or remote embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality (e.g. 384, 1536)."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the canonical model name."""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate vector embedding for a single text query or snippet."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a batch of text chunks."""
        pass
