"""Deterministic offline embedding provider for isolated testing and fallback."""

import hashlib
import math
import random

from app.rag.embeddings.base import BaseEmbeddingProvider


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """Zero-download deterministic embedding provider for test isolation."""

    def __init__(self, dimension: int = 384, model_name: str = "deterministic-384d") -> None:
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _generate_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit-normalized vector for the input string."""
        # Seed random number generator with SHA-256 of text
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed_int = int(digest[:16], 16)
        rng = random.Random(seed_int)

        # Generate Gaussian random values
        raw_vector = [rng.gauss(0.0, 1.0) for _ in range(self._dimension)]

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in raw_vector))
        if norm == 0:
            return [1.0 / math.sqrt(self._dimension)] * self._dimension

        return [round(x / norm, 6) for x in raw_vector]

    async def embed_text(self, text: str) -> list[float]:
        return self._generate_vector(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._generate_vector(t) for t in texts]
