"""Unit tests for embedding providers and factory."""

import math

import pytest

from app.rag.embeddings.deterministic import DeterministicEmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider


@pytest.mark.asyncio
async def test_deterministic_embedding_provider():
    """Verify deterministic provider vector normalization and consistency."""
    provider = DeterministicEmbeddingProvider(dimension=384)
    assert provider.dimension == 384
    assert provider.model_name == "deterministic-384d"

    text1 = "SentinelRAG multi-agent verification system"
    vec1 = await provider.embed_text(text1)

    assert len(vec1) == 384
    # Verify unit normalization (L2 norm ≈ 1.0)
    norm = math.sqrt(sum(x * x for x in vec1))
    assert pytest.approx(norm, rel=1e-2) == 1.0

    # Verify identical inputs produce identical vectors
    vec1_repeat = await provider.embed_text(text1)
    assert vec1 == vec1_repeat

    # Batch embedding
    batch_vecs = await provider.embed_batch([text1, "Different text snippet"])
    assert len(batch_vecs) == 2
    assert batch_vecs[0] == vec1


def test_embedding_factory():
    """Verify provider instantiation from factory."""
    det_provider = get_embedding_provider("deterministic")
    assert isinstance(det_provider, DeterministicEmbeddingProvider)
