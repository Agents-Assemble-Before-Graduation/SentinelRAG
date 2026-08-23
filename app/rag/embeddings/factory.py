"""Factory for resolving configured embedding providers."""


from app.core.config import get_settings
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.deterministic import DeterministicEmbeddingProvider
from app.rag.embeddings.fastembed_provider import FastEmbedEmbeddingProvider

_cached_provider: BaseEmbeddingProvider | None = None


def get_embedding_provider(provider_type: str | None = None) -> BaseEmbeddingProvider:
    """Retrieve or instantiate singleton embedding provider based on configuration."""
    global _cached_provider

    settings = get_settings()
    selected_type = (provider_type or settings.EMBEDDING_PROVIDER).lower()

    if _cached_provider is not None and not provider_type:
        return _cached_provider

    if selected_type == "fastembed":
        provider = FastEmbedEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)
    elif selected_type in {"deterministic", "mock", "test"}:
        provider = DeterministicEmbeddingProvider(dimension=settings.EMBEDDING_DIMENSION)
    else:
        # Default fallback to FastEmbed
        provider = FastEmbedEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)

    if not provider_type:
        _cached_provider = provider

    return provider
