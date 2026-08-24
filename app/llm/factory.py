"""Factory for resolving configured LLM providers."""

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider

logger = get_logger(__name__)

_cached_provider: BaseLLMProvider | None = None


def get_llm_provider(provider_type: str | None = None) -> BaseLLMProvider:
    """Retrieve or instantiate the configured LLM provider singleton.

    Args:
        provider_type: Override the provider type from settings.
                       Supported values: 'openai', 'mock'.
                       If None, reads LLM_PROVIDER from application config.

    Returns:
        Configured BaseLLMProvider instance.
    """
    global _cached_provider

    settings = get_settings()
    selected_type = (provider_type or settings.LLM_PROVIDER).lower()

    # Return cached singleton unless an explicit override is requested
    if _cached_provider is not None and not provider_type:
        return _cached_provider

    provider: BaseLLMProvider

    if selected_type in {"mock", "test", "stub"}:
        from app.llm.mock_provider import MockLLMProvider

        provider = MockLLMProvider()
        logger.info("LLM provider: MockLLMProvider (test/stub mode)")
    else:
        # Default: OpenAI-compatible (covers openai, groq, ollama, etc.)
        from app.llm.openai_provider import OpenAIProvider

        provider = OpenAIProvider()
        if provider.is_available():
            logger.info(
                "LLM provider: OpenAIProvider (model=%s)", provider.model_name
            )
        else:
            logger.warning(
                "LLM provider: OpenAIProvider configured but LLM_API_KEY is missing. "
                "Generation will fail until a valid key is set in .env."
            )

    # Cache the singleton only when no explicit override was given
    if not provider_type:
        _cached_provider = provider

    return provider


def reset_llm_provider_cache() -> None:
    """Clear the cached LLM provider singleton. Intended for testing only."""
    global _cached_provider
    _cached_provider = None
