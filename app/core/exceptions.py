"""Application custom domain exceptions."""

from typing import Any


class SentinelRAGException(Exception):
    """Base exception for SentinelRAG application."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(SentinelRAGException):
    """Raised when application configuration is invalid or missing required variables."""
    pass


class DatabaseConnectionError(SentinelRAGException):
    """Raised when connection to PostgreSQL database fails."""
    pass


class VectorStoreError(SentinelRAGException):
    """Raised when operations or connection to vector store fail."""
    pass


class ResourceNotFoundError(SentinelRAGException):
    """Raised when requested entity/resource is not found."""
    pass


class ValidationError(SentinelRAGException):
    """Raised when validation fails on business logic or data payload."""
    pass


class LLMProviderError(SentinelRAGException):
    """Raised when an LLM provider fails, is misconfigured, or is unavailable."""
    pass
