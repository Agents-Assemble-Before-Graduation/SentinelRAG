"""Unit tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_default_settings():
    """Verify default settings instantiation and types."""
    settings = Settings()
    assert settings.APP_NAME == "SentinelRAG"
    assert settings.APP_VERSION == "0.1.0"
    assert settings.API_PORT == 8000
    assert settings.FRONTEND_PORT == 8501
    assert "postgres" in settings.DATABASE_URL
    assert "6333" in settings.QDRANT_URL


def test_log_level_validation():
    """Verify that only valid log levels are accepted."""
    settings = Settings(LOG_LEVEL="debug")
    assert settings.LOG_LEVEL == "DEBUG"

    with pytest.raises(ValidationError):
        Settings(LOG_LEVEL="INVALID_LOG_LEVEL")


def test_environment_helpers():
    """Verify environment property helper methods."""
    dev_settings = Settings(ENVIRONMENT="development")
    assert dev_settings.is_development is True
    assert dev_settings.is_production is False
    assert dev_settings.is_testing is False

    prod_settings = Settings(ENVIRONMENT="production")
    assert prod_settings.is_development is False
    assert prod_settings.is_production is True
    assert prod_settings.is_testing is False

    test_settings = Settings(ENVIRONMENT="testing")
    assert test_settings.is_testing is True
