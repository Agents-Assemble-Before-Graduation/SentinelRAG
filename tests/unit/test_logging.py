"""Unit tests for structured logging and redaction."""

import json
import logging

from app.core.logging import (
    JSONFormatter,
    get_logger,
    redact_sensitive_data,
    request_id_ctx_var,
    setup_logging,
)


def test_redact_sensitive_data():
    """Verify that passwords, API keys, and bearer tokens are properly redacted."""
    raw_text = '{"api_key": "sk-1234567890abcdef", "password": "supersecretpassword"}'
    redacted = redact_sensitive_data(raw_text)
    assert "sk-1234567890abcdef" not in redacted
    assert "supersecretpassword" not in redacted
    assert "***REDACTED***" in redacted

    bearer_text = "Authorization: Bearer secret-token-xyz"
    redacted_bearer = redact_sensitive_data(bearer_text)
    assert "secret-token-xyz" not in redacted_bearer


def test_request_id_propagation_in_logging():
    """Verify request ID is included in structured log output."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test_logger")
    record = logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "Hello World", (), None)

    # Without request ID
    formatted_no_id = formatter.format(record)
    data_no_id = json.loads(formatted_no_id)
    assert data_no_id["message"] == "Hello World"
    assert "request_id" not in data_no_id

    # With request ID set
    token = request_id_ctx_var.set("req-test-999")
    try:
        formatted_with_id = formatter.format(record)
        data_with_id = json.loads(formatted_with_id)
        assert data_with_id["request_id"] == "req-test-999"
    finally:
        request_id_ctx_var.reset(token)


def test_setup_logging_and_get_logger():
    """Verify setup_logging configures root logger without errors."""
    setup_logging("DEBUG")
    log = get_logger("sentinel_test")
    assert log.name == "sentinel_test"
