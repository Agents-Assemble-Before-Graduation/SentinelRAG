"""Centralized structured logging module for SentinelRAG."""

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

# Context variable for request ID propagation across async calls
request_id_ctx_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Pattern for redacting sensitive values like API keys, tokens, passwords
SENSITIVE_KEY_VALUE_PATTERN = re.compile(
    r'(["\']?(?:api[_-]?key|password|secret|token|authorization)["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])',
    re.IGNORECASE,
)
BEARER_TOKEN_PATTERN = re.compile(r'(Bearer\s+)[A-Za-z0-9_\-\.]+', re.IGNORECASE)


def redact_sensitive_data(message: str) -> str:
    """Redact sensitive patterns from log strings."""
    redacted = SENSITIVE_KEY_VALUE_PATTERN.sub(r'\1***REDACTED***\3', message)
    redacted = BEARER_TOKEN_PATTERN.sub(r'\1***REDACTED***', redacted)
    return redacted


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production and structured environments."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = request_id_ctx_var.get()
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_data(record.getMessage()),
            "module": record.module,
            "line": record.lineno,
        }

        if req_id:
            log_obj["request_id"] = req_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class StandardFormatter(logging.Formatter):
    """Clean readable formatter for local development console."""

    def format(self, record: logging.LogRecord) -> str:
        req_id = request_id_ctx_var.get()
        req_part = f" [{req_id}]" if req_id else ""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = redact_sensitive_data(record.getMessage())
        formatted = f"{timestamp} | {record.levelname:<8} | {record.name}:{record.lineno}{req_part} - {msg}"
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        return formatted


def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Configure root and application loggers."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(StandardFormatter())

    root_logger.addHandler(handler)

    # Suppress verbose noisy logs from 3rd party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
