"""Structured query telemetry for SentinelRAG observability.

Emits a single structured INFO log event at the end of every query execution.
In production (JSON logging mode) this becomes a queryable log entry.
In development it appears as a human-readable structured summary.

Usage::

    telemetry = QueryTelemetry(
        request_id="abc123",
        question_length=42,
        ...
    )
    emit_query_telemetry(telemetry)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryTelemetry:
    """Full observability snapshot for one query execution.

    Every field maps directly to a measurable event property. This is the
    single source of truth for post-query telemetry — rather than spreading
    log calls across multiple nodes, one event is emitted at the end.
    """

    # Identity
    request_id: str = "unknown"

    # Input
    question_length: int = 0
    query_type: str = ""

    # Retrieval
    retrieval_strategy: str = ""
    chunks_retrieved: int = 0
    context_chars: int = 0

    # Memory
    lessons_used: int = 0

    # Agent pipeline
    repair_count: int = 0
    llm_calls: int = 0

    # Outcome
    final_decision: str = ""
    confidence: float = 0.0
    grounded: bool = False

    # Latency breakdown (ms)
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    # Cost
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model_used: str = ""

    # Extra fields for extensibility
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for logging."""
        return asdict(self)


def emit_query_telemetry(telemetry: QueryTelemetry) -> None:
    """Log a structured observability event for one completed query.

    The event is always INFO level with the ``TELEMETRY`` prefix so it
    can be filtered independently from operational logs.

    In JSON logging mode (production), the ``extra`` fields in the log
    record are picked up by the JSONFormatter for structured querying.

    Args:
        telemetry: Fully populated QueryTelemetry snapshot.
    """
    try:
        d = telemetry.to_dict()
        logger.info(
            "[TELEMETRY] request_id=%s query_type=%s strategy=%s "
            "chunks=%d decision=%s confidence=%.2f repairs=%d "
            "latency_ms=%.1f tokens=%d cost_usd=$%.6f lessons=%d",
            d.get("request_id"),
            d.get("query_type"),
            d.get("retrieval_strategy"),
            d.get("chunks_retrieved", 0),
            d.get("final_decision"),
            d.get("confidence", 0.0),
            d.get("repair_count", 0),
            d.get("total_latency_ms", 0.0),
            d.get("total_tokens", 0),
            d.get("estimated_cost_usd", 0.0),
            d.get("lessons_used", 0),
        )
    except Exception as exc:
        logger.warning("[TELEMETRY] Failed to emit telemetry event: %s", str(exc))
