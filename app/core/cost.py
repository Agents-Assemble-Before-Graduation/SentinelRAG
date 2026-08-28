"""Per-query cost tracking for SentinelRAG.

Tracks token usage and estimated USD cost per agent call and accumulates
totals for the entire query lifecycle. Cost estimates are calculated from a
static price table and are approximate — actual billing depends on provider
invoices.

Usage::

    tracker = CostTracker(provider="openai", model="gpt-4o")
    tracker.record_call("planner", prompt_tokens=200, completion_tokens=50)
    tracker.record_call("generator", prompt_tokens=800, completion_tokens=300)
    print(tracker.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Price table — USD per 1,000 tokens
# Source: OpenAI public pricing page (approximate, update as needed)
# ---------------------------------------------------------------------------
_PRICE_TABLE: dict[tuple[str, str], tuple[float, float]] = {
    # (provider, model): (prompt_cost_per_1k, completion_cost_per_1k)
    ("openai", "gpt-4o"):           (0.005,  0.015),
    ("openai", "gpt-4o-mini"):      (0.00015, 0.0006),
    ("openai", "gpt-4-turbo"):      (0.010,  0.030),
    ("openai", "gpt-3.5-turbo"):    (0.0005, 0.0015),
    ("openai", "gpt-4"):            (0.030,  0.060),
    # Anthropic approximate pricing
    ("anthropic", "claude-3-5-sonnet"): (0.003,  0.015),
    ("anthropic", "claude-3-haiku"):    (0.00025, 0.00125),
    # Mock / local — no cost
    ("mock", "mock-model-1.0"):     (0.0, 0.0),
}

# Fallback pricing when model is not in the table
_DEFAULT_PRICE: tuple[float, float] = (0.001, 0.002)


def _lookup_price(provider: str, model: str) -> tuple[float, float]:
    """Return (prompt_per_1k, completion_per_1k) for the given provider + model."""
    key = (provider.lower(), model.lower())
    if key in _PRICE_TABLE:
        return _PRICE_TABLE[key]
    # Try prefix match (e.g. "gpt-4o-2024-05-13" → "gpt-4o")
    for (p, m), prices in _PRICE_TABLE.items():
        if provider.lower() == p and model.lower().startswith(m):
            return prices
    logger.debug(
        "[CostTracker] Unknown model '%s/%s', using default pricing.", provider, model
    )
    return _DEFAULT_PRICE


@dataclass
class AgentCallRecord:
    """Token and cost record for a single agent LLM call."""
    agent: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """Accumulates token usage and USD cost across agent calls in one query.

    Attributes:
        provider: LLM provider name (e.g. 'openai', 'anthropic').
        model: Model name (e.g. 'gpt-4o').
        calls: Per-agent call records.
        total_prompt_tokens: Cumulative prompt token count.
        total_completion_tokens: Cumulative completion token count.
        total_tokens: Total tokens consumed.
        total_cost_usd: Estimated total cost in USD.
        llm_call_count: Number of LLM calls made.
    """

    provider: str = "unknown"
    model: str = "unknown"
    calls: list[AgentCallRecord] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_call_count: int = 0

    def record_call(
        self,
        agent: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> AgentCallRecord:
        """Record one LLM call from a named agent.

        Args:
            agent: Agent name (e.g. 'planner', 'generator', 'critic').
            prompt_tokens: Number of prompt/input tokens consumed.
            completion_tokens: Number of completion/output tokens consumed.

        Returns:
            The created AgentCallRecord.
        """
        prompt_per_1k, completion_per_1k = _lookup_price(self.provider, self.model)
        call_cost = (
            prompt_tokens / 1000.0 * prompt_per_1k
            + completion_tokens / 1000.0 * completion_per_1k
        )

        record = AgentCallRecord(
            agent=agent,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(call_cost, 6),
        )

        self.calls.append(record)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_cost_usd = round(self.total_cost_usd + call_cost, 6)
        self.llm_call_count += 1

        logger.debug(
            "[CostTracker] agent=%s tokens=%d+%d cost=$%.6f | running_total=$%.6f",
            agent, prompt_tokens, completion_tokens, call_cost, self.total_cost_usd,
        )
        return record

    def summary(self) -> dict[str, Any]:
        """Return a structured cost summary for telemetry and response metadata.

        Returns:
            Dict with provider, model, token counts, cost, and per-agent breakdown.
        """
        per_agent: dict[str, dict[str, Any]] = {}
        for rec in self.calls:
            if rec.agent not in per_agent:
                per_agent[rec.agent] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            per_agent[rec.agent]["calls"] += 1
            per_agent[rec.agent]["prompt_tokens"] += rec.prompt_tokens
            per_agent[rec.agent]["completion_tokens"] += rec.completion_tokens
            per_agent[rec.agent]["total_tokens"] += rec.total_tokens
            per_agent[rec.agent]["cost_usd"] = round(
                per_agent[rec.agent]["cost_usd"] + rec.cost_usd, 6
            )

        return {
            "provider": self.provider,
            "model": self.model,
            "llm_call_count": self.llm_call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.total_cost_usd,
            "per_agent": per_agent,
        }
