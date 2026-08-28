"""Unit tests for Phase 8 CostTracker."""

import pytest
from app.core.cost import CostTracker, AgentCallRecord, _lookup_price, _DEFAULT_PRICE


class TestLookupPrice:
    """Tests for internal price table lookup."""

    def test_known_model_returns_correct_price(self):
        prompt, completion = _lookup_price("openai", "gpt-4o")
        assert prompt == 0.005
        assert completion == 0.015

    def test_case_insensitive_lookup(self):
        p1, c1 = _lookup_price("OpenAI", "GPT-4O")
        p2, c2 = _lookup_price("openai", "gpt-4o")
        assert p1 == p2 and c1 == c2

    def test_unknown_model_returns_default(self):
        price = _lookup_price("openai", "gpt-99-unknown")
        assert price == _DEFAULT_PRICE

    def test_mock_provider_returns_zero_cost(self):
        prompt, completion = _lookup_price("mock", "mock-model-1.0")
        assert prompt == 0.0
        assert completion == 0.0

    def test_prefix_match_works(self):
        """gpt-4o-2024-05-13 should match gpt-4o entry."""
        p, c = _lookup_price("openai", "gpt-4o-2024-05-13")
        # Should match gpt-4o pricing
        assert p == 0.005
        assert c == 0.015


class TestCostTrackerRecordCall:
    """Tests for CostTracker.record_call()."""

    def test_record_call_returns_agent_call_record(self):
        tracker = CostTracker(provider="openai", model="gpt-4o")
        rec = tracker.record_call("planner", prompt_tokens=100, completion_tokens=50)
        assert isinstance(rec, AgentCallRecord)
        assert rec.agent == "planner"
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50
        assert rec.total_tokens == 150

    def test_record_call_accumulates_totals(self):
        tracker = CostTracker(provider="openai", model="gpt-4o")
        tracker.record_call("planner", prompt_tokens=200, completion_tokens=50)
        tracker.record_call("generator", prompt_tokens=800, completion_tokens=300)
        assert tracker.total_prompt_tokens == 1000
        assert tracker.total_completion_tokens == 350
        assert tracker.total_tokens == 1350

    def test_record_call_increments_llm_call_count(self):
        tracker = CostTracker(provider="mock", model="mock-model-1.0")
        assert tracker.llm_call_count == 0
        tracker.record_call("agent1")
        tracker.record_call("agent2")
        assert tracker.llm_call_count == 2

    def test_cost_calculated_from_price_table(self):
        tracker = CostTracker(provider="openai", model="gpt-4o")
        tracker.record_call("generator", prompt_tokens=1000, completion_tokens=1000)
        # 1000 prompt tokens @ $0.005/1k = $0.005
        # 1000 completion tokens @ $0.015/1k = $0.015
        # total = $0.020
        assert abs(tracker.total_cost_usd - 0.020) < 1e-6

    def test_mock_provider_zero_cost(self):
        tracker = CostTracker(provider="mock", model="mock-model-1.0")
        tracker.record_call("generator", prompt_tokens=5000, completion_tokens=2000)
        assert tracker.total_cost_usd == 0.0

    def test_zero_tokens_no_cost(self):
        tracker = CostTracker(provider="openai", model="gpt-4o")
        tracker.record_call("planner", prompt_tokens=0, completion_tokens=0)
        assert tracker.total_cost_usd == 0.0


class TestCostTrackerSummary:
    """Tests for CostTracker.summary()."""

    def test_summary_returns_all_required_keys(self):
        tracker = CostTracker(provider="openai", model="gpt-4o")
        summary = tracker.summary()
        required_keys = {
            "provider", "model", "llm_call_count",
            "total_prompt_tokens", "total_completion_tokens",
            "total_tokens", "estimated_cost_usd", "per_agent",
        }
        assert required_keys.issubset(set(summary.keys()))

    def test_summary_per_agent_breakdown(self):
        tracker = CostTracker(provider="openai", model="gpt-4o")
        tracker.record_call("planner", prompt_tokens=100, completion_tokens=50)
        tracker.record_call("generator", prompt_tokens=500, completion_tokens=200)
        tracker.record_call("generator", prompt_tokens=300, completion_tokens=100)

        summary = tracker.summary()
        per_agent = summary["per_agent"]

        assert "planner" in per_agent
        assert "generator" in per_agent
        assert per_agent["planner"]["calls"] == 1
        assert per_agent["generator"]["calls"] == 2
        assert per_agent["generator"]["prompt_tokens"] == 800
        assert per_agent["generator"]["completion_tokens"] == 300

    def test_summary_reflects_correct_totals(self):
        tracker = CostTracker(provider="openai", model="gpt-4o-mini")
        tracker.record_call("critic", prompt_tokens=200, completion_tokens=80)
        summary = tracker.summary()
        assert summary["total_prompt_tokens"] == 200
        assert summary["total_completion_tokens"] == 80
        assert summary["total_tokens"] == 280
        assert summary["llm_call_count"] == 1

    def test_empty_tracker_summary(self):
        tracker = CostTracker(provider="openai", model="gpt-4o")
        summary = tracker.summary()
        assert summary["llm_call_count"] == 0
        assert summary["total_tokens"] == 0
        assert summary["estimated_cost_usd"] == 0.0
        assert summary["per_agent"] == {}

    def test_summary_provider_and_model_match(self):
        tracker = CostTracker(provider="anthropic", model="claude-3-5-sonnet")
        summary = tracker.summary()
        assert summary["provider"] == "anthropic"
        assert summary["model"] == "claude-3-5-sonnet"
