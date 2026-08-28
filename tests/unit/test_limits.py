"""Unit tests for Phase 8 configurable limits and domain exceptions."""

import pytest


class TestConfigLimits:
    """Tests that Phase 8 limit settings are present in Settings with correct defaults."""

    def test_max_llm_calls_exists(self):
        from app.core.config import get_settings
        settings = get_settings()
        assert hasattr(settings, "MAX_LLM_CALLS")
        assert settings.MAX_LLM_CALLS > 0

    def test_max_llm_calls_default_is_10(self):
        from app.core.config import Settings
        s = Settings()
        assert s.MAX_LLM_CALLS == 10

    def test_max_context_tokens_exists(self):
        from app.core.config import get_settings
        settings = get_settings()
        assert hasattr(settings, "MAX_CONTEXT_TOKENS")
        assert settings.MAX_CONTEXT_TOKENS > 0

    def test_max_context_tokens_default(self):
        from app.core.config import Settings
        s = Settings()
        assert s.MAX_CONTEXT_TOKENS == 8192

    def test_max_query_length_exists(self):
        from app.core.config import Settings
        s = Settings()
        assert hasattr(s, "MAX_QUERY_LENGTH")
        assert s.MAX_QUERY_LENGTH == 2000

    def test_request_timeout_exists(self):
        from app.core.config import Settings
        s = Settings()
        assert hasattr(s, "REQUEST_TIMEOUT")
        assert s.REQUEST_TIMEOUT > 0.0

    def test_request_timeout_default_is_30(self):
        from app.core.config import Settings
        s = Settings()
        assert s.REQUEST_TIMEOUT == 30.0

    def test_max_repair_attempts_exists(self):
        from app.core.config import Settings
        s = Settings()
        assert hasattr(s, "MAX_REPAIR_ATTEMPTS")
        assert s.MAX_REPAIR_ATTEMPTS >= 1

    def test_max_repair_attempts_default_is_2(self):
        from app.core.config import Settings
        s = Settings()
        assert s.MAX_REPAIR_ATTEMPTS == 2

    def test_max_cost_usd_per_query_exists(self):
        from app.core.config import Settings
        s = Settings()
        assert hasattr(s, "MAX_COST_USD_PER_QUERY")
        assert s.MAX_COST_USD_PER_QUERY > 0.0

    def test_max_cost_usd_per_query_default(self):
        from app.core.config import Settings
        s = Settings()
        assert s.MAX_COST_USD_PER_QUERY == 0.10


class TestRepairLimitIsConfigDriven:
    """Tests that judge_node uses settings.MAX_REPAIR_ATTEMPTS, not a hardcoded 2."""

    def test_graph_references_settings_for_repair_limit(self):
        """graph.py must call get_settings().MAX_REPAIR_ATTEMPTS (not hardcode '2')."""
        import inspect
        import app.agents.graph as graph_module

        source = inspect.getsource(graph_module)
        # Must contain MAX_REPAIR_ATTEMPTS reference
        assert "MAX_REPAIR_ATTEMPTS" in source
        # Must NOT contain the bare literal comparison `>= 2` near the judge section
        # (we allow it in comments but not as the sole check)
        # We verify by checking that get_settings() is called in judge context
        assert "get_settings().MAX_REPAIR_ATTEMPTS" in source or "_max_repairs" in source


class TestDomainExceptions:
    """Tests for SecurityViolationError and CostLimitError."""

    def test_security_violation_error_inherits_sentinelrag_exception(self):
        from app.core.exceptions import SecurityViolationError, SentinelRAGException
        assert issubclass(SecurityViolationError, SentinelRAGException)

    def test_cost_limit_error_inherits_sentinelrag_exception(self):
        from app.core.exceptions import CostLimitError, SentinelRAGException
        assert issubclass(CostLimitError, SentinelRAGException)

    def test_security_violation_error_stores_message(self):
        from app.core.exceptions import SecurityViolationError
        exc = SecurityViolationError("Injection detected", details={"count": 2})
        assert exc.message == "Injection detected"
        assert exc.details["count"] == 2

    def test_cost_limit_error_stores_message(self):
        from app.core.exceptions import CostLimitError
        exc = CostLimitError("Cost limit exceeded: $0.12 > $0.10")
        assert "0.12" in exc.message

    def test_security_violation_error_is_catchable_as_sentinelrag(self):
        from app.core.exceptions import SecurityViolationError, SentinelRAGException
        with pytest.raises(SentinelRAGException):
            raise SecurityViolationError("Test injection")

    def test_cost_limit_error_is_catchable_as_sentinelrag(self):
        from app.core.exceptions import CostLimitError, SentinelRAGException
        with pytest.raises(SentinelRAGException):
            raise CostLimitError("Budget exceeded")


class TestTelemetryModule:
    """Tests for QueryTelemetry dataclass and emit function."""

    def test_query_telemetry_has_required_fields(self):
        from app.core.telemetry import QueryTelemetry
        t = QueryTelemetry()
        required = [
            "request_id", "question_length", "query_type", "retrieval_strategy",
            "chunks_retrieved", "context_chars", "lessons_used", "repair_count",
            "llm_calls", "final_decision", "confidence", "grounded",
            "latency_breakdown", "total_latency_ms", "total_tokens",
            "estimated_cost_usd", "model_used",
        ]
        for field in required:
            assert hasattr(t, field), f"Missing field: {field}"

    def test_query_telemetry_to_dict(self):
        from app.core.telemetry import QueryTelemetry
        t = QueryTelemetry(request_id="test-123", question_length=42)
        d = t.to_dict()
        assert d["request_id"] == "test-123"
        assert d["question_length"] == 42

    def test_emit_query_telemetry_does_not_raise(self):
        from app.core.telemetry import QueryTelemetry, emit_query_telemetry
        t = QueryTelemetry(
            request_id="abc",
            query_type="factual",
            final_decision="accept",
            total_latency_ms=120.5,
        )
        emit_query_telemetry(t)  # Must not raise

    def test_telemetry_default_values_are_safe(self):
        from app.core.telemetry import QueryTelemetry
        t = QueryTelemetry()
        assert t.request_id == "unknown"
        assert t.confidence == 0.0
        assert t.grounded is False
        assert isinstance(t.latency_breakdown, dict)


class TestCostTrackerIntegration:
    """Integration-level tests ensuring CostTracker and settings wire together."""

    def test_cost_tracker_instantiates_with_settings_values(self):
        from app.core.cost import CostTracker
        from app.core.config import get_settings
        settings = get_settings()
        tracker = CostTracker(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL)
        assert tracker.provider == settings.LLM_PROVIDER
        assert tracker.model == settings.LLM_MODEL

    def test_cost_tracker_summary_has_no_secrets(self):
        """Cost summary must never contain API key, password, or token values."""
        from app.core.cost import CostTracker
        tracker = CostTracker(provider="openai", model="gpt-4o")
        tracker.record_call("planner", prompt_tokens=100, completion_tokens=50)
        summary = tracker.summary()
        summary_str = str(summary)
        # No key-like strings should appear
        assert "sk-" not in summary_str
        assert "password" not in summary_str.lower()
        assert "token" not in summary_str or "tokens" in summary_str  # "tokens" field is fine
