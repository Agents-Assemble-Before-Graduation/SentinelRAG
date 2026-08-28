"""Security tests: workspace access isolation.

Verifies that retrieval operations correctly scope queries to the caller's
workspace and that cross-workspace data leakage is prevented at the
vector store query layer.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.rag.retrieval.retriever import DenseRetriever


class TestWorkspaceIsolation:
    """Tests that retrieval always respects workspace_id ownership filters."""

    @pytest.mark.asyncio
    async def test_dense_retrieval_passes_workspace_filter(self):
        """DenseRetriever must include workspace_id in the vector store query filter."""
        workspace_id = str(uuid.uuid4())
        query_vector = [0.1] * 384

        mock_vector_store = AsyncMock()
        mock_vector_store.search_similar.return_value = []

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=query_vector)

        retriever = DenseRetriever(
            embedding_provider=mock_embedder,
            vector_store=mock_vector_store,
        )

        await retriever.retrieve("What is the rate limit?", workspace_id=workspace_id)

        mock_vector_store.search_similar.assert_called_once()
        call_kwargs = mock_vector_store.search_similar.call_args.kwargs
        filter_conditions = call_kwargs.get("filter_conditions") or {}
        assert filter_conditions.get("workspace_id") == workspace_id, (
            f"Expected workspace_id={workspace_id} in filter_conditions, "
            f"got: {filter_conditions}"
        )

    @pytest.mark.asyncio
    async def test_dense_retrieval_no_workspace_passes_no_filter(self):
        """Without a workspace_id, no workspace filter should be applied."""
        query_vector = [0.1] * 384

        mock_vector_store = AsyncMock()
        mock_vector_store.search_similar.return_value = []

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=query_vector)

        retriever = DenseRetriever(
            embedding_provider=mock_embedder,
            vector_store=mock_vector_store,
        )

        await retriever.retrieve("General question", workspace_id=None)

        mock_vector_store.search_similar.assert_called_once()
        call_kwargs = mock_vector_store.search_similar.call_args.kwargs
        # Either no filter_conditions key, or it has no workspace_id
        filter_conditions = call_kwargs.get("filter_conditions") or {}
        assert "workspace_id" not in filter_conditions or filter_conditions.get("workspace_id") is None

    @pytest.mark.asyncio
    async def test_different_workspace_ids_produce_different_filters(self):
        """Two queries with different workspace_ids must use different filter values."""
        ws_a = str(uuid.uuid4())
        ws_b = str(uuid.uuid4())
        query_vector = [0.1] * 384

        mock_vector_store = AsyncMock()
        mock_vector_store.search_similar.return_value = []

        mock_embedder = AsyncMock()
        mock_embedder.embed_text = AsyncMock(return_value=query_vector)

        retriever = DenseRetriever(
            embedding_provider=mock_embedder,
            vector_store=mock_vector_store,
        )

        await retriever.retrieve("Question A", workspace_id=ws_a)
        await retriever.retrieve("Question B", workspace_id=ws_b)

        assert mock_vector_store.search_similar.call_count == 2
        calls = mock_vector_store.search_similar.call_args_list

        filter_a = (calls[0].kwargs.get("filter_conditions") or {}).get("workspace_id")
        filter_b = (calls[1].kwargs.get("filter_conditions") or {}).get("workspace_id")

        assert filter_a == ws_a
        assert filter_b == ws_b
        assert filter_a != filter_b


class TestSecretIsolation:
    """Verifies that secrets are never surfaced in logs or responses."""

    def test_secret_violation_error_is_domain_exception(self):
        from app.core.exceptions import SecurityViolationError, SentinelRAGException
        assert issubclass(SecurityViolationError, SentinelRAGException)

    def test_cost_limit_error_is_domain_exception(self):
        from app.core.exceptions import CostLimitError, SentinelRAGException
        assert issubclass(CostLimitError, SentinelRAGException)

    def test_log_redaction_strips_api_key_patterns(self):
        """Log messages containing API key patterns must be redacted."""
        from app.core.logging import redact_sensitive_data
        message = 'api_key: "sk-abc123def456"'
        redacted = redact_sensitive_data(message)
        assert "sk-abc123def456" not in redacted
        assert "REDACTED" in redacted

    def test_log_redaction_strips_bearer_tokens(self):
        from app.core.logging import redact_sensitive_data
        message = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature"
        redacted = redact_sensitive_data(message)
        assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
        assert "REDACTED" in redacted

    def test_log_redaction_strips_password(self):
        from app.core.logging import redact_sensitive_data
        message = "password: 'mysecretpassword'"
        redacted = redact_sensitive_data(message)
        assert "mysecretpassword" not in redacted
        assert "REDACTED" in redacted

    def test_log_redaction_preserves_normal_message(self):
        from app.core.logging import redact_sensitive_data
        message = "Processing query 'What is the rate limit?' for workspace abc123"
        redacted = redact_sensitive_data(message)
        assert redacted == message
