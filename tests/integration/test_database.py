"""Integration tests for database configuration and abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.database.base import Base
from app.database.session import check_database_health
from app.models import AgentRunRecord, DocumentRecord, QueryLog, Workspace


def test_models_registered_in_metadata():
    """Verify that foundational models are properly mapped in Base.metadata."""
    # Ensure models are referenced
    assert Workspace.__tablename__ == "workspaces"
    assert DocumentRecord.__tablename__ == "documents"
    assert AgentRunRecord.__tablename__ == "agent_runs"
    assert QueryLog.__tablename__ == "query_logs"

    table_names = list(Base.metadata.tables.keys())
    assert "workspaces" in table_names
    assert "documents" in table_names
    assert "agent_runs" in table_names
    assert "query_logs" in table_names


@pytest.mark.asyncio
async def test_database_healthcheck_success():
    """Verify check_database_health returns healthy when query succeeds."""
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1
    mock_conn.execute.return_value = mock_result
    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
    mock_engine.connect.return_value.__aexit__.return_value = None

    with patch("app.database.session.engine", mock_engine):
        res = await check_database_health()
        assert res["status"] == "healthy"
        assert res["connected"] is True
        assert "latency_ms" in res


@pytest.mark.asyncio
async def test_database_healthcheck_failure():
    """Verify check_database_health handles connection failure gracefully."""
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = Exception("Database down")

    with patch("app.database.session.engine", mock_engine):
        res = await check_database_health()
        assert res["status"] == "unhealthy"
        assert res["connected"] is False
        assert "error" in res
