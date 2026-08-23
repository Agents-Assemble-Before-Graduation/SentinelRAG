"""Integration tests for /health and /ready endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(async_client: AsyncClient):
    """Verify /health returns 200 and valid liveness payload."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "app_name" in data
    assert "timestamp" in data
    assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_api_v1_health_endpoint(async_client: AsyncClient):
    """Verify /api/v1/health also works identically."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_ready_endpoint_all_healthy(async_client: AsyncClient):
    """Verify /ready when all dependencies are healthy."""
    with patch("app.api.v1.health.check_database_health", new_callable=AsyncMock) as mock_db, \
         patch("app.api.v1.health.get_vector_store") as mock_get_vs:

        mock_db.return_value = {"status": "healthy", "connected": True, "latency_ms": 1.5}
        mock_vs = AsyncMock()
        mock_vs.health_check.return_value = {
            "status": "healthy",
            "connected": True,
            "latency_ms": 2.1,
            "collections_count": 0,
        }
        mock_get_vs.return_value = mock_vs

        response = await async_client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["components"]["database"]["connected"] is True
        assert data["components"]["vector_store"]["connected"] is True


@pytest.mark.asyncio
async def test_ready_endpoint_unready(async_client: AsyncClient):
    """Verify /ready returns 503 when all dependencies fail."""
    with patch("app.api.v1.health.check_database_health", new_callable=AsyncMock) as mock_db, \
         patch("app.api.v1.health.get_vector_store") as mock_get_vs:

        mock_db.return_value = {"status": "unhealthy", "connected": False, "error": "Connection refused"}
        mock_vs = AsyncMock()
        mock_vs.health_check.return_value = {
            "status": "unhealthy",
            "connected": False,
            "error": "Failed to connect to Qdrant",
        }
        mock_get_vs.return_value = mock_vs

        response = await async_client.get("/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unready"
        assert data["components"]["database"]["connected"] is False
        assert data["components"]["vector_store"]["connected"] is False


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Verify / root metadata endpoint."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SentinelRAG"
    assert data["status"] == "online"
