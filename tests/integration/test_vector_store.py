"""Integration tests for Vector Store abstraction and Qdrant implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.base_vector_store import BaseVectorStore
from app.services.vector_store import QdrantVectorStore


def test_qdrant_vector_store_implements_base_interface():
    """Verify QdrantVectorStore adheres to BaseVectorStore interface."""
    store = QdrantVectorStore()
    assert isinstance(store, BaseVectorStore)
    assert hasattr(store, "health_check")
    assert hasattr(store, "create_collection")
    assert hasattr(store, "delete_collection")
    assert hasattr(store, "collection_exists")
    assert hasattr(store, "close")


@pytest.mark.asyncio
async def test_qdrant_health_check_success():
    """Verify QdrantVectorStore health_check handles successful collection query."""
    store = QdrantVectorStore(url="http://mock-qdrant:6333")
    mock_client = AsyncMock()
    mock_collections = MagicMock()
    mock_collections.collections = ["coll1", "coll2"]
    mock_client.get_collections.return_value = mock_collections

    store._client = mock_client
    res = await store.health_check()

    assert res["status"] == "healthy"
    assert res["connected"] is True
    assert res["collections_count"] == 2
    assert "latency_ms" in res


@pytest.mark.asyncio
async def test_qdrant_health_check_failure():
    """Verify QdrantVectorStore health_check handles offline server gracefully."""
    store = QdrantVectorStore(url="http://mock-qdrant:6333")
    mock_client = AsyncMock()
    mock_client.get_collections.side_effect = Exception("Connection refused")

    store._client = mock_client
    res = await store.health_check()

    assert res["status"] == "unhealthy"
    assert res["connected"] is False
    assert "error" in res


@pytest.mark.asyncio
async def test_qdrant_close():
    """Verify vector store close handles cleanup without errors."""
    store = QdrantVectorStore()
    mock_client = AsyncMock()
    store._client = mock_client
    await store.close()
    assert store._client is None
