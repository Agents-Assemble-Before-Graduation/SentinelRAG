"""Qdrant Vector Store implementation for SentinelRAG."""

import time
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.services.base_vector_store import BaseVectorStore

logger = get_logger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed implementation of BaseVectorStore."""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        settings = get_settings()
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.timeout = timeout or settings.QDRANT_TIMEOUT

        self._client: Optional[AsyncQdrantClient] = None

    @property
    def client(self) -> AsyncQdrantClient:
        """Lazy initializer for AsyncQdrantClient."""
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health by listing collections or pinging the endpoint."""
        start_time = time.perf_counter()
        try:
            collections = await self.client.get_collections()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            collection_count = len(collections.collections) if collections else 0

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "connected": True,
                "collections_count": collection_count,
            }
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning("Qdrant health check failed: %s", str(e))
            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "connected": False,
                "error": str(e),
            }

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a specific vector collection exists in Qdrant."""
        try:
            return await self.client.collection_exists(collection_name=collection_name)
        except Exception as e:
            logger.error("Failed to check if collection '%s' exists: %s", collection_name, str(e))
            raise VectorStoreError(f"Error checking collection existence: {str(e)}") from e

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> bool:
        """Create a new collection with vector configurations."""
        try:
            dist_enum = getattr(qmodels.Distance, distance.upper(), qmodels.Distance.COSINE)
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=dist_enum,
                ),
            )
            logger.info("Created Qdrant collection '%s' with size %d", collection_name, vector_size)
            return True
        except Exception as e:
            logger.error("Failed to create collection '%s': %s", collection_name, str(e))
            raise VectorStoreError(f"Error creating collection: {str(e)}") from e

    async def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> bool:
        """Ensure collection exists, creating it if it does not."""
        exists = await self.collection_exists(collection_name)
        if not exists:
            return await self.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
            )
        return True

    async def upsert_chunks(
        self,
        collection_name: str,
        chunk_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> bool:
        """Insert or update embedded chunk points in Qdrant."""
        if not chunk_ids:
            return True

        if len(chunk_ids) != len(vectors) or len(chunk_ids) != len(payloads):
            raise VectorStoreError("Mismatch between chunk_ids, vectors, and payloads lengths.")

        points = [
            qmodels.PointStruct(
                id=cid,
                vector=vec,
                payload=payload,
            )
            for cid, vec, payload in zip(chunk_ids, vectors, payloads, strict=False)
        ]

        try:
            await self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            logger.info("Upserted %d chunk vectors into collection '%s'", len(points), collection_name)
            return True
        except Exception as e:
            logger.error("Failed to upsert points into '%s': %s", collection_name, str(e))
            raise VectorStoreError(f"Error upserting vectors: {str(e)}") from e

    async def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for top-K similar vector points matching optional filter conditions."""
        query_filter: Optional[qmodels.Filter] = None
        if filter_conditions:
            must_conditions = []
            for key, val in filter_conditions.items():
                if val is not None:
                    must_conditions.append(
                        qmodels.FieldCondition(
                            key=key,
                            match=qmodels.MatchValue(value=str(val)),
                        )
                    )
            if must_conditions:
                query_filter = qmodels.Filter(must=must_conditions)

        try:
            # Check if collection exists first
            if not await self.collection_exists(collection_name):
                logger.warning("Collection '%s' does not exist for search.", collection_name)
                return []

            hits = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )

            results: List[Dict[str, Any]] = []
            for hit in hits:
                results.append(
                    {
                        "id": str(hit.id),
                        "score": float(hit.score),
                        "payload": hit.payload or {},
                    }
                )
            return results
        except Exception as e:
            logger.error("Vector search failed on '%s': %s", collection_name, str(e))
            raise VectorStoreError(f"Error during vector search: {str(e)}") from e

    async def delete_document_chunks(self, collection_name: str, document_id: str) -> bool:
        """Delete all chunk points belonging to a specific document ID."""
        try:
            if not await self.collection_exists(collection_name):
                return True

            await self.client.delete(
                collection_name=collection_name,
                points_selector=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                ),
            )
            logger.info("Deleted chunks for document '%s' in collection '%s'", document_id, collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete chunks for document '%s': %s", document_id, str(e))
            raise VectorStoreError(f"Error deleting document vectors: {str(e)}") from e

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete an existing collection in Qdrant."""
        try:
            if not await self.collection_exists(collection_name):
                return True
            result = await self.client.delete_collection(collection_name=collection_name)
            logger.info("Deleted Qdrant collection '%s'", collection_name)
            return bool(result)
        except Exception as e:
            logger.error("Failed to delete collection '%s': %s", collection_name, str(e))
            raise VectorStoreError(f"Error deleting collection: {str(e)}") from e

    async def close(self) -> None:
        """Close Qdrant client connection cleanly."""
        if self._client is not None:
            await self._client.close()
            self._client = None


_global_vector_store: Optional[QdrantVectorStore] = None


def get_vector_store() -> QdrantVectorStore:
    """Dependency provider for Qdrant Vector Store."""
    global _global_vector_store
    if _global_vector_store is None:
        _global_vector_store = QdrantVectorStore()
    return _global_vector_store
