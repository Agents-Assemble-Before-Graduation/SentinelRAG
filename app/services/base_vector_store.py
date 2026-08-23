"""Abstract Base Class for Vector Store implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVectorStore(ABC):
    """Abstract vector store interface isolating persistence engine details."""

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check vector store connectivity and operational readiness."""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a specific vector collection exists."""
        pass

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> bool:
        """Create a new collection with the given vector dimensions and distance metric."""
        pass

    @abstractmethod
    async def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> bool:
        """Ensure collection exists, creating it if it does not."""
        pass

    @abstractmethod
    async def upsert_chunks(
        self,
        collection_name: str,
        chunk_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> bool:
        """Insert or update embedded chunk points with associated payload metadata."""
        pass

    @abstractmethod
    async def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for top-K similar vector points matching optional filter conditions."""
        pass

    @abstractmethod
    async def delete_document_chunks(self, collection_name: str, document_id: str) -> bool:
        """Delete all chunk points belonging to a specific document ID."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete an existing collection by name."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close vector store client connections."""
        pass
