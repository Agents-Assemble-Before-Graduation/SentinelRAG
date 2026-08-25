"""Base interface for all retrievers in SentinelRAG."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store with provenance metadata."""

    chunk_id: str
    content: str
    score: float
    document_id: str
    document_title: str
    filename: str
    page_number: int | None = None
    section_heading: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract base class defining the search retrieval contract."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        workspace_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks matching a query.

        Args:
            query: The user query string.
            top_k: The number of candidate chunks to retrieve.
            score_threshold: The minimum score to include a candidate chunk.
            workspace_id: Filter by workspace ID if provided.

        Returns:
            List of RetrievedChunk sorted by relevance score descending.
        """
        pass
