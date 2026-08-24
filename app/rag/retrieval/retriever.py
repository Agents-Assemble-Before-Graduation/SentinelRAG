"""Dense semantic retriever backed by Qdrant vector search."""

from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider
from app.services.vector_store import QdrantVectorStore, get_vector_store

logger = get_logger(__name__)


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


class DenseRetriever:
    """Semantic retrieval using dense vector embeddings and Qdrant ANN search.

    Embeds the incoming query with the configured embedding provider,
    issues an ANN search against the Qdrant collection, and returns
    typed ``RetrievedChunk`` objects enriched with document provenance.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
        vector_store: QdrantVectorStore | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider or get_embedding_provider()
        self._vector_store = vector_store or get_vector_store()
        settings = get_settings()
        self._collection_name = collection_name or settings.DEFAULT_COLLECTION_NAME

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        workspace_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the query and retrieve the most semantically similar chunks.

        Args:
            query: The raw user question / search string.
            top_k: Maximum number of chunks to return. Defaults to RAG_TOP_K config.
            score_threshold: Minimum cosine similarity score. Defaults to RAG_SCORE_THRESHOLD.
            workspace_id: Optional UUID string to restrict retrieval to one workspace.

        Returns:
            Ordered list of RetrievedChunk (highest score first).
        """
        settings = get_settings()
        effective_top_k = top_k if top_k is not None else settings.RAG_TOP_K
        effective_threshold = (
            score_threshold if score_threshold is not None else settings.RAG_SCORE_THRESHOLD
        )

        if not query or not query.strip():
            logger.warning("DenseRetriever received an empty query; returning empty results.")
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_text(query.strip())
        logger.debug(
            "Query embedded (dim=%d) for retrieval (top_k=%d, threshold=%.2f)",
            len(query_vector),
            effective_top_k,
            effective_threshold,
        )

        # Build optional workspace filter
        filter_conditions: dict[str, Any] | None = None
        if workspace_id:
            filter_conditions = {"workspace_id": workspace_id}

        # Search vector store
        raw_hits = await self._vector_store.search_similar(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=effective_top_k,
            score_threshold=effective_threshold,
            filter_conditions=filter_conditions,
        )

        if not raw_hits:
            logger.info("No chunks retrieved for query (collection='%s')", self._collection_name)
            return []

        # Map raw Qdrant hits to typed RetrievedChunk objects
        chunks: list[RetrievedChunk] = []
        for hit in raw_hits:
            payload = hit.get("payload", {})
            chunk = RetrievedChunk(
                chunk_id=str(hit.get("id", "")),
                content=payload.get("content", ""),
                score=float(hit.get("score", 0.0)),
                document_id=str(payload.get("document_id", "")),
                document_title=payload.get("title", payload.get("document_title", "Unknown")),
                filename=payload.get("filename", ""),
                page_number=payload.get("page_number"),
                section_heading=payload.get("section_heading"),
                chunk_index=int(payload.get("chunk_index", 0)),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in {
                        "content", "document_id", "workspace_id",
                        "page_number", "section_heading", "chunk_index",
                        "title", "document_title", "filename",
                    }
                },
            )
            chunks.append(chunk)

        logger.info(
            "Retrieved %d chunk(s) for query (top score=%.4f)",
            len(chunks),
            chunks[0].score if chunks else 0.0,
        )
        return chunks
