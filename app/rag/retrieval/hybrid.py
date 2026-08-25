"""Hybrid retriever combining dense semantic search and BM25 keyword search."""

import uuid
from typing import Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retrieval.base import BaseRetriever
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.retriever import DenseRetriever, RetrievedChunk

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """Combines Dense semantic search and BM25 keyword search scores."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()

        settings = get_settings()
        self.dense_weight = (
            dense_weight
            if dense_weight is not None
            else settings.RAG_DENSE_WEIGHT
        )
        self.bm25_weight = (
            bm25_weight if bm25_weight is not None else settings.RAG_BM25_WEIGHT
        )

        # Normalize weights so they sum to 1.0
        total_weight = self.dense_weight + self.bm25_weight
        if total_weight > 0.0:
            self.dense_weight /= total_weight
            self.bm25_weight /= total_weight

    def _normalize_scores(
        self, chunks: List[RetrievedChunk]
    ) -> Dict[str, float]:
        """Normalize chunk scores using Min-Max scaling.

        Returns mapping of chunk_id -> normalized score (0.0 to 1.0).
        """
        if not chunks:
            return {}

        scores = [c.score for c in chunks]
        min_score = min(scores)
        max_score = max(scores)
        denom = max_score - min_score

        normalized: Dict[str, float] = {}
        for chunk in chunks:
            if denom == 0.0:
                normalized[chunk.chunk_id] = 1.0
            else:
                normalized[chunk.chunk_id] = (chunk.score - min_score) / denom

        return normalized

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        workspace_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Perform hybrid retrieval combining dense and BM25 results."""
        settings = get_settings()
        effective_top_k = top_k if top_k is not None else settings.RAG_TOP_K

        # Fetch candidate pools from both retrievers
        # Retrieve twice the target top_k from each to ensure a healthy candidate pool for fusion
        candidate_limit = effective_top_k * 2

        dense_chunks = await self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_limit,
            score_threshold=score_threshold,
            workspace_id=workspace_id,
        )

        bm25_chunks = await self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_limit,
            score_threshold=None,  # Do not apply threshold directly to BM25 raw score
            workspace_id=workspace_id,
        )

        if not dense_chunks and not bm25_chunks:
            return []

        # Min-max normalize scores within each pool
        dense_norm = self._normalize_scores(dense_chunks)
        bm25_norm = self._normalize_scores(bm25_chunks)

        # Merge and deduplicate candidates
        unique_chunks: Dict[str, RetrievedChunk] = {}
        for chunk in dense_chunks:
            unique_chunks[chunk.chunk_id] = chunk
        for chunk in bm25_chunks:
            if chunk.chunk_id not in unique_chunks:
                unique_chunks[chunk.chunk_id] = chunk

        # Compute weighted fused score
        fused_chunks: List[RetrievedChunk] = []
        for chunk_id, chunk in unique_chunks.items():
            d_score = dense_norm.get(chunk_id, 0.0)
            b_score = bm25_norm.get(chunk_id, 0.0)

            # Combined weighted score
            fused_score = (self.dense_weight * d_score) + (
                self.bm25_weight * b_score
            )

            # Update score on the RetrievedChunk object
            chunk.score = fused_score
            fused_chunks.append(chunk)

        # Sort by combined score descending and limit to top_k
        fused_chunks.sort(key=lambda x: x.score, reverse=True)
        result_chunks = fused_chunks[:effective_top_k]

        logger.info(
            "Hybrid fusion complete. Merged %d dense and %d BM25 candidates into top %d.",
            len(dense_chunks),
            len(bm25_chunks),
            len(result_chunks),
        )
        return result_chunks
