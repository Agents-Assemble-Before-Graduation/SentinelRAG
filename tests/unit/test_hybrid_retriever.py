"""Unit tests for HybridRetriever."""

import pytest
from unittest.mock import AsyncMock

from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.retriever import RetrievedChunk


@pytest.mark.asyncio
async def test_hybrid_fusion_scoring():
    """Verify that score normalisation and weighted linear combination work correctly."""
    mock_dense = AsyncMock()
    mock_dense.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            content="overlap chunk content",
            score=0.8,  # max dense
            document_id="d1",
            document_title="Doc A",
            filename="a.pdf"
        ),
        RetrievedChunk(
            chunk_id="c2",
            content="dense only chunk",
            score=0.4,  # min dense
            document_id="d2",
            document_title="Doc B",
            filename="b.pdf"
        )
    ]

    mock_bm25 = AsyncMock()
    mock_bm25.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c3",
            content="bm25 only chunk",
            score=20.0,  # max bm25
            document_id="d3",
            document_title="Doc C",
            filename="c.pdf"
        ),
        RetrievedChunk(
            chunk_id="c1",
            content="overlap chunk content",
            score=10.0,  # min bm25
            document_id="d1",
            document_title="Doc A",
            filename="a.pdf"
        )
    ]

    # Initialize HybridRetriever with 0.5 weights
    retriever = HybridRetriever(
        dense_retriever=mock_dense,
        bm25_retriever=mock_bm25,
        dense_weight=0.5,
        bm25_weight=0.5
    )

    results = await retriever.retrieve(query="overlap test", top_k=3)

    # Assertions
    assert len(results) == 3

    # Verification of normalized calculations:
    # pool dense:
    # c1 score 0.8 (norm = 1.0)
    # c2 score 0.4 (norm = 0.0)
    # pool bm25:
    # c3 score 20.0 (norm = 1.0)
    # c1 score 10.0 (norm = 0.0)
    # Fused scores:
    # c1: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # c2: 0.5 * 0.0 + 0.5 * 0.0 = 0.0
    # c3: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    
    # Expected order: c1 and c3 tied at first rank with 0.5, c2 at third rank with 0.0
    assert results[0].chunk_id in {"c1", "c3"}
    assert results[1].chunk_id in {"c1", "c3"}
    assert results[2].chunk_id == "c2"

    assert results[2].score == 0.0
    # verify that chunks are unique
    assert len(set(c.chunk_id for c in results)) == 3
