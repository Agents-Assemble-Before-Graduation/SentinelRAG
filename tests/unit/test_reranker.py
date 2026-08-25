"""Unit tests for Rerankers (MockReranker and FastEmbedReranker)."""

import pytest
from unittest.mock import MagicMock, patch

from app.rag.retrieval.reranker import MockReranker, FastEmbedReranker, get_reranker, reset_reranker_cache
from app.rag.retrieval.retriever import RetrievedChunk


@pytest.mark.asyncio
async def test_mock_reranker():
    """Verify that MockReranker re-scores and re-orders chunks based on word overlap."""
    reranker = MockReranker()
    
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            content="completely unrelated content",
            score=0.9,
            document_id="d1",
            document_title="Doc A",
            filename="a.pdf"
        ),
        RetrievedChunk(
            chunk_id="c2",
            content="term overlap RAG specification doc",
            score=0.1,
            document_id="d2",
            document_title="Doc B",
            filename="b.pdf"
        )
    ]

    reranked = await reranker.rerank(query="RAG specification", chunks=chunks)

    assert len(reranked) == 2
    # c2 has 2 word overlaps, so it should jump to rank 1
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score > 2.0  # overlap score 2.0 + small base score
    assert reranked[1].chunk_id == "c1"


def test_reranker_factory():
    """Verify that reranker factory returns MockReranker in tests by default."""
    reset_reranker_cache()
    reranker = get_reranker()
    # In test environment settings (ENVIRONMENT=testing), it must default to MockReranker
    assert isinstance(reranker, MockReranker)


@pytest.mark.asyncio
@patch("app.rag.retrieval.reranker.FastEmbedReranker.encoder", new_callable=MagicMock)
async def test_fastembed_reranker_success(mock_encoder):
    """Verify FastEmbedReranker calls the underlying cross-encoder model and updates scores."""
    # Mock fastembed TextCrossEncoder rerank return value
    mock_encoder.rerank.return_value = [0.12, 0.95]
    
    reranker = FastEmbedReranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
    # inject mock encoder
    reranker._encoder = mock_encoder

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            content="chunk 1 content",
            score=0.9,
            document_id="d1",
            document_title="Doc A",
            filename="a.pdf"
        ),
        RetrievedChunk(
            chunk_id="c2",
            content="chunk 2 content",
            score=0.1,
            document_id="d2",
            document_title="Doc B",
            filename="b.pdf"
        )
    ]

    result = await reranker.rerank(query="test query", chunks=chunks)

    assert len(result) == 2
    # c2 score is 0.95 (rank 1), c1 score is 0.12 (rank 2)
    assert result[0].chunk_id == "c2"
    assert result[0].score == 0.95
    assert result[1].chunk_id == "c1"
    assert result[1].score == 0.12
