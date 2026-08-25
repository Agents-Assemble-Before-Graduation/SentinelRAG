"""Unit tests for BM25Retriever and BM25Engine."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.rag.retrieval.bm25 import tokenize, BM25Engine, BM25Retriever
from app.rag.retrieval.retriever import RetrievedChunk


def test_tokenize():
    """Verify tokenizer extracts alphanumeric words and lowercases."""
    assert tokenize("SentinelRAG: Phase 4 retrieval!") == ["sentinelrag", "phase", "4", "retrieval"]
    assert tokenize("") == []


def test_bm25_engine_scoring():
    """Verify BM25Engine calculates exact expected keyword match rankings."""
    corpus = [
        {"chunk_id": "c1", "content": "retrieval augmented generation platform"},
        {"chunk_id": "c2", "content": "dense semantic vector retrieval using Qdrant"},
        {"chunk_id": "c3", "content": "completely unrelated text keyword"},
    ]

    engine = BM25Engine(corpus, k1=1.5, b=0.75)
    hits = engine.search("retrieval platform", top_k=5)

    assert len(hits) == 2
    # Document 1 has both words, so it should rank first
    assert hits[0][0]["chunk_id"] == "c1"
    assert hits[1][0]["chunk_id"] == "c2"
    # Document 3 has no overlapping words, so it shouldn't show up
    assert all(h[0]["chunk_id"] != "c3" for h in hits)


@pytest.mark.asyncio
@patch("app.rag.retrieval.bm25.AsyncSessionLocal")
async def test_bm25_retriever_query(mock_session_cls):
    """Verify BM25Retriever correctly queries database and formats results."""
    mock_session = AsyncMock()
    mock_session_cls.return_value.__aenter__.return_value = mock_session

    # Mock chunk count query
    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 2

    # Mock rows retrieval query
    mock_chunk = AsyncMock()
    mock_chunk.id = "chunk-uuid-1"
    mock_chunk.content = "retrieval query terms"
    mock_chunk.document_id = "doc-uuid-1"
    mock_chunk.workspace_id = "00000000-0000-0000-0000-000000000001"
    mock_chunk.page_number = 2
    mock_chunk.section_heading = "Header"
    mock_chunk.chunk_index = 0
    mock_chunk.metadata_json = {"ext": "val"}

    mock_doc = AsyncMock()
    mock_doc.title = "Source Doc"
    mock_doc.filename = "source.pdf"

    mock_rows_result = MagicMock()
    mock_rows_result.all.return_value = [(mock_chunk, mock_doc)]

    mock_session.execute.side_effect = [mock_count_result, mock_rows_result]

    retriever = BM25Retriever()
    # Reset cache to ensure database is queried
    BM25Retriever._engines_cache.clear()

    chunks = await retriever.retrieve(
        query="retrieval",
        top_k=5,
        score_threshold=0.0,
        workspace_id="00000000-0000-0000-0000-000000000001"
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.chunk_id == "chunk-uuid-1"
    assert chunk.content == "retrieval query terms"
    assert chunk.document_title == "Source Doc"
    assert chunk.filename == "source.pdf"
    assert chunk.page_number == 2
