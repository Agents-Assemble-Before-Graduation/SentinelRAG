"""Unit tests for DenseRetriever."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.rag.retrieval.retriever import DenseRetriever, RetrievedChunk


@pytest.mark.asyncio
async def test_dense_retriever_success():
    """Verify that DenseRetriever calls embedding provider and vector store correctly, parsing raw hits."""
    mock_embedding = AsyncMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]

    mock_vector_store = AsyncMock()
    mock_vector_store.search_similar.return_value = [
        {
            "id": "chunk-uuid-1",
            "score": 0.85,
            "payload": {
                "content": "This is sample retrieved content.",
                "document_id": "doc-uuid-1",
                "title": "Sample Document",
                "filename": "sample.pdf",
                "page_number": 4,
                "section_heading": "Introduction",
                "chunk_index": 0,
                "extra_metadata_field": "val"
            }
        }
    ]

    retriever = DenseRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vector_store,
        collection_name="test_collection"
    )

    chunks = await retriever.retrieve(
        query="test query",
        top_k=3,
        score_threshold=0.5,
        workspace_id="workspace-uuid-1"
    )

    # Assertions
    mock_embedding.embed_text.assert_called_once_with("test query")
    mock_vector_store.search_similar.assert_called_once_with(
        collection_name="test_collection",
        query_vector=[0.1, 0.2, 0.3],
        limit=3,
        score_threshold=0.5,
        filter_conditions={"workspace_id": "workspace-uuid-1"}
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.chunk_id == "chunk-uuid-1"
    assert chunk.content == "This is sample retrieved content."
    assert chunk.score == 0.85
    assert chunk.document_id == "doc-uuid-1"
    assert chunk.document_title == "Sample Document"
    assert chunk.filename == "sample.pdf"
    assert chunk.page_number == 4
    assert chunk.section_heading == "Introduction"
    assert chunk.chunk_index == 0
    assert chunk.metadata == {"extra_metadata_field": "val"}


@pytest.mark.asyncio
async def test_dense_retriever_empty_query():
    """Verify that DenseRetriever returns empty list and does not perform search on empty query."""
    mock_embedding = AsyncMock()
    mock_vector_store = AsyncMock()

    retriever = DenseRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vector_store
    )

    chunks = await retriever.retrieve(query="   ")
    assert chunks == []
    mock_embedding.embed_text.assert_not_called()
    mock_vector_store.search_similar.assert_not_called()
