"""Integration tests for RAGQueryService."""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock

from app.services.query_service import RAGQueryService, QueryResult
from app.rag.embeddings.deterministic import DeterministicEmbeddingProvider
from app.llm.mock_provider import MockLLMProvider
from app.models.base import QueryLog


@pytest.mark.asyncio
async def test_query_service_full_pipeline():
    """Verify full retrieval -> context -> generation RAG orchestration works and records database logs."""
    # Mock vector store returning search hits
    mock_vector_store = AsyncMock()
    mock_vector_store.search_similar.return_value = [
        {
            "id": "chunk-1",
            "score": 0.88,
            "payload": {
                "content": "RAG is retrieval augmented generation.",
                "document_id": str(uuid.uuid4()),
                "title": "RAG Spec",
                "filename": "spec.txt",
                "page_number": 1,
                "section_heading": "Def",
                "chunk_index": 0
            }
        }
    ]

    # Mock DB session
    mock_db = AsyncMock()

    # Instantiate query service with mocks
    service = RAGQueryService(
        embedding_provider=DeterministicEmbeddingProvider(dimension=384),
        vector_store=mock_vector_store,
        llm_provider=MockLLMProvider(response_text="RAG is retrieval augmented generation."),
        collection_name="test_collection",
        top_k=5,
        score_threshold=0.2
    )

    workspace_id = str(uuid.uuid4())
    result = await service.query(
        question="What is RAG?",
        db=mock_db,
        workspace_id=workspace_id,
        top_k=3,
        score_threshold=0.1
    )

    # Telemetry and result verification
    assert isinstance(result, QueryResult)
    assert result.answer == "RAG is retrieval augmented generation."
    assert len(result.sources) == 1
    assert result.sources[0].document_title == "RAG Spec"
    assert result.sources[0].score == 0.88
    assert result.chunks_retrieved == 1
    assert result.total_latency_ms > 0
    assert result.grounded is True

    # Verify vector store call received top_k and threshold overrides and workspace filter
    mock_vector_store.search_similar.assert_called_once_with(
        collection_name="test_collection",
        query_vector=pytest.approx(await DeterministicEmbeddingProvider(dimension=384).embed_text("What is RAG?")),
        limit=3,
        score_threshold=0.1,
        filter_conditions={"workspace_id": workspace_id}
    )

    # Verify database query log was persisted
    assert mock_db.add.called
    added_obj = mock_db.add.call_args[0][0]
    assert isinstance(added_obj, QueryLog)
    assert added_obj.query_text == "What is RAG?"
    assert added_obj.answer_text == "RAG is retrieval augmented generation."
    assert added_obj.workspace_id == uuid.UUID(workspace_id)
    assert mock_db.commit.called
