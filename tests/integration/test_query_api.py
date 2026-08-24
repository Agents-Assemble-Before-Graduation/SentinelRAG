"""Integration tests for FastAPI query endpoint."""

import pytest
import uuid
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient

from app.core.exceptions import LLMProviderError
from app.services.query_service import QueryResult, QuerySource


@pytest.mark.asyncio
async def test_query_documents_endpoint_success(async_client: AsyncClient):
    """Verify POST /api/v1/query returns 200 with grounded answer and citations."""
    ws_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    
    mock_source = QuerySource(
        document_title="Sample Document",
        filename="sample.pdf",
        page_number=3,
        section_heading="Background",
        chunk_index=12,
        score=0.88,
        document_id=str(doc_id)
    )
    
    mock_result = QueryResult(
        answer="According to the spec, SentinelRAG is active.",
        sources=[mock_source],
        retrieval_latency_ms=15.5,
        generation_latency_ms=250.0,
        total_latency_ms=265.5,
        model_used="gpt-4o",
        chunks_retrieved=1,
        context_chars=150,
        grounded=True,
        tokens_used=120,
        metadata={"finish_reason": "stop"}
    )

    with patch("app.api.v1.query.RAGQueryService.query", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_result

        payload = {
            "question": "What is SentinelRAG?",
            "top_k": 5,
            "score_threshold": 0.3,
            "workspace_id": str(ws_id)
        }
        
        response = await async_client.post("/api/v1/query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "According to the spec, SentinelRAG is active."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_title"] == "Sample Document"
        assert data["sources"][0]["page_number"] == 3
        assert data["sources"][0]["score"] == 0.88
        assert data["retrieval_latency_ms"] == 15.5
        assert data["generation_latency_ms"] == 250.0
        assert data["total_latency_ms"] == 265.5
        assert data["model_used"] == "gpt-4o"
        assert "request_id" in data


@pytest.mark.asyncio
async def test_query_documents_endpoint_invalid_payload(async_client: AsyncClient):
    """Verify POST /api/v1/query validates question length / presence (422 error)."""
    # Empty question
    response = await async_client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_documents_endpoint_llm_error(async_client: AsyncClient):
    """Verify POST /api/v1/query handles LLM provider errors gracefully returning 503."""
    with patch("app.api.v1.query.RAGQueryService.query", new_callable=AsyncMock) as mock_query:
        mock_query.side_effect = LLMProviderError("OpenAI API is down or invalid key.")
        
        payload = {"question": "What is RAG?"}
        response = await async_client.post("/api/v1/query", json=payload)
        
        assert response.status_code == 503
        data = response.json()
        assert "LLMProviderError" in data["detail"]["error"]
        assert "OpenAI API is down" in data["detail"]["message"]
        assert "request_id" in data["detail"]
