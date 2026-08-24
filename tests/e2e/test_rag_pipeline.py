"""End-to-end simulation of the RAG pipeline from document ingestion to cited query generation."""

import pytest
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from app.models.base import Workspace, DocumentRecord, DocumentChunkRecord
from app.rag.embeddings.deterministic import DeterministicEmbeddingProvider
from app.llm.mock_provider import MockLLMProvider
from app.services.ingestion_service import DocumentIngestionService
from app.services.query_service import RAGQueryService, QueryResult
from app.services.base_vector_store import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for testing end-to-end ingestion and query pipeline."""

    def __init__(self) -> None:
        self.collections: Dict[str, List[Dict[str, Any]]] = {}

    async def health_check(self) -> Dict[str, Any]:
        return {"status": "healthy", "connected": True}

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> bool:
        self.collections[collection_name] = []
        return True

    async def ensure_collection(self, collection_name: str, vector_size: int, distance: str = "Cosine") -> bool:
        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name, vector_size, distance)
        return True

    async def upsert_chunks(self, collection_name: str, chunk_ids: List[str], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> bool:
        for cid, vec, payload in zip(chunk_ids, vectors, payloads):
            self.collections[collection_name].append({
                "id": cid,
                "vector": vec,
                "payload": payload
            })
        return True

    async def search_similar(self, collection_name: str, query_vector: List[float], limit: int = 5, score_threshold: Optional[float] = None, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if collection_name not in self.collections:
            return []
        
        # Simple dot product / cosine similarity approximation for deterministic vectors
        results = []
        for point in self.collections[collection_name]:
            # Apply filter conditions if any
            if filter_conditions:
                match = True
                for k, v in filter_conditions.items():
                    if point["payload"].get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Calculate dot product as score
            score = sum(a * b for a, b in zip(query_vector, point["vector"]))
            
            # Since vectors from DeterministicEmbeddingProvider are normalized,
            # dot product is the cosine similarity.
            if score_threshold is not None and score < score_threshold:
                continue

            results.append({
                "id": point["id"],
                "score": score,
                "payload": point["payload"]
            })

        # Sort by score desc and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def delete_document_chunks(self, collection_name: str, document_id: str) -> bool:
        if collection_name in self.collections:
            self.collections[collection_name] = [
                pt for pt in self.collections[collection_name]
                if pt["payload"].get("document_id") != document_id
            ]
        return True

    async def delete_collection(self, collection_name: str) -> bool:
        self.collections.pop(collection_name, None)
        return True

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_e2e_document_ingestion_and_query_flow():
    """Verify end-to-end ingestion and querying with source-grounded answering works correctly."""
    # Setup shared in-memory vector store and deterministic embedding provider
    vector_store = InMemoryVectorStore()
    embedding_provider = DeterministicEmbeddingProvider(dimension=384)
    llm_provider = MockLLMProvider(
        response_text="Based on the specification, SentinelRAG is a local-first multi-agent system. [Evidence 1].",
        model="gpt-mock-e2e"
    )

    # Initialize services
    ingestion_service = DocumentIngestionService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chunk_size=150,
        chunk_overlap=20,
        min_chunk_size=10
    )

    query_service = RAGQueryService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider,
        top_k=3,
        score_threshold=0.1
    )

    # Mock DB Session
    mock_db = AsyncMock()
    workspace_id = uuid.uuid4()
    workspace = Workspace(id=workspace_id, name="e2e_test_workspace")

    # DB mocks for ingestion: return workspace, return duplicate=None on query, return none on commit
    mock_db.execute.side_effect = [
        MagicMock(scalar_one_or_none=MagicMock(return_value=workspace)),  # get workspace
        MagicMock(scalar_one_or_none=MagicMock(return_value=None)),       # check duplicate
    ]

    # Ingest document
    doc_content = b"""# SentinelRAG System Specification
SentinelRAG is a local-first multi-agent system designed for robust evidence verification.
It uses a Postgres database and Qdrant vector store.
"""
    ingest_res = await ingestion_service.ingest_document(
        db=mock_db,
        filename="sys_spec.md",
        content=doc_content,
        workspace_id=workspace_id
    )

    assert ingest_res["status"] == "indexed"
    assert ingest_res["chunk_count"] >= 1
    doc_id = ingest_res["document_id"]

    # Verify document chunks were upserted to in-memory vector store
    assert "sentinel_chunks" in vector_store.collections
    assert len(vector_store.collections["sentinel_chunks"]) == ingest_res["chunk_count"]

    # Query the RAG Pipeline
    # Direct query flow: Query RAG using same workspace ID
    query_res = await query_service.query(
        question="What is SentinelRAG?",
        db=mock_db,
        workspace_id=str(workspace_id),
        top_k=2,
        score_threshold=-1.0
    )

    # Verify query response and citations
    assert isinstance(query_res, QueryResult)
    assert "SentinelRAG is a local-first multi-agent system" in query_res.answer
    assert "[Evidence 1]" in query_res.answer
    assert len(query_res.sources) >= 1
    assert query_res.sources[0].document_title == "SentinelRAG System Specification"
    assert query_res.sources[0].filename == "sys_spec.md"
    assert query_res.sources[0].score > -1.0
    assert query_res.grounded is True
    assert query_res.model_used == "gpt-mock-e2e"
