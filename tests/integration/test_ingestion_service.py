"""Integration tests for DocumentIngestionService."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.base import DocumentRecord, Workspace
from app.rag.embeddings.deterministic import DeterministicEmbeddingProvider
from app.services.ingestion_service import DocumentIngestionService


@pytest.mark.asyncio
async def test_ingest_markdown_document():
    """Verify full ingestion pipeline for markdown content."""
    mock_db = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.ensure_collection.return_value = True
    mock_vector_store.upsert_chunks.return_value = True

    embedding_provider = DeterministicEmbeddingProvider(dimension=384)
    service = DocumentIngestionService(
        embedding_provider=embedding_provider,
        vector_store=mock_vector_store,
        chunk_size=150,
        chunk_overlap=30,
        min_chunk_size=20,
    )

    workspace_id = uuid.uuid4()
    workspace = Workspace(id=workspace_id, name="test_ws")

    # Mock DB queries: get workspace -> return workspace, check duplicate -> return None
    mock_db.execute.side_effect = [
        MagicMock(scalar_one_or_none=MagicMock(return_value=workspace)),  # get workspace
        MagicMock(scalar_one_or_none=MagicMock(return_value=None)),       # check duplicate
    ]

    md_content = b"""# Research Paper
Abstract section explaining SentinelRAG.

## Section 1: Ingestion
Details about chunking and embedding.
"""

    result = await service.ingest_document(
        db=mock_db,
        filename="test_paper.md",
        content=md_content,
        workspace_id=workspace_id,
    )

    assert result["status"] == "indexed"
    assert result["filename"] == "test_paper.md"
    assert result["source_type"] == "markdown"
    assert result["chunk_count"] >= 2
    assert mock_vector_store.upsert_chunks.called


@pytest.mark.asyncio
async def test_duplicate_document_detection():
    """Verify ingestion service identifies identical duplicate document."""
    mock_db = AsyncMock()
    mock_vector_store = AsyncMock()

    service = DocumentIngestionService(
        embedding_provider=DeterministicEmbeddingProvider(dimension=384),
        vector_store=mock_vector_store,
    )

    workspace_id = uuid.uuid4()
    workspace = Workspace(id=workspace_id, name="test_ws")
    existing_doc = DocumentRecord(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        filename="existing.txt",
        title="Existing",
        source_type="text",
        content_hash="abc123hash",
        chunk_count=3,
        status="indexed",
    )

    # Return workspace, then return existing_doc on duplicate check
    mock_db.execute.side_effect = [
        MagicMock(scalar_one_or_none=MagicMock(return_value=workspace)),
        MagicMock(scalar_one_or_none=MagicMock(return_value=existing_doc)),
    ]

    result = await service.ingest_document(
        db=mock_db,
        filename="duplicate_copy.txt",
        content=b"Sample duplicate text content for hashing",
        workspace_id=workspace_id,
    )

    assert result["status"] == "duplicate"
    assert result["document_id"] == str(existing_doc.id)
    assert not mock_vector_store.upsert_chunks.called


@pytest.mark.asyncio
async def test_delete_document():
    """Verify document deletion removes records and triggers vector deletion."""
    mock_db = AsyncMock()
    mock_vector_store = AsyncMock()
    mock_vector_store.delete_document_chunks.return_value = True

    service = DocumentIngestionService(
        embedding_provider=DeterministicEmbeddingProvider(dimension=384),
        vector_store=mock_vector_store,
    )

    doc_id = uuid.uuid4()
    doc_record = DocumentRecord(id=doc_id, filename="to_delete.txt", title="Delete Me", source_type="text", content_hash="hash")
    mock_db.execute.return_value = MagicMock(scalar_one_or_none=MagicMock(return_value=doc_record))

    success = await service.delete_document(db=mock_db, document_id=doc_id)
    assert success is True
    assert mock_vector_store.delete_document_chunks.called
    assert mock_db.delete.called
    assert mock_db.commit.called
