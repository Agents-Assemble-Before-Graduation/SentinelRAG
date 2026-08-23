"""Integration tests for FastAPI Document CRUD endpoints."""

import io
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.models.base import DocumentChunkRecord, DocumentRecord


@pytest.mark.asyncio
async def test_upload_document_endpoint(async_client: AsyncClient):
    """Verify POST /api/v1/documents successfully uploads and indexes file."""
    mock_ingest_res = {
        "status": "indexed",
        "document_id": str(uuid.uuid4()),
        "workspace_id": str(uuid.uuid4()),
        "filename": "notes.txt",
        "title": "Notes",
        "source_type": "text",
        "content_hash": "samplehash123",
        "file_size_bytes": 100,
        "chunk_count": 2,
        "created_at": "2026-08-23T12:00:00Z",
    }

    with patch("app.api.v1.documents.DocumentIngestionService.ingest_document", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_ingest_res

        files = {"file": ("notes.txt", io.BytesIO(b"Hello world from SentinelRAG test."), "text/plain")}
        response = await async_client.post("/api/v1/documents", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "indexed"
        assert data["filename"] == "notes.txt"
        assert data["chunk_count"] == 2


@pytest.mark.asyncio
async def test_list_documents_endpoint(async_client: AsyncClient):
    """Verify GET /api/v1/documents returns paginated list."""
    doc_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    mock_doc = DocumentRecord(
        id=doc_id,
        workspace_id=ws_id,
        filename="test.md",
        title="Test Document",
        source_type="markdown",
        content_hash="hash123",
        file_size_bytes=200,
        chunk_count=3,
        status="indexed",
    )

    with patch("app.api.v1.documents.DocumentIngestionService.list_documents", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = [mock_doc]

        response = await async_client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["filename"] == "test.md"


@pytest.mark.asyncio
async def test_get_document_details_endpoint(async_client: AsyncClient):
    """Verify GET /api/v1/documents/{id} returns document and chunk details."""
    doc_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    mock_doc = DocumentRecord(
        id=doc_id,
        workspace_id=ws_id,
        filename="paper.pdf",
        title="Sample Paper",
        source_type="pdf",
        content_hash="pdfhash",
        file_size_bytes=1024,
        chunk_count=1,
        status="indexed",
    )
    mock_chunk = DocumentChunkRecord(
        id=chunk_id,
        document_id=doc_id,
        workspace_id=ws_id,
        chunk_index=0,
        content="Abstract text snippet",
        token_count=10,
        page_number=1,
        section_heading="Abstract",
        chunk_hash="chunkhash",
    )
    mock_doc.chunks = [mock_chunk]

    with patch("app.api.v1.documents.DocumentIngestionService.get_document", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_doc

        response = await async_client.get(f"/api/v1/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(doc_id)
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["section_heading"] == "Abstract"


@pytest.mark.asyncio
async def test_delete_document_endpoint(async_client: AsyncClient):
    """Verify DELETE /api/v1/documents/{id} successfully deletes document."""
    doc_id = uuid.uuid4()

    with patch("app.api.v1.documents.DocumentIngestionService.delete_document", new_callable=AsyncMock) as mock_del:
        mock_del.return_value = True

        response = await async_client.delete(f"/api/v1/documents/{doc_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["document_id"] == str(doc_id)
