"""FastAPI document ingestion and management endpoints."""

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ResourceNotFoundError, ValidationError
from app.database.session import get_db
from app.services.ingestion_service import DocumentIngestionService

router = APIRouter(prefix="/documents", tags=["Document Ingestion & Management"])


class DocumentChunkSummary(BaseModel):
    """Chunk summary representation."""

    id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    page_number: int | None = None
    section_heading: str | None = None
    chunk_hash: str


class DocumentDetailResponse(BaseModel):
    """Detailed document response with chunk breakdown."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    filename: str
    title: str
    source_type: str
    content_hash: str
    file_size_bytes: int
    chunk_count: int
    status: str
    created_at: str
    metadata_json: dict[str, Any] | None = None
    chunks: list[DocumentChunkSummary] = []


class DocumentListItem(BaseModel):
    """Document overview for list responses."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    filename: str
    title: str
    source_type: str
    content_hash: str
    file_size_bytes: int
    chunk_count: int
    status: str
    created_at: str


class DocumentListResponse(BaseModel):
    """Paginated document list response."""

    total: int
    documents: list[DocumentListItem]


class DocumentIngestResponse(BaseModel):
    """Response returned upon document ingestion or deduplication."""

    status: str = Field(description="'indexed' or 'duplicate'")
    document_id: str
    workspace_id: str
    filename: str
    title: str
    source_type: str | None = None
    content_hash: str
    file_size_bytes: int | None = None
    chunk_count: int
    message: str | None = None
    created_at: str


@router.post(
    "",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest Document",
    description="Validate, extract, chunk, deduplicate, embed, and index a document (PDF, Markdown, TXT, DOCX).",
)
async def upload_and_ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    workspace_id: uuid.UUID | None = Form(default=None, description="Optional workspace UUID"),
    db: AsyncSession = Depends(get_db),
) -> DocumentIngestResponse:
    """Upload and ingest a document into SentinelRAG."""
    ingestion_service = DocumentIngestionService()
    try:
        content = await file.read()
        result = await ingestion_service.ingest_document(
            db=db,
            filename=file.filename or "uploaded_file.txt",
            content=content,
            workspace_id=workspace_id,
        )
        return DocumentIngestResponse(**result)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "ValidationError", "message": e.message},
        ) from e
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ResourceNotFoundError", "message": e.message},
        ) from e


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List Ingested Documents",
    description="Retrieve all indexed documents with optional workspace filtering and pagination.",
)
async def list_documents(
    workspace_id: uuid.UUID | None = Query(default=None, description="Filter by workspace ID"),
    skip: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=50, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """List documents stored in the database."""
    ingestion_service = DocumentIngestionService()
    docs = await ingestion_service.list_documents(
        db=db, workspace_id=workspace_id, skip=skip, limit=limit
    )

    items = [
        DocumentListItem(
            id=d.id,
            workspace_id=d.workspace_id,
            filename=d.filename,
            title=d.title,
            source_type=d.source_type,
            content_hash=d.content_hash,
            file_size_bytes=d.file_size_bytes,
            chunk_count=d.chunk_count,
            status=d.status,
            created_at=d.created_at.isoformat() if d.created_at else datetime.now(UTC).isoformat(),
        )
        for d in docs
    ]
    return DocumentListResponse(total=len(items), documents=items)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get Document Details",
    description="Retrieve document metadata and all constituent chunks.",
)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    """Get single document with its extracted chunks."""
    ingestion_service = DocumentIngestionService()
    try:
        doc = await ingestion_service.get_document(db=db, document_id=document_id)
        chunk_items = [
            DocumentChunkSummary(
                id=c.id,
                chunk_index=c.chunk_index,
                content=c.content,
                token_count=c.token_count,
                page_number=c.page_number,
                section_heading=c.section_heading,
                chunk_hash=c.chunk_hash,
            )
            for c in doc.chunks
        ]
        created_at_str = doc.created_at.isoformat() if doc.created_at else datetime.now(UTC).isoformat()
        return DocumentDetailResponse(
            id=doc.id,
            workspace_id=doc.workspace_id,
            filename=doc.filename,
            title=doc.title,
            source_type=doc.source_type,
            content_hash=doc.content_hash,
            file_size_bytes=doc.file_size_bytes,
            chunk_count=doc.chunk_count,
            status=doc.status,
            created_at=created_at_str,
            metadata_json=doc.metadata_json,
            chunks=chunk_items,
        )
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ResourceNotFoundError", "message": e.message},
        ) from e


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Delete a document, its database chunks, and indexed Qdrant vectors.",
)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Delete document and its vector representations."""
    ingestion_service = DocumentIngestionService()
    try:
        await ingestion_service.delete_document(db=db, document_id=document_id)
        return {
            "status": "deleted",
            "document_id": str(document_id),
            "message": "Document and associated vectors successfully removed.",
        }
    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "ResourceNotFoundError", "message": e.message},
        ) from e
