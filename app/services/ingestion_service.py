"""Document ingestion coordinator orchestrating validation, extraction, chunking, embedding, and storage."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.exceptions import ResourceNotFoundError, ValidationError
from app.core.logging import get_logger
from app.models.base import DocumentChunkRecord, DocumentRecord, Workspace
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.ingestion.chunker import DocumentChunk, StructureAwareChunker
from app.rag.ingestion.deduplication import calculate_content_hash
from app.rag.ingestion.extractors import ExtractedDocument, get_extractor_for_type
from app.rag.ingestion.validator import validate_document_file
from app.services.vector_store import QdrantVectorStore, get_vector_store

logger = get_logger(__name__)
settings = get_settings()


class DocumentIngestionService:
    """Coordinates end-to-end document processing, deduplication, and hybrid storage."""

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
        vector_store: QdrantVectorStore | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self.vector_store = vector_store or get_vector_store()
        self.chunker = StructureAwareChunker(
            chunk_size=chunk_size or settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP,
            min_chunk_size=min_chunk_size or settings.MIN_CHUNK_SIZE,
        )
        self.collection_name = collection_name or settings.DEFAULT_COLLECTION_NAME

    async def get_or_create_workspace(
        self, db: AsyncSession, workspace_id: uuid.UUID | None = None, name: str = "default"
    ) -> Workspace:
        """Fetch workspace by ID or return/create default workspace."""
        if workspace_id:
            query = select(Workspace).where(Workspace.id == workspace_id)
            result = await db.execute(query)
            ws = result.scalar_one_or_none()
            if not ws:
                raise ResourceNotFoundError(f"Workspace with ID '{workspace_id}' not found.")
            return ws

        # Query default workspace by name
        query = select(Workspace).where(Workspace.name == name)
        result = await db.execute(query)
        ws = result.scalar_one_or_none()
        if not ws:
            ws = Workspace(
                name=name,
                description="Default SentinelRAG workspace",
                is_active=True,
            )
            db.add(ws)
            await db.commit()
            await db.refresh(ws)
            logger.info("Created default workspace '%s' (%s)", ws.name, ws.id)
        return ws

    async def check_duplicate(
        self, db: AsyncSession, workspace_id: uuid.UUID, content_hash: str
    ) -> DocumentRecord | None:
        """Check if a document with the exact content hash already exists in this workspace."""
        query = (
            select(DocumentRecord)
            .where(
                DocumentRecord.workspace_id == workspace_id,
                DocumentRecord.content_hash == content_hash,
                DocumentRecord.status == "indexed",
            )
            .options(selectinload(DocumentRecord.chunks))
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def ingest_document(
        self,
        db: AsyncSession,
        filename: str,
        content: bytes,
        workspace_id: uuid.UUID | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute full ingestion pipeline for a document."""
        # 1. Validation & Source Identification
        clean_filename, source_type = validate_document_file(
            filename=filename,
            content=content,
            max_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
        )
        file_size_bytes = len(content)

        # Resolve Workspace
        workspace = await self.get_or_create_workspace(db, workspace_id)

        # 2. Extraction
        extractor = get_extractor_for_type(source_type)
        extracted_doc: ExtractedDocument = extractor.extract(content, clean_filename)

        if not extracted_doc.full_text.strip():
            raise ValidationError(f"Document '{clean_filename}' contains no readable text.")

        # 3. Deduplication Check
        content_hash = calculate_content_hash(extracted_doc.full_text)
        existing_doc = await self.check_duplicate(db, workspace.id, content_hash)

        if existing_doc:
            logger.info(
                "Duplicate document detected: '%s' matches existing document %s (hash: %s)",
                clean_filename,
                existing_doc.id,
                content_hash,
            )
            created_at_str = (
                existing_doc.created_at.isoformat()
                if existing_doc.created_at
                else datetime.now(UTC).isoformat()
            )
            return {
                "status": "duplicate",
                "document_id": str(existing_doc.id),
                "workspace_id": str(workspace.id),
                "filename": existing_doc.filename,
                "title": existing_doc.title,
                "content_hash": content_hash,
                "chunk_count": existing_doc.chunk_count,
                "message": "Document with identical content already indexed in workspace.",
                "created_at": created_at_str,
            }

        # 4. Structure-aware Chunking
        chunks: list[DocumentChunk] = self.chunker.chunk_document(extracted_doc)
        logger.info(
            "Extracted and chunked '%s' into %d chunks (source_type=%s)",
            clean_filename,
            len(chunks),
            source_type,
        )

        # 5. Embedding Generation
        chunk_texts = [c.content for c in chunks]
        embeddings = await self.embedding_provider.embed_batch(chunk_texts)

        # 6. Database Persistence (PostgreSQL)
        doc_id = uuid.uuid4()
        merged_metadata = {
            **(extracted_doc.metadata or {}),
            **(custom_metadata or {}),
            "embedding_model": self.embedding_provider.model_name,
            "embedding_dimension": self.embedding_provider.dimension,
        }

        doc_record = DocumentRecord(
            id=doc_id,
            workspace_id=workspace.id,
            filename=clean_filename,
            title=extracted_doc.title,
            source_type=source_type,
            content_hash=content_hash,
            file_size_bytes=file_size_bytes,
            chunk_count=len(chunks),
            status="pending",
            metadata_json=merged_metadata,
        )
        db.add(doc_record)

        chunk_records: list[DocumentChunkRecord] = []
        chunk_ids: list[str] = []
        payloads: list[dict[str, Any]] = []

        for _idx, chunk in enumerate(chunks):
            chunk_uuid = uuid.uuid4()
            chunk_id_str = str(chunk_uuid)
            chunk_ids.append(chunk_id_str)

            chunk_record = DocumentChunkRecord(
                id=chunk_uuid,
                document_id=doc_id,
                workspace_id=workspace.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                token_count=chunk.token_count,
                page_number=chunk.page_number,
                section_heading=chunk.section_heading,
                chunk_hash=chunk.chunk_hash,
                metadata_json=chunk.metadata,
            )
            chunk_records.append(chunk_record)

            payloads.append(
                {
                    "chunk_id": chunk_id_str,
                    "document_id": str(doc_id),
                    "workspace_id": str(workspace.id),
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "section_heading": chunk.section_heading,
                    "title": extracted_doc.title,
                    "filename": clean_filename,
                    "source_type": source_type,
                }
            )

        db.add_all(chunk_records)
        doc_record.status = "indexed"
        await db.commit()
        await db.refresh(doc_record)

        # 7. Qdrant Vector Indexing
        try:
            await self.vector_store.ensure_collection(
                collection_name=self.collection_name,
                vector_size=self.embedding_provider.dimension,
            )
            await self.vector_store.upsert_chunks(
                collection_name=self.collection_name,
                chunk_ids=chunk_ids,
                vectors=embeddings,
                payloads=payloads,
            )
        except Exception as e:
            logger.warning("Vector indexing to Qdrant encountered warning/error: %s", str(e))
            # Keep document indexed in relational store, record warning in metadata
            doc_record.error_message = f"Vector indexing partial: {str(e)}"
            await db.commit()

        logger.info(
            "Successfully ingested document '%s' (ID=%s) with %d chunks",
            clean_filename,
            doc_id,
            len(chunks),
        )

        created_at_str = (
            doc_record.created_at.isoformat()
            if doc_record.created_at
            else datetime.now(UTC).isoformat()
        )

        return {
            "status": "indexed",
            "document_id": str(doc_record.id),
            "workspace_id": str(workspace.id),
            "filename": doc_record.filename,
            "title": doc_record.title,
            "source_type": doc_record.source_type,
            "content_hash": content_hash,
            "file_size_bytes": file_size_bytes,
            "chunk_count": len(chunks),
            "created_at": created_at_str,
        }

    async def list_documents(
        self,
        db: AsyncSession,
        workspace_id: uuid.UUID | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[DocumentRecord]:
        """List documents in workspace with pagination."""
        query = select(DocumentRecord)
        if workspace_id:
            query = query.where(DocumentRecord.workspace_id == workspace_id)
        query = query.order_by(DocumentRecord.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_document(
        self, db: AsyncSession, document_id: uuid.UUID
    ) -> DocumentRecord:
        """Get document details with its chunks."""
        query = (
            select(DocumentRecord)
            .where(DocumentRecord.id == document_id)
            .options(selectinload(DocumentRecord.chunks))
        )
        result = await db.execute(query)
        doc = result.scalar_one_or_none()
        if not doc:
            raise ResourceNotFoundError(f"Document '{document_id}' not found.")
        return doc

    async def delete_document(self, db: AsyncSession, document_id: uuid.UUID) -> bool:
        """Delete document from PostgreSQL and corresponding vectors from Qdrant."""
        doc = await self.get_document(db, document_id)

        # 1. Delete vectors from Qdrant
        try:
            await self.vector_store.delete_document_chunks(
                collection_name=self.collection_name,
                document_id=str(document_id),
            )
        except Exception as e:
            logger.warning("Could not delete Qdrant vectors for doc %s: %s", document_id, str(e))

        # 2. Delete from PostgreSQL (chunks cascade-deleted automatically)
        await db.delete(doc)
        await db.commit()
        logger.info("Deleted document %s and associated chunks.", document_id)
        return True
