"""FastAPI query endpoint — POST /api/v1/query."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import LLMProviderError, ValidationError
from app.core.logging import get_logger, request_id_ctx_var
from app.database.session import get_db
from app.services.query_service import QueryResult, RAGQueryService

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["RAG Query"])


# ── Request / Response schemas ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming RAG query payload."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question to answer from indexed documents.",
        examples=["What is retrieval augmented generation?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of document chunks to retrieve.",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for retrieved chunks.",
    )
    workspace_id: uuid.UUID | None = Field(
        default=None,
        description="Optional workspace UUID to scope retrieval.",
    )


class QuerySourceResponse(BaseModel):
    """Source citation for a single retrieved evidence chunk."""

    document_title: str
    filename: str
    page_number: int | None = None
    section_heading: str | None = None
    chunk_index: int
    score: float
    document_id: str


class QueryResponse(BaseModel):
    """Full RAG query response with answer, sources, and telemetry."""

    answer: str
    sources: list[QuerySourceResponse]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model_used: str
    chunks_retrieved: int
    context_chars: int
    request_id: str
    grounded: bool
    tokens_used: int = 0
    metadata: dict[str, Any] = {}


# ── Endpoint ────────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="RAG Query",
    description=(
        "Submit a question. The system retrieves semantically relevant document chunks, "
        "builds a grounded evidence context, and generates a cited answer using the "
        "configured LLM. Retrieved documents are treated as evidence, not instructions."
    ),
)
async def query_documents(
    body: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Execute the RAG pipeline and return a grounded, cited answer."""
    request_id = request_id_ctx_var.get() or "unknown"

    settings = get_settings()
    service = RAGQueryService(
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        max_context_chars=settings.RAG_MAX_CONTEXT_CHARS,
    )

    workspace_id_str = str(body.workspace_id) if body.workspace_id else None

    try:
        result: QueryResult = await service.query(
            question=body.question,
            db=db,
            workspace_id=workspace_id_str,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
        )
    except LLMProviderError as exc:
        logger.error("LLM provider error during query: %s", exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "LLMProviderError",
                "message": exc.message,
                "hint": "Ensure LLM_API_KEY is set in your .env file.",
                "request_id": request_id,
            },
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "ValidationError", "message": exc.message, "request_id": request_id},
        ) from exc

    sources = [
        QuerySourceResponse(
            document_title=s.document_title,
            filename=s.filename,
            page_number=s.page_number,
            section_heading=s.section_heading,
            chunk_index=s.chunk_index,
            score=s.score,
            document_id=s.document_id,
        )
        for s in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
        total_latency_ms=result.total_latency_ms,
        model_used=result.model_used,
        chunks_retrieved=result.chunks_retrieved,
        context_chars=result.context_chars,
        request_id=request_id,
        grounded=result.grounded,
        tokens_used=result.tokens_used,
        metadata=result.metadata,
    )
