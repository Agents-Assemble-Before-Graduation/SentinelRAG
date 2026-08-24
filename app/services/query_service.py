"""RAG query orchestration service: retrieval → context → generation → audit log."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider
from app.llm.factory import get_llm_provider
from app.models.base import QueryLog
from app.rag.context.builder import BuiltContext, ContextBuilder, SourceCitation
from app.rag.embeddings.base import BaseEmbeddingProvider
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.generation.generator import GenerationResult, RAGGenerator
from app.rag.retrieval.retriever import DenseRetriever, RetrievedChunk
from app.services.vector_store import QdrantVectorStore, get_vector_store

logger = get_logger(__name__)


@dataclass
class QuerySource:
    """Serialisable source citation for API responses."""

    document_title: str
    filename: str
    page_number: int | None
    section_heading: str | None
    chunk_index: int
    score: float
    document_id: str

    @classmethod
    def from_citation(cls, citation: SourceCitation) -> "QuerySource":
        return cls(
            document_title=citation.document_title,
            filename=citation.filename,
            page_number=citation.page_number,
            section_heading=citation.section_heading,
            chunk_index=citation.chunk_index,
            score=round(citation.score, 4),
            document_id=citation.document_id,
        )


@dataclass
class QueryResult:
    """Full result of a RAG query including answer, sources, and telemetry."""

    answer: str
    sources: list[QuerySource]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    model_used: str
    chunks_retrieved: int
    context_chars: int
    grounded: bool
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGQueryService:
    """Orchestrates the full RAG pipeline for a single user query.

    Pipeline:
        embed query → dense retrieve → build context → generate → log → return

    All components are injectable for testing.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
        vector_store: QdrantVectorStore | None = None,
        llm_provider: BaseLLMProvider | None = None,
        collection_name: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        max_context_chars: int | None = None,
    ) -> None:
        settings = get_settings()

        _embedding = embedding_provider or get_embedding_provider()
        _vector_store = vector_store or get_vector_store()
        _llm = llm_provider or get_llm_provider()
        _collection = collection_name or settings.DEFAULT_COLLECTION_NAME

        self._top_k = top_k if top_k is not None else settings.RAG_TOP_K
        self._score_threshold = (
            score_threshold if score_threshold is not None else settings.RAG_SCORE_THRESHOLD
        )
        self._max_context_chars = max_context_chars or settings.RAG_MAX_CONTEXT_CHARS

        self._retriever = DenseRetriever(
            embedding_provider=_embedding,
            vector_store=_vector_store,
            collection_name=_collection,
        )
        self._context_builder = ContextBuilder(max_context_chars=self._max_context_chars)
        self._generator = RAGGenerator(llm_provider=_llm)

    async def query(
        self,
        question: str,
        db: AsyncSession | None = None,
        workspace_id: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> QueryResult:
        """Execute the full RAG pipeline for a user question.

        Args:
            question: Raw user question string.
            db: Optional AsyncSession for persisting QueryLog.
            workspace_id: Optional workspace UUID string for scoped retrieval.
            top_k: Per-request override for number of chunks to retrieve.
            score_threshold: Per-request override for similarity threshold.

        Returns:
            QueryResult with answer, source citations, and latency breakdown.

        Raises:
            LLMProviderError: If the LLM provider is unavailable.
        """
        pipeline_start = time.perf_counter()

        effective_top_k = top_k if top_k is not None else self._top_k
        effective_threshold = (
            score_threshold if score_threshold is not None else self._score_threshold
        )

        # ── Step 1: Dense retrieval ──────────────────────────────────────────
        retrieval_start = time.perf_counter()
        chunks: list[RetrievedChunk] = await self._retriever.retrieve(
            query=question,
            top_k=effective_top_k,
            score_threshold=effective_threshold,
            workspace_id=workspace_id,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        # ── Step 2: Context assembly ──────────────────────────────────────────
        context: BuiltContext = self._context_builder.build(chunks)

        # ── Step 3: LLM generation ────────────────────────────────────────────
        generation_start = time.perf_counter()
        result: GenerationResult = await self._generator.generate(
            question=question,
            context=context,
        )
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000

        # ── Step 4: Audit log ─────────────────────────────────────────────────
        if db is not None:
            await self._persist_query_log(
                db=db,
                workspace_id=workspace_id,
                question=question,
                answer=result.answer,
                latency_ms=total_latency_ms,
            )

        # ── Step 5: Build response ────────────────────────────────────────────
        sources = [QuerySource.from_citation(c) for c in result.sources]

        return QueryResult(
            answer=result.answer,
            sources=sources,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            model_used=result.model_used,
            chunks_retrieved=len(chunks),
            context_chars=context.total_chars,
            grounded=result.grounded,
            tokens_used=result.tokens_used,
            metadata={
                "context_truncated": context.was_truncated,
                "finish_reason": result.finish_reason,
            },
        )

    async def _persist_query_log(
        self,
        db: AsyncSession,
        workspace_id: str | None,
        question: str,
        answer: str,
        latency_ms: float,
    ) -> None:
        """Write a QueryLog record to PostgreSQL for telemetry and auditing."""
        try:
            ws_uuid: uuid.UUID | None = None
            if workspace_id:
                ws_uuid = uuid.UUID(workspace_id)

            log_entry = QueryLog(
                workspace_id=ws_uuid,
                query_text=question,
                answer_text=answer,
                is_killed=False,
                latency_ms=round(latency_ms, 2),
            )
            db.add(log_entry)
            await db.commit()
        except Exception as exc:
            logger.warning("Failed to persist query log: %s", str(exc))
            # Never let audit logging crash the query response
            try:
                await db.rollback()
            except Exception:
                pass
