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
from app.rag.retrieval import DenseRetriever, BM25Retriever, HybridRetriever, RetrievedChunk, get_reranker
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
        self._retrawal_mode = settings.RAG_RETRIEVAL_MODE
        self._rerank_enabled = settings.RAG_RERANK_ENABLED

        # Instantiate all retrievers for runtime flexibility
        self._dense_retriever = DenseRetriever(
            embedding_provider=_embedding,
            vector_store=_vector_store,
            collection_name=_collection,
        )
        self._bm25_retriever = BM25Retriever()
        self._hybrid_retriever = HybridRetriever(
            dense_retriever=self._dense_retriever,
            bm25_retriever=self._bm25_retriever,
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
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
    ) -> QueryResult:
        """Execute the full LangGraph multi-agent RAG pipeline for a user question.

        Args:
            question: Raw user question string.
            db: Optional AsyncSession for persisting QueryLog.
            workspace_id: Optional workspace UUID string for scoped retrieval.
            top_k: Per-request override for number of chunks to retrieve.
            score_threshold: Per-request override for similarity threshold.
            retrieval_mode: 'dense', 'bm25', or 'hybrid'.
            rerank_enabled: Boolean flag to enable/disable cross-encoder reranking.

        Returns:
            QueryResult with answer, source citations, and latency breakdown.

        Raises:
            LLMProviderError: If the LLM provider is unavailable.
        """
        pipeline_start = time.perf_counter()

        # Initialize LangGraph state
        initial_state = {
            "question": question,
            "original_question": question,
            "workspace_id": workspace_id,
            "top_k_override": None,
            "query_type": "",
            "plan": "",
            "subquestions": [],
            "retrieval_strategy": "",
            "retrieved_documents": [],
            "reranked_documents": [],
            "context": "",
            "draft_answer": "",
            "final_answer": "",
            "sources": [],
            "claims": [],
            "verification": [],
            "critique": "",
            "critic_score": 0.0,
            "issues": [],
            "repair_strategy": "",
            "retry_count": 0,
            "final_decision": "",
            "confidence": 0.0,
            "cost": 0.0,
            "latency": {},
        }

        # Run compiled LangGraph state graph
        config = {
            "configurable": {
                "db": db,
                "top_k": top_k or self._top_k,
                "score_threshold": score_threshold or self._score_threshold,
                "retrieval_mode": retrieval_mode,
                "rerank_enabled": rerank_enabled,
                "dense_retriever": self._dense_retriever,
                "bm25_retriever": self._bm25_retriever,
                "hybrid_retriever": self._hybrid_retriever,
                "context_builder": self._context_builder,
                "generator": self._generator,
            }
        }

        from app.agents.graph import compiled_graph
        final_state = await compiled_graph.ainvoke(initial_state, config=config)

        total_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0

        # ── Step 4: Audit log ─────────────────────────────────────────────────
        if db is not None:
            await self._persist_query_log(
                db=db,
                workspace_id=workspace_id,
                question=question,
                answer=final_state.get("final_answer", ""),
                latency_ms=total_latency_ms,
            )

        # ── Step 5: Build response ────────────────────────────────────────────
        sources_dict = final_state.get("sources") or []
        sources = [
            QuerySource(
                document_title=s["document_title"],
                filename=s["filename"],
                page_number=s["page_number"],
                section_heading=s["section_heading"],
                chunk_index=s["chunk_index"],
                score=s["score"],
                document_id=s["document_id"],
            )
            for s in sources_dict
        ]

        # Extract latencies from graph run
        latencies = final_state.get("latency") or {}
        retrieval_latency = latencies.get("retrieval", 0.0)
        generation_latency = latencies.get("generation", 0.0)

        # Retrieve dynamic parameters chosen by the agent
        chosen_strategy = final_state.get("retrieval_strategy", "dense")
        query_type = final_state.get("query_type", "factual")

        settings = get_settings()
        effective_rerank = (
            rerank_enabled if rerank_enabled is not None else settings.RAG_RERANK_ENABLED
        )
        reranked = bool(effective_rerank and final_state.get("reranked_documents"))

        return QueryResult(
            answer=final_state.get("final_answer", ""),
            sources=sources,
            retrieval_latency_ms=round(retrieval_latency, 2),
            generation_latency_ms=round(generation_latency, 2),
            total_latency_ms=round(total_latency_ms, 2),
            model_used=self._generator._llm.model_name,
            chunks_retrieved=len(final_state.get("retrieved_documents") or []),
            context_chars=len(final_state.get("context", "")),
            grounded=str(final_state.get("final_decision")).lower() == "accept",
            tokens_used=0,
            metadata={
                "context_truncated": False,
                "finish_reason": "stop",
                "retrieval_mode": chosen_strategy,
                "query_type": query_type,
                "rerank_enabled": effective_rerank,
                "reranked": reranked,
                "plan": final_state.get("plan", ""),
                "latency_breakdown": latencies,
                # Phase 6 additions
                "final_decision": final_state.get("final_decision", "accept"),
                "critic_score": final_state.get("critic_score", 1.0),
                "confidence": final_state.get("confidence", 1.0),
                "retry_count": final_state.get("retry_count", 0),
                "issues": final_state.get("issues") or [],
                "claims": final_state.get("claims") or [],
                "verification": final_state.get("verification") or [],
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
