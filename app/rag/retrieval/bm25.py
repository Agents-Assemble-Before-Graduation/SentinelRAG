"""BM25 keyword retriever implementing the BaseRetriever interface."""

import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select, func

from app.core.config import get_settings
from app.core.logging import get_logger
from app.database.session import AsyncSessionLocal
from app.models.base import DocumentChunkRecord, DocumentRecord
from app.rag.retrieval.base import BaseRetriever
from app.rag.retrieval.retriever import RetrievedChunk

logger = get_logger(__name__)


def tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric word tokens."""
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Engine:
    """Lightweight in-memory BM25 ranker for a document chunk corpus."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_count = len(corpus)

        # Calculate document lengths and average length
        self.doc_lengths = [len(tokenize(doc["content"])) for doc in corpus]
        self.avg_doc_len = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 1.0
        )

        # Term document frequency and per-doc term frequency mapping
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.term_df: Dict[str, int] = {}

        for doc in corpus:
            tokens = tokenize(doc["content"])
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)

            for token in tf:
                self.term_df[token] = self.term_df.get(token, 0) + 1

        # Precompute IDF values (standard Lucene/Elasticsearch BM25 IDF variant)
        self.idf: Dict[str, float] = {}
        for term, df in self.term_df.items():
            self.idf[term] = math.log(
                1.0 + (self.doc_count - df + 0.5) / (df + 0.5)
            )

    def search(self, query: str, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """Run BM25 scoring over the corpus and return top K matching items."""
        query_tokens = tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []

        scored_docs: List[Tuple[Dict[str, Any], float]] = []

        for idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_lengths[idx]
            tf_map = self.doc_term_freqs[idx]

            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = tf_map.get(token, 0)
                idf = self.idf[token]

                # BM25 tf scaling formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                )
                score += idf * (numerator / denominator)

            if score > 0.0:
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


class BM25Retriever(BaseRetriever):
    """Keyword-based retriever using BM25 scoring over SQL-stored document chunks."""

    # Thread-safe/process-safe lazy cache for BM25 engines keyed by (workspace_id, chunk_count)
    _engines_cache: Dict[Tuple[Optional[str], int], BM25Engine] = {}

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b

    async def _get_engine(self, workspace_id: Optional[str] = None) -> BM25Engine:
        """Fetch all chunks from database, building or returning cached BM25 engine."""
        async with AsyncSessionLocal() as session:
            # Check the current count of chunks in the database for cache validation
            count_query = select(func.count(DocumentChunkRecord.id))
            if workspace_id:
                count_query = count_query.where(
                    DocumentChunkRecord.workspace_id == uuid.UUID(workspace_id)
                )
            count_result = await session.execute(count_query)
            chunk_count = count_result.scalar() or 0

            cache_key = (workspace_id, chunk_count)
            if cache_key in self._engines_cache:
                return self._engines_cache[cache_key]

            # Cache miss: fetch all chunks and their associated document title/filename
            query = select(DocumentChunkRecord, DocumentRecord).join(
                DocumentRecord, DocumentChunkRecord.document_id == DocumentRecord.id
            )
            if workspace_id:
                query = query.where(
                    DocumentChunkRecord.workspace_id == uuid.UUID(workspace_id)
                )

            result = await session.execute(query)
            rows = result.all()

            corpus: List[Dict[str, Any]] = []
            for chunk, doc in rows:
                corpus.append(
                    {
                        "chunk_id": str(chunk.id),
                        "content": chunk.content,
                        "document_id": str(chunk.document_id),
                        "workspace_id": str(chunk.workspace_id),
                        "title": doc.title,
                        "filename": doc.filename,
                        "page_number": chunk.page_number,
                        "section_heading": chunk.section_heading,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata_json or {},
                    }
                )

            engine = BM25Engine(corpus, k1=self.k1, b=self.b)
            self._engines_cache[cache_key] = engine
            logger.info(
                "Built new BM25 engine for workspace %s with %d chunks.",
                workspace_id or "all",
                chunk_count,
            )
            return engine

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        workspace_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Perform BM25 retrieval over document chunks."""
        settings = get_settings()
        effective_top_k = top_k if top_k is not None else settings.RAG_TOP_K
        effective_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.RAG_SCORE_THRESHOLD
        )

        if not query or not query.strip():
            return []

        engine = await self._get_engine(workspace_id)
        hits = engine.search(query.strip(), effective_top_k)

        chunks: list[RetrievedChunk] = []
        for doc, score in hits:
            # Score thresholds can be applied if configured
            if effective_threshold is not None and score < effective_threshold:
                continue

            chunk = RetrievedChunk(
                chunk_id=doc["chunk_id"],
                content=doc["content"],
                score=score,
                document_id=doc["document_id"],
                document_title=doc["title"],
                filename=doc["filename"],
                page_number=doc["page_number"],
                section_heading=doc["section_heading"],
                chunk_index=doc["chunk_index"],
                metadata=doc["metadata"],
            )
            chunks.append(chunk)

        return chunks
