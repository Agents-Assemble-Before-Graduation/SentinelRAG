"""Context builder: deduplication, ranking, size enforcement, and source citation."""

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retrieval.retriever import RetrievedChunk

logger = get_logger(__name__)

# Evidence block template — each chunk is wrapped as numbered evidence
_EVIDENCE_BLOCK_TEMPLATE = (
    "[Evidence {n}]\n"
    "Source: {title} | Page: {page} | Section: {section}\n"
    "---\n"
    "{content}\n"
)


@dataclass
class SourceCitation:
    """Citation metadata for a single evidence chunk used in generation."""

    document_title: str
    filename: str
    page_number: int | None
    section_heading: str | None
    chunk_index: int
    score: float
    document_id: str


@dataclass
class BuiltContext:
    """Assembled context ready for injection into the LLM prompt."""

    context_text: str                    # Formatted numbered evidence blocks
    sources: list[SourceCitation]        # One citation per included chunk
    total_chunks: int                    # Chunks before dedup/truncation
    included_chunks: int                 # Chunks actually included
    total_chars: int                     # Character count of context_text
    was_truncated: bool = False          # True if context was cut for size


class ContextBuilder:
    """Builds a size-bounded, deduplicated context from retrieved chunks.

    Responsibilities:
    - Remove duplicate chunks (same chunk_id or identical content).
    - Sort by descending relevance score.
    - Enforce a maximum character limit to protect token budgets.
    - Format each chunk as a numbered evidence block with source metadata.
    - Return structured source citations for downstream citation generation.
    """

    def __init__(self, max_context_chars: int | None = None) -> None:
        settings = get_settings()
        self._max_context_chars = max_context_chars or settings.RAG_MAX_CONTEXT_CHARS

    def build(self, chunks: list[RetrievedChunk]) -> BuiltContext:
        """Assemble a BuiltContext from a list of retrieved chunks.

        Args:
            chunks: Raw list of RetrievedChunk objects from the retriever.

        Returns:
            BuiltContext with formatted evidence text and source citations.
        """
        total_chunks = len(chunks)

        if not chunks:
            logger.debug("ContextBuilder received 0 chunks — returning empty context.")
            return BuiltContext(
                context_text="",
                sources=[],
                total_chunks=0,
                included_chunks=0,
                total_chars=0,
                was_truncated=False,
            )

        # Step 1: Deduplicate by chunk_id (fall back to content fingerprint)
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []

        for chunk in chunks:
            dedup_key = chunk.chunk_id if chunk.chunk_id else None
            content_key = chunk.content[:200]  # fingerprint on first 200 chars

            if dedup_key and dedup_key in seen_ids:
                continue
            if content_key in seen_content:
                continue

            if dedup_key:
                seen_ids.add(dedup_key)
            seen_content.add(content_key)
            unique_chunks.append(chunk)

        # Step 2: Sort by descending similarity score
        unique_chunks.sort(key=lambda c: c.score, reverse=True)

        # Step 3: Enforce character budget — include as many high-scoring chunks as fit
        evidence_blocks: list[str] = []
        sources: list[SourceCitation] = []
        total_chars = 0
        was_truncated = False

        for idx, chunk in enumerate(unique_chunks, start=1):
            page_str = str(chunk.page_number) if chunk.page_number is not None else "N/A"
            section_str = chunk.section_heading or "N/A"

            block = _EVIDENCE_BLOCK_TEMPLATE.format(
                n=idx,
                title=chunk.document_title or "Unknown",
                page=page_str,
                section=section_str,
                content=chunk.content.strip(),
            )

            block_len = len(block)
            if total_chars + block_len > self._max_context_chars:
                logger.info(
                    "Context limit reached at chunk %d/%d (%.0f chars used of %d max).",
                    idx - 1,
                    len(unique_chunks),
                    total_chars,
                    self._max_context_chars,
                )
                was_truncated = True
                break

            evidence_blocks.append(block)
            total_chars += block_len
            sources.append(
                SourceCitation(
                    document_title=chunk.document_title or "Unknown",
                    filename=chunk.filename or "",
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    document_id=chunk.document_id,
                )
            )

        context_text = "\n".join(evidence_blocks)

        logger.debug(
            "Context built: %d/%d chunks included, %d chars, truncated=%s",
            len(sources),
            total_chunks,
            total_chars,
            was_truncated,
        )

        return BuiltContext(
            context_text=context_text,
            sources=sources,
            total_chunks=total_chunks,
            included_chunks=len(sources),
            total_chars=total_chars,
            was_truncated=was_truncated,
        )
