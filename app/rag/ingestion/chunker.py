"""Structure-aware text chunking with metadata preservation."""

import re
from dataclasses import dataclass, field
from typing import Any

from app.rag.ingestion.deduplication import calculate_content_hash
from app.rag.ingestion.extractors import ExtractedDocument


@dataclass
class DocumentChunk:
    """A single granular text chunk ready for embedding and vector indexing."""

    chunk_index: int
    content: str
    token_count: int
    page_number: int | None = None
    section_heading: str | None = None
    chunk_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    """Chunker that preserves document section hierarchies, headings, and page numbers."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (approx. 4 chars per token or whitespace token count)."""
        words = len(text.split())
        chars_est = len(text) // 4
        return max(words, chars_est)

    def _split_text_with_overlap(self, text: str) -> list[str]:
        """Split a block of text into overlapping windows respecting sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []

        # Split into sentences
        sentences = re.split(r"(?<=[.?!])\s+", text)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if not sentence.strip():
                continue

            if current_len + sent_len > self.chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk).strip()
                if len(chunk_str) >= self.min_chunk_size:
                    chunks.append(chunk_str)

                # Maintain overlap from trailing sentences
                overlap_chunk: list[str] = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = overlap_len

            current_chunk.append(sentence)
            current_len += sent_len

        if current_chunk:
            final_str = " ".join(current_chunk).strip()
            if len(final_str) >= self.min_chunk_size:
                chunks.append(final_str)
            elif chunks:
                # Merge small trailing fragment with previous chunk if allowed
                chunks[-1] = (chunks[-1] + " " + final_str).strip()

        return chunks

    def chunk_document(self, extracted_doc: ExtractedDocument) -> list[DocumentChunk]:
        """Chunk all sections of an extracted document while maintaining provenance metadata."""
        chunks: list[DocumentChunk] = []
        chunk_idx = 0

        for section in extracted_doc.sections:
            sec_text = section.content.strip()
            if not sec_text:
                continue

            section_chunks = self._split_text_with_overlap(sec_text)
            for text_chunk in section_chunks:
                chunk_hash = calculate_content_hash(text_chunk)
                tokens = self.estimate_tokens(text_chunk)

                chunk = DocumentChunk(
                    chunk_index=chunk_idx,
                    content=text_chunk,
                    token_count=tokens,
                    page_number=section.page_number,
                    section_heading=section.heading,
                    chunk_hash=chunk_hash,
                    metadata={
                        **section.metadata,
                        "title": extracted_doc.title,
                        "char_length": len(text_chunk),
                    },
                )
                chunks.append(chunk)
                chunk_idx += 1

        # Fallback if no chunks were generated (e.g. text was shorter than min_chunk_size)
        if not chunks and extracted_doc.full_text.strip():
            raw_text = extracted_doc.full_text.strip()
            chunk_hash = calculate_content_hash(raw_text)
            chunks.append(
                DocumentChunk(
                    chunk_index=0,
                    content=raw_text,
                    token_count=self.estimate_tokens(raw_text),
                    page_number=1,
                    section_heading="Full Text",
                    chunk_hash=chunk_hash,
                    metadata={"title": extracted_doc.title, "char_length": len(raw_text)},
                )
            )

        return chunks
