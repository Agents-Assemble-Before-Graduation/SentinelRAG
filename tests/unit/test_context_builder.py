"""Unit tests for ContextBuilder."""

import pytest

from app.rag.context.builder import ContextBuilder, BuiltContext, SourceCitation
from app.rag.retrieval.retriever import RetrievedChunk


def test_context_builder_empty():
    """Verify ContextBuilder handles empty retriever outputs gracefully."""
    builder = ContextBuilder(max_context_chars=1000)
    context = builder.build([])
    
    assert isinstance(context, BuiltContext)
    assert context.context_text == ""
    assert context.sources == []
    assert context.total_chunks == 0
    assert context.included_chunks == 0
    assert context.total_chars == 0
    assert context.was_truncated is False


def test_context_builder_deduplication_and_sorting():
    """Verify ContextBuilder deduplicates chunks by ID / content fingerprint, and sorts by score desc."""
    builder = ContextBuilder(max_context_chars=5000)
    
    # 200 character base for fingerprint match
    fingerprint_base = "This is chunk 2 content." + "A" * 180

    chunks = [
        # Lower score first to verify sorting
        RetrievedChunk(
            chunk_id="chunk-2",
            content=fingerprint_base,
            score=0.6,
            document_id="doc-1",
            document_title="Doc A",
            filename="doc_a.pdf",
            page_number=2,
            section_heading="Sec 2"
        ),
        # Higher score chunk
        RetrievedChunk(
            chunk_id="chunk-1",
            content="This is chunk 1 content.",
            score=0.9,
            document_id="doc-1",
            document_title="Doc A",
            filename="doc_a.pdf",
            page_number=1,
            section_heading="Sec 1"
        ),
        # Duplicate chunk ID
        RetrievedChunk(
            chunk_id="chunk-1",
            content="This is duplicate chunk 1 content.",
            score=0.95,
            document_id="doc-1",
            document_title="Doc A",
            filename="doc_a.pdf",
            page_number=1,
            section_heading="Sec 1"
        ),
        # Duplicate content fingerprint (first 200 chars same)
        RetrievedChunk(
            chunk_id="chunk-3",
            content=fingerprint_base + " extra suffix after 200 chars",
            score=0.8,
            document_id="doc-1",
            document_title="Doc A",
            filename="doc_a.pdf",
            page_number=2,
            section_heading="Sec 2"
        )
    ]

    context = builder.build(chunks)

    assert context.total_chunks == 4
    # Only chunk-1 and chunk-2 should survive dedup (chunk-3 matches chunk-2 fingerprint, duplicate chunk-1 id rejected)
    assert context.included_chunks == 2
    assert context.was_truncated is False
    
    # Verify sorting: chunk-1 (score 0.9) must be before chunk-2 (score 0.6)
    assert context.sources[0].document_title == "Doc A"
    assert context.sources[0].page_number == 1
    assert context.sources[0].score == 0.9

    assert context.sources[1].page_number == 2
    assert context.sources[1].score == 0.6

    # Verify context formatting contains "[Evidence 1]" and "[Evidence 2]"
    assert "[Evidence 1]" in context.context_text
    assert "[Evidence 2]" in context.context_text
    # Evidence 1 content should come before Evidence 2 content
    idx1 = context.context_text.index("This is chunk 1 content.")
    idx2 = context.context_text.index("This is chunk 2 content.")
    assert idx1 < idx2


def test_context_builder_character_truncation():
    """Verify ContextBuilder truncates chunks that exceed maximum character limits."""
    # Let's set a small limit so we trigger truncation
    builder = ContextBuilder(max_context_chars=200)
    
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            content="This is a very long chunk content that will definitely take up most of the budget by itself.",
            score=0.9,
            document_id="doc-1",
            document_title="Doc A",
            filename="doc_a.pdf"
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            content="This second chunk will not fit because of the low character limit.",
            score=0.8,
            document_id="doc-1",
            document_title="Doc B",
            filename="doc_b.pdf"
        )
    ]

    context = builder.build(chunks)
    assert context.total_chunks == 2
    assert context.included_chunks == 1
    assert context.was_truncated is True
    assert "Doc B" not in context.context_text
    assert context.sources[0].document_title == "Doc A"
