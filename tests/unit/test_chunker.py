"""Unit tests for structure-aware text chunking."""

from app.rag.ingestion.chunker import StructureAwareChunker
from app.rag.ingestion.extractors import ExtractedDocument, ExtractedSection


def test_chunker_basic_splitting():
    """Verify chunker splits sentences within size limits."""
    chunker = StructureAwareChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=30)
    text = (
        "Sentence one is concise. Sentence two provides additional context. "
        "Sentence three elaborates on the system architecture. "
        "Sentence four concludes the evaluation benchmark."
    )
    doc = ExtractedDocument(
        title="Test Doc",
        full_text=text,
        sections=[ExtractedSection(content=text, heading="Overview", page_number=1)],
        total_pages=1,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) >= 30
        assert chunk.page_number == 1
        assert chunk.section_heading == "Overview"
        assert chunk.chunk_hash != ""


def test_chunker_provenance_preservation():
    """Verify chunker preserves multiple sections and page numbers."""
    chunker = StructureAwareChunker(chunk_size=200, chunk_overlap=30, min_chunk_size=20)
    sections = [
        ExtractedSection(content="Page 1 introduction text for testing.", heading="Intro", page_number=1),
        ExtractedSection(content="Page 2 methodology text with detailed algorithms.", heading="Methods", page_number=2),
    ]
    doc = ExtractedDocument(
        title="Multi-Page Doc",
        full_text="Combined text",
        sections=sections,
        total_pages=2,
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].section_heading == "Intro"
    assert chunks[1].page_number == 2
    assert chunks[1].section_heading == "Methods"


def test_chunker_token_estimation():
    """Verify estimated token count logic."""
    chunker = StructureAwareChunker()
    text = "The quick brown fox jumps over the lazy dog."
    tokens = chunker.estimate_tokens(text)
    assert tokens >= 9
