"""Unit tests for RAGGenerator."""

import pytest
from unittest.mock import AsyncMock

from app.rag.generation.generator import RAGGenerator, GenerationResult, _NO_EVIDENCE_RESPONSE
from app.rag.context.builder import BuiltContext, SourceCitation
from app.llm.mock_provider import MockLLMProvider
from app.llm.base import MessageRole


@pytest.mark.asyncio
async def test_rag_generator_no_evidence():
    """Verify generator immediately returns insufficient evidence response when context is empty."""
    llm = MockLLMProvider()
    generator = RAGGenerator(llm_provider=llm)
    
    empty_context = BuiltContext(
        context_text="",
        sources=[],
        total_chunks=0,
        included_chunks=0,
        total_chars=0,
        was_truncated=False
    )

    result = await generator.generate(
        question="What is RAG?",
        context=empty_context
    )

    assert isinstance(result, GenerationResult)
    assert result.answer == _NO_EVIDENCE_RESPONSE
    assert result.sources == []
    assert result.grounded is False
    # Verify no LLM call was issued
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_rag_generator_success():
    """Verify generator constructs prompt correctly and requests generation with temperature=0.0."""
    llm = MockLLMProvider(response_text="Grounded answer from mock.", model="gpt-mock")
    generator = RAGGenerator(llm_provider=llm)

    citation = SourceCitation(
        document_title="Sample Doc",
        filename="sample.pdf",
        page_number=1,
        section_heading="Intro",
        chunk_index=0,
        score=0.9,
        document_id="doc-1"
    )
    context = BuiltContext(
        context_text="[Evidence 1]\nSource: Sample Doc\n---\nPassage content.",
        sources=[citation],
        total_chunks=1,
        included_chunks=1,
        total_chars=50,
        was_truncated=False
    )

    result = await generator.generate(
        question="What is RAG?",
        context=context
    )

    assert isinstance(result, GenerationResult)
    assert result.answer == "Grounded answer from mock."
    assert result.sources == [citation]
    assert result.model_used == "gpt-mock"
    assert result.grounded is True
    
    # Verify LLM calls
    assert llm.call_count == 1
    # Check that system prompt and user prompts are passed
    messages = llm.last_messages
    assert len(messages) == 2
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    
    # Verify prompt structures
    assert "EXCLUSIVELY on the evidence passages" in messages[0].content
    assert "<evidence>" in messages[1].content
    assert "Passage content." in messages[1].content
    assert "What is RAG?" in messages[1].content
