"""Unit tests for retrieval and generation evaluation metrics."""

import pytest
from unittest.mock import AsyncMock

from app.rag.retrieval.retriever import RetrievedChunk
from app.rag.context.builder import BuiltContext, SourceCitation
from app.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
)
from app.evaluation.generation_metrics import LLMJudgeEvaluator


@pytest.fixture
def retrieved_chunks():
    return [
        RetrievedChunk(
            chunk_id="c1",
            content="first chunk content",
            score=0.9,
            document_id="d1",
            document_title="Title 1",
            filename="doc1.pdf",
            page_number=1
        ),
        RetrievedChunk(
            chunk_id="c2",
            content="second chunk content",
            score=0.8,
            document_id="d2",
            document_title="Title 2",
            filename="doc2.pdf",
            page_number=2
        ),
        RetrievedChunk(
            chunk_id="c3",
            content="third chunk content",
            score=0.7,
            document_id="d1",
            document_title="Title 1",
            filename="doc1.pdf",
            page_number=3
        ),
    ]


def test_retrieval_metrics(retrieved_chunks):
    """Verify Recall, Precision, MRR, and NDCG scoring algorithms."""
    # Hit is doc2.pdf page 2
    assert calculate_recall_at_k(retrieved_chunks, "doc2.pdf", 2, k=1) == 0.0
    assert calculate_recall_at_k(retrieved_chunks, "doc2.pdf", 2, k=2) == 1.0

    assert calculate_precision_at_k(retrieved_chunks, "doc2.pdf", 2, k=2) == 0.5
    assert calculate_precision_at_k(retrieved_chunks, "doc2.pdf", 2, k=1) == 0.0

    assert calculate_mrr(retrieved_chunks, "doc2.pdf", 2) == 0.5
    assert calculate_mrr(retrieved_chunks, "doc3.pdf", 1) == 0.0

    # NDCG calculation for hit at rank 2:
    # DCG = 0 / log2(2) + 1 / log2(3) = 1 / 1.58496 = 0.6309
    # IDCG = 1 / log2(2) = 1.0
    # NDCG = 0.6309
    ndcg = calculate_ndcg_at_k(retrieved_chunks, "doc2.pdf", 2, k=3)
    assert pytest.approx(ndcg, rel=1e-3) == 0.6309


@pytest.mark.asyncio
async def test_llm_judge_fallback_eval():
    """Verify evaluator handles unconfigured/mock environments gracefully with fallback scores."""
    mock_llm = AsyncMock()
    mock_llm.provider_name = "mock"
    mock_llm.is_available.return_value = True

    evaluator = LLMJudgeEvaluator(llm_provider=mock_llm)
    
    faith = await evaluator.score_faithfulness("answer", "context")
    ans_rel = await evaluator.score_answer_relevance("question", "answer")
    
    # In mock/unconfigured mode, it should return safe fallback 1.0
    assert faith == 1.0
    assert ans_rel == 1.0


def test_citation_metrics():
    """Verify citation correctness and completeness heuristic checks."""
    evaluator = LLMJudgeEvaluator()
    
    answer = "SentinelRAG is a multi-agent system [Evidence 1]. While traditional systems use single-pass RAG [Evidence 2]."
    
    # 2 citations: Evidence 1 and Evidence 2. Both sentences have citations.
    assert evaluator.score_citation_completeness(answer) == 1.0

    # Sentence 2 doesn't have a citation
    answer_incomplete = "Sentence 1 is cited [Evidence 1]. Sentence 2 is uncited."
    assert evaluator.score_citation_completeness(answer_incomplete) == 0.5

    # Mock context sources for correctness validation
    citation1 = SourceCitation(
        document_title="Doc A",
        filename="a.pdf",
        page_number=1,
        section_heading="Intro",
        chunk_index=0,
        score=0.9,
        document_id="d1"
    )
    citation2 = SourceCitation(
        document_title="Doc B",
        filename="b.pdf",
        page_number=1,
        section_heading="Intro",
        chunk_index=0,
        score=0.8,
        document_id="d2"
    )
    mock_context = type("BuiltContext", (), {
        "context_text": "[Evidence 1]\nSource: Doc A\n---\nSentinelRAG is a multi-agent system.\n\n[Evidence 2]\nSource: Doc B\n---\nTraditional systems are single-pass pipelines.",
        "sources": [citation1, citation2]
    })()

    correctness = evaluator.score_citation_correctness(answer, mock_context)
    assert correctness == 1.0
