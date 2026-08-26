"""Unit tests for LangGraph multi-agent RAG workflow (Planner, Generator, State Graph)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.generator import GeneratorAgent
from app.agents.graph import compiled_graph, planner_node, retrieval_node, reranking_node, context_builder_node, generator_node
from app.rag.retrieval.retriever import RetrievedChunk
from app.rag.context.builder import ContextBuilder, BuiltContext
from app.rag.generation.generator import GenerationResult


def test_planner_agent_heuristics():
    """Verify rule-based local heuristics categorise and select strategies correctly."""
    planner = PlannerAgent()

    # Definition query
    p1 = planner._heuristics_plan("What is retrieval augmented generation?")
    assert p1["query_type"] == "definition"
    assert p1["retrieval_strategy"] == "dense"

    # Comparison query
    p2 = planner._heuristics_plan("compare dense versus sparse retrieval")
    assert p2["query_type"] == "comparison"
    assert p2["retrieval_strategy"] == "hybrid"

    # Numerical query
    p3 = planner._heuristics_plan("how many features were added in version 0.2.0?")
    assert p3["query_type"] == "numerical"
    assert p3["retrieval_strategy"] == "bm25"


@pytest.mark.asyncio
async def test_planner_agent_llm_parsing():
    """Verify PlannerAgent correctly prompts LLM and parses JSON query plan outputs."""
    mock_llm = AsyncMock()
    mock_llm.is_available.return_value = True
    mock_llm.provider_name = "test-provider"
    
    mock_response = MagicMock()
    mock_response.content = """```json
{
  "query_type": "comparison",
  "retrieval_strategy": "hybrid",
  "plan": "Retrieve differences between systems.",
  "subquestions": ["What is system A?", "What is system B?"]
}
```"""
    mock_llm.chat_complete.return_value = mock_response

    planner = PlannerAgent(llm_provider=mock_llm)
    res = await planner.plan("compare sys A vs B")

    assert res["query_type"] == "comparison"
    assert res["retrieval_strategy"] == "hybrid"
    assert res["plan"] == "Retrieve differences between systems."
    assert res["subquestions"] == ["What is system A?", "What is system B?"]


@pytest.mark.asyncio
async def test_generator_agent():
    """Verify GeneratorAgent wraps RAGGenerator, converts document structures, and updates decision states."""
    mock_generator = AsyncMock()
    mock_generator.generate.return_value = GenerationResult(
        answer="Agent test answer",
        sources=[],
        model_used="test-model",
        grounded=True,
    )

    agent = GeneratorAgent(generator=mock_generator)
    
    state: AgentState = {
        "question": "test question",
        "workspace_id": None,
        "query_type": "factual",
        "plan": "",
        "subquestions": [],
        "retrieval_strategy": "dense",
        "retrieved_documents": [],
        "reranked_documents": [
            RetrievedChunk(
                chunk_id="c1",
                content="test context",
                score=0.9,
                document_id="d1",
                document_title="Title A",
                filename="a.pdf"
            )
        ],
        "context": "formatted context string",
        "draft_answer": "",
        "final_answer": "",
        "sources": [],
        "claims": [],
        "critique": "",
        "verification": {},
        "repair_strategy": "",
        "retry_count": 0,
        "final_decision": "",
        "confidence": 0.0,
        "cost": 0.0,
        "latency": {},
    }

    res = await agent.generate(state)
    assert res["draft_answer"] == "Agent test answer"
    assert res["final_answer"] == "Agent test answer"
    assert res["final_decision"] == "accept"
    assert res["confidence"] == 1.0


@pytest.mark.asyncio
async def test_graph_nodes_and_execution():
    """Verify LangGraph nodes execute successfully and propagate state changes."""
    # Build a valid starting state
    state: AgentState = {
        "question": "What is SentinelRAG?",
        "workspace_id": None,
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
        "critique": "",
        "verification": {},
        "repair_strategy": "",
        "retry_count": 0,
        "final_decision": "",
        "confidence": 0.0,
        "cost": 0.0,
        "latency": {},
    }

    # 1. Test Planner Node (uses heuristics because LLM is unavailable in tests)
    plan_out = await planner_node(state, config={})
    assert plan_out["query_type"] == "definition"
    assert plan_out["retrieval_strategy"] == "dense"
    assert "planning" in plan_out["latency"]

    # Update state
    state.update(plan_out)

    # 2. Test Retrieval Node using mock retriever
    mock_retriever = AsyncMock()
    mock_chunk = RetrievedChunk(
        chunk_id="chunk-1",
        content="mock database chunk",
        score=0.95,
        document_id="doc-1",
        document_title="Mock Title",
        filename="mock.pdf",
        page_number=1
    )
    mock_retriever.retrieve.return_value = [mock_chunk]

    config = {
        "configurable": {
            "dense_retriever": mock_retriever,
            "top_k": 3,
            "score_threshold": 0.1,
        }
    }
    ret_out = await retrieval_node(state, config=config)
    assert len(ret_out["retrieved_documents"]) == 1
    assert ret_out["retrieved_documents"][0].chunk_id == "chunk-1"
    assert "retrieval" in ret_out["latency"]

    # Update state
    state.update(ret_out)

    # 3. Test Reranking Node
    rerank_out = await reranking_node(state, config={"configurable": {"rerank_enabled": False}})
    assert len(rerank_out["reranked_documents"]) == 1
    assert "reranking" in rerank_out["latency"]

    # Update state
    state.update(rerank_out)

    # 4. Test Context Builder Node
    mock_builder = MagicMock()
    mock_built = BuiltContext(
        context_text="Compiled: mock database chunk",
        sources=[],
        total_chunks=1,
        included_chunks=1,
        total_chars=30,
        was_truncated=False
    )
    mock_builder.build.return_value = mock_built

    cb_out = await context_builder_node(state, config={"configurable": {"context_builder": mock_builder}})
    assert cb_out["context"] == "Compiled: mock database chunk"
    assert "context_building" in cb_out["latency"]


@pytest.mark.asyncio
async def test_full_graph_invocation():
    """Verify that CompiledStateGraph coordinates the sequential nodes and outputs final state."""
    initial_state: AgentState = {
        "question": "What is SentinelRAG?",
        "workspace_id": None,
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
        "critique": "",
        "verification": {},
        "repair_strategy": "",
        "retry_count": 0,
        "final_decision": "",
        "confidence": 0.0,
        "cost": 0.0,
        "latency": {},
    }

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            content="SentinelRAG is a self-improving platform",
            score=0.99,
            document_id="d1",
            document_title="Title A",
            filename="a.pdf"
        )
    ]

    mock_generator = AsyncMock()
    mock_generator.generate.return_value = GenerationResult(
        answer="SentinelRAG is a self-improving platform",
        sources=[],
        model_used="test-model",
        grounded=True,
    )

    config = {
        "configurable": {
            "dense_retriever": mock_retriever,
            "generator": mock_generator,
            "rerank_enabled": False,
        }
    }

    final_state = await compiled_graph.ainvoke(initial_state, config=config)

    assert final_state["query_type"] == "definition"
    assert final_state["retrieval_strategy"] == "dense"
    assert final_state["context"] != ""
    assert final_state["final_answer"] == "SentinelRAG is a self-improving platform"
    assert final_state["final_decision"] == "ACCEPT"
