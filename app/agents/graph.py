"""LangGraph multi-agent RAG workflow definition."""

import time
from typing import Any, Dict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from app.core.config import get_settings
from app.core.logging import get_logger
from app.agents.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.generator import GeneratorAgent
from app.rag.retrieval.retriever import DenseRetriever
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.reranker import get_reranker
from app.rag.context.builder import ContextBuilder

logger = get_logger(__name__)

# Singletons / Lazy instantiations
_dense_retriever = None
_bm25_retriever = None
_hybrid_retriever = None
_context_builder = None


def get_default_dense_retriever():
    global _dense_retriever
    if _dense_retriever is None:
        _dense_retriever = DenseRetriever()
    return _dense_retriever


def get_default_bm25_retriever():
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25Retriever()
    return _bm25_retriever


def get_default_hybrid_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def get_default_context_builder():
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder()
    return _context_builder


async def planner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """planner node: analyzes question and plans retrieval."""
    start_time = time.perf_counter()
    question = state["question"]

    logger.info("[Agent Node: Planner] analyzing query '%s'", question[:50])
    planner = PlannerAgent()
    plan_data = await planner.plan(question)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["planning"] = round(latency_ms, 2)

    return {
        "query_type": plan_data["query_type"],
        "retrieval_strategy": plan_data["retrieval_strategy"],
        "plan": plan_data["plan"],
        "subquestions": plan_data["subquestions"],
        "latency": current_latency,
    }


async def retrieval_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """retrieval node: retrieves documents based on chosen strategy."""
    start_time = time.perf_counter()
    override_strategy = config.get("configurable", {}).get("retrieval_mode")
    strategy = override_strategy or state.get("retrieval_strategy", "dense")
    question = state["question"]
    workspace_id = state.get("workspace_id")
    top_k = config.get("configurable", {}).get("top_k", 5)
    score_threshold = config.get("configurable", {}).get("score_threshold", 0.3)

    logger.info("[Agent Node: Retrieval] strategy: %s, query: %s", strategy, question[:50])

    if strategy == "bm25":
        retriever = config.get("configurable", {}).get("bm25_retriever") or get_default_bm25_retriever()
    elif strategy == "hybrid":
        retriever = config.get("configurable", {}).get("hybrid_retriever") or get_default_hybrid_retriever()
    else:
        retriever = config.get("configurable", {}).get("dense_retriever") or get_default_dense_retriever()

    # Fetch chunks
    docs = await retriever.retrieve(
        query=question,
        top_k=top_k,
        score_threshold=score_threshold,
        workspace_id=workspace_id,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["retrieval"] = round(latency_ms, 2)

    return {
        "retrieved_documents": docs,
        "latency": current_latency,
    }


async def reranking_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """reranking node: applies cross-encoder re-scoring if enabled."""
    start_time = time.perf_counter()
    settings = get_settings()
    rerank_enabled = config.get("configurable", {}).get("rerank_enabled")
    if rerank_enabled is None:
        rerank_enabled = settings.RAG_RERANK_ENABLED
    question = state["question"]
    docs = state.get("retrieved_documents") or []

    if not rerank_enabled or not docs:
        logger.info("[Agent Node: Reranking] skipped (enabled: %s, doc count: %d)", rerank_enabled, len(docs))
        reranked_docs = docs
    else:
        logger.info("[Agent Node: Reranking] applying cross-encoder to %d documents", len(docs))
        reranker = get_reranker()
        reranked_docs = await reranker.rerank(query=question, chunks=docs)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["reranking"] = round(latency_ms, 2)

    return {
        "reranked_documents": reranked_docs,
        "latency": current_latency,
    }


async def context_builder_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """context_builder node: normalises and builds evidence context block."""
    start_time = time.perf_counter()
    docs = state.get("reranked_documents") or []

    logger.info("[Agent Node: Context Builder] building context block from %d documents", len(docs))
    context_builder = config.get("configurable", {}).get("context_builder") or get_default_context_builder()
    built = context_builder.build(docs)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["context_building"] = round(latency_ms, 2)

    return {
        "context": built.context_text,
        "latency": current_latency,
    }


async def generator_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """generator node: executes LLM generation using context and template."""
    start_time = time.perf_counter()

    logger.info("[Agent Node: Generator] generating answer")
    generator_instance = config.get("configurable", {}).get("generator")
    generator = GeneratorAgent(generator=generator_instance)
    gen_result = await generator.generate(state)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["generation"] = round(latency_ms, 2)

    # Merge generator outputs and add latency
    result = dict(gen_result)
    result["latency"] = current_latency

    return result


# ── LangGraph Workflow Orchestration ────────────────────────────────────────

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("reranking", reranking_node)
workflow.add_node("context_builder", context_builder_node)
workflow.add_node("generator", generator_node)

# Connect edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retrieval")
workflow.add_edge("retrieval", "reranking")
workflow.add_edge("reranking", "context_builder")
workflow.add_edge("context_builder", "generator")
workflow.add_edge("generator", END)

# Compile graph
compiled_graph = workflow.compile()
