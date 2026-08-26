"""LangGraph multi-agent RAG workflow definition with Critic, Verifier, Repair, and Kill nodes."""

import time
from typing import Any, Dict, List
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from app.core.config import get_settings
from app.core.logging import get_logger
from app.agents.state import AgentState
from app.agents.planner import PlannerAgent
from app.agents.generator import GeneratorAgent
from app.agents.critic import CriticAgent
from app.agents.claim_extractor import ClaimExtractorAgent
from app.agents.verifier import EvidenceVerifierAgent
from app.agents.repair import RepairEngine
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


# ── Graph Nodes ─────────────────────────────────────────────────────────────

async def planner_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Planner node: Classifies query and maps retrieval steps."""
    start_time = time.perf_counter()
    question = state["question"]

    logger.info("[Agent Node: Planner] Analyzing query '%s'", question[:50])
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
        "original_question": question,  # Keep backup of original question
        "latency": current_latency,
        "retry_count": 0,
    }


async def retrieval_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieval node: Executes search based on strategy or repair settings."""
    start_time = time.perf_counter()
    override_strategy = config.get("configurable", {}).get("retrieval_mode")
    strategy = override_strategy or state.get("retrieval_strategy", "dense")
    question = state["question"]
    workspace_id = state.get("workspace_id")
    
    # Check for top-k overrides (from repair engine retrieval expansion)
    top_k_override = state.get("top_k_override")
    config_top_k = config.get("configurable", {}).get("top_k", 5)
    top_k = top_k_override or config_top_k
    
    score_threshold = config.get("configurable", {}).get("score_threshold", 0.3)

    logger.info("[Agent Node: Retrieval] Strategy: %s, top-K: %d, query: %s", strategy, top_k, question[:50])

    if strategy == "bm25":
        retriever = config.get("configurable", {}).get("bm25_retriever") or get_default_bm25_retriever()
    elif strategy == "hybrid":
        retriever = config.get("configurable", {}).get("hybrid_retriever") or get_default_hybrid_retriever()
    else:
        retriever = config.get("configurable", {}).get("dense_retriever") or get_default_dense_retriever()

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
    """Reranking node: Re-scores candidates via cross-encoder."""
    start_time = time.perf_counter()
    settings = get_settings()
    rerank_enabled = config.get("configurable", {}).get("rerank_enabled")
    if rerank_enabled is None:
        rerank_enabled = settings.RAG_RERANK_ENABLED
    question = state["question"]
    docs = state.get("retrieved_documents") or []

    if not rerank_enabled or not docs:
        logger.info("[Agent Node: Reranking] Skipped (enabled: %s, doc count: %d)", rerank_enabled, len(docs))
        reranked_docs = docs
    else:
        logger.info("[Agent Node: Reranking] Applying cross-encoder to %d documents", len(docs))
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
    """Context builder node: Standardizes evidence blocks."""
    start_time = time.perf_counter()
    docs = state.get("reranked_documents") or []

    logger.info("[Agent Node: Context Builder] Formatting %d documents", len(docs))
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
    """Generator node: Creates draft answer from context."""
    start_time = time.perf_counter()

    logger.info("[Agent Node: Generator] Generating draft answer")
    generator_instance = config.get("configurable", {}).get("generator")
    generator = GeneratorAgent(generator=generator_instance)
    gen_result = await generator.generate(state)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["generation"] = round(latency_ms, 2)

    result = dict(gen_result)
    result["latency"] = current_latency
    return result


async def claim_extractor_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Claim extractor node: Extracts atomic claims from generated draft."""
    start_time = time.perf_counter()
    draft = state.get("draft_answer", "")

    logger.info("[Agent Node: Claim Extractor] Decomposing response facts")
    extractor = ClaimExtractorAgent()
    claims = await extractor.extract_claims(draft)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["claim_extraction"] = round(latency_ms, 2)

    return {
        "claims": claims,
        "latency": current_latency,
    }


async def verifier_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Verifier node: Evaluates claim truthfulness against context."""
    start_time = time.perf_counter()
    claims = state.get("claims") or []
    context = state.get("context", "")

    logger.info("[Agent Node: Evidence Verifier] Auditing %d atomic claims", len(claims))
    verifier = EvidenceVerifierAgent()
    verifications = await verifier.verify_claims(claims, context)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["verification"] = round(latency_ms, 2)

    return {
        "verification": verifications,
        "latency": current_latency,
    }


async def critic_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Critic node: Evaluates support, coverage, and decides accept/repair/kill."""
    start_time = time.perf_counter()
    draft = state.get("draft_answer", "")
    verifications = state.get("verification") or []

    logger.info("[Agent Node: Critic] Critiquing answer support")
    critic = CriticAgent()
    critique_data = await critic.critique(draft, verifications)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["critic"] = round(latency_ms, 2)

    return {
        "final_decision": critique_data["decision"],
        "critic_score": critique_data["score"],
        "issues": critique_data["issues"],
        "repair_strategy": critique_data["recommended_repair"],
        "latency": current_latency,
    }


async def judge_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Judge node: Applies confidence scoring and hard safety overrides."""
    start_time = time.perf_counter()
    verifications = state.get("verification") or []
    critic_score = state.get("critic_score", 0.0)
    decision = state.get("final_decision", "ACCEPT")
    retry_count = state.get("retry_count", 0)

    # 1. Calculate claim coverage
    total_claims = len(verifications)
    supported_claims = sum(1 for v in verifications if v["status"] == "SUPPORTED")
    partial_claims = sum(1 for v in verifications if v["status"] == "PARTIALLY_SUPPORTED")
    contradictions = sum(1 for v in verifications if v["status"] == "CONTRADICTED")

    coverage = (supported_claims + 0.5 * partial_claims) / total_claims if total_claims > 0 else 1.0

    # 2. Retrieval quality check (first chunk score)
    retrieved_docs = state.get("reranked_documents") or []
    retrieval_quality = retrieved_docs[0].score if retrieved_docs else 0.0
    retrieval_quality = max(0.0, min(1.0, retrieval_quality))

    # 3. Calculate final confidence (System estimate)
    # Weights: 50% claim support, 30% critic score, 20% search retrieval quality
    confidence = (0.5 * coverage) + (0.3 * critic_score) + (0.2 * retrieval_quality)
    confidence = round(max(0.0, min(1.0, confidence)), 2)

    logger.info(
        "[Agent Node: Judge] Coverage: %.2f, Confidence: %.2f, Initial Decision: %s",
        coverage, confidence, decision
    )

    # 4. Hard safety override rules
    # RULE A: If claims are contradicted -> KILL
    if contradictions > 0:
        logger.warning("[Agent Node: Judge Override] Direct contradiction found in claims! Forcing KILL.")
        decision = "KILL"
    
    # RULE B: If retry count exceeded -> Forcibly KILL rather than continuing repair
    if retry_count >= 2 and decision == "REPAIR":
        logger.warning("[Agent Node: Judge Override] Retry limit reached! Forcing KILL.")
        decision = "KILL"

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["judge"] = round(latency_ms, 2)

    return {
        "final_decision": decision,
        "confidence": confidence,
        "latency": current_latency,
    }


async def repair_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Repair node: Applies recommended query rewrites or strategy shifts."""
    start_time = time.perf_counter()
    strategy = state.get("repair_strategy", "QUERY_REWRITE")
    question = state.get("question", "")
    original = state.get("original_question", question)
    issues = state.get("issues") or []
    retry_count = state.get("retry_count", 0)

    logger.info("[Agent Node: Repair] Action: %s, Current retry: %d", strategy, retry_count)
    engine = RepairEngine()

    new_question = question
    top_k = state.get("top_k_override")
    ret_strategy = state.get("retrieval_strategy", "dense")

    if strategy == "QUERY_REWRITE":
        new_question = await engine.rewrite_query(original, issues)
        logger.info("[Agent Node: Repair] Rewrote query to: '%s'", new_question)
    elif strategy == "RETRIEVAL_EXPANSION":
        top_k = (state.get("top_k_override") or 5) + 5
        logger.info("[Agent Node: Repair] Expanded retrieval search top-k to %d", top_k)
    elif strategy == "RETRIEVAL_WEIGHT_CHANGE":
        ret_strategy = engine.shift_strategy(ret_strategy)
        logger.info("[Agent Node: Repair] Shifted retrieval strategy to: %s", ret_strategy)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["repair"] = round(latency_ms, 2)

    return {
        "question": new_question,
        "top_k_override": top_k,
        "retrieval_strategy": ret_strategy,
        "retry_count": retry_count + 1,
        "latency": current_latency,
    }


async def kill_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Kill node: Terminates execution safely with refusal reason and attempts count."""
    start_time = time.perf_counter()
    attempts = state.get("retry_count", 0) + 1
    issues = state.get("issues") or []
    
    # Extract missing claims to help inform user
    unsupported = [v["id"] for v in state.get("verification", []) if v["status"] in ("UNSUPPORTED", "CONTRADICTED")]

    refusal_msg = (
        "Refusal: Insufficient evidence in source documents to answer the question safely. "
        "The system has terminated execution to prevent hallucinations."
    )
    
    meta_refusal = {
        "reason": "Critic rejected support status or contradictions found.",
        "critic_issues": issues,
        "unverified_claims": unsupported,
        "repair_attempts": attempts,
    }

    logger.info("[Agent Node: Kill] Pipeline terminated after %d attempts.", attempts)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    current_latency = state.get("latency") or {}
    current_latency["kill"] = round(latency_ms, 2)

    return {
        "final_answer": refusal_msg,
        "sources": [],
        "final_decision": "kill",
        "latency": current_latency,
    }


async def accept_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Accept node: Sets draft answer as final answer."""
    logger.info("[Agent Node: Accept] Answer successfully validated.")
    return {
        "final_answer": state.get("draft_answer", ""),
    }


# ── State Graph Routing ─────────────────────────────────────────────────────

def route_decision(state: AgentState) -> str:
    """Evaluate final_decision state variable and route edge transitions."""
    decision = state.get("final_decision", "ACCEPT")
    if decision == "ACCEPT":
        return "accept"
    elif decision == "REPAIR":
        return "repair"
    else:
        return "kill"


# ── LangGraph Workflow Orchestration ────────────────────────────────────────

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("reranking", reranking_node)
workflow.add_node("context_builder", context_builder_node)
workflow.add_node("generator", generator_node)
workflow.add_node("claim_extractor", claim_extractor_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("critic", critic_node)
workflow.add_node("judge", judge_node)
workflow.add_node("repair", repair_node)
workflow.add_node("kill", kill_node)
workflow.add_node("accept", accept_node)

# Connect edges
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retrieval")
workflow.add_edge("retrieval", "reranking")
workflow.add_edge("reranking", "context_builder")
workflow.add_edge("context_builder", "generator")
workflow.add_edge("generator", "claim_extractor")
workflow.add_edge("claim_extractor", "verifier")
workflow.add_edge("verifier", "critic")
workflow.add_edge("critic", "judge")

# Conditional Routing from Judge
workflow.add_conditional_edges(
    "judge",
    route_decision,
    {
        "accept": "accept",
        "repair": "repair",
        "kill": "kill",
    }
)

# Loop back from Repair Node to Retrieval
workflow.add_edge("repair", "retrieval")

# Standard terminations
workflow.add_edge("accept", END)
workflow.add_edge("kill", END)

# Compile graph
compiled_graph = workflow.compile()
