#!/usr/bin/env python
"""Experiment runner evaluating Dense and Hybrid retrieval RAG models on the benchmark dataset."""

import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to python path to resolve app imports
sys.path.append(str(Path(__file__).parent.parent))

from app.database.session import AsyncSessionLocal
from app.core.config import get_settings
from app.core.exceptions import LLMProviderError
from app.models.base import Workspace, DocumentRecord
from app.services.query_service import RAGQueryService
from app.services.vector_store import get_vector_store
from app.rag.embeddings.factory import get_embedding_provider
from app.rag.retrieval.bm25 import BM25Retriever
from app.rag.retrieval.retriever import DenseRetriever
from app.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
)
from app.evaluation.generation_metrics import LLMJudgeEvaluator
from sqlalchemy import select

# Configure paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "benchmark.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"


async def check_database_has_data() -> tuple[bool, str | None]:
    """Verify if database has active workspaces and ingested documents."""
    try:
        async with AsyncSessionLocal() as session:
            # Check workspaces
            ws_res = await session.execute(select(Workspace).limit(1))
            ws = ws_res.scalar_one_or_none()
            if not ws:
                return False, None

            # Check documents
            doc_res = await session.execute(
                select(DocumentRecord).where(DocumentRecord.status == "indexed").limit(1)
            )
            doc = doc_res.scalar_one_or_none()
            if not doc:
                return False, str(ws.id)

            return True, str(ws.id)
    except Exception as e:
        print(f"Database connection could not be established: {e}")
        return False, None


async def run_evaluation():
    """Run RAG evaluation pipeline."""
    print("=" * 70)
    print(" SentinelRAG Research Evaluation Runner")
    print("=" * 70)

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}. Please create the dataset first.")
        return

    with open(DATASET_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation queries from dataset.")

    # Check setup status
    has_data, workspace_id = await check_database_has_data()
    if not has_data:
        print("\nWARNING: No indexed documents found in database.")
        print("Please run database migrations and ingest sample documents first:")
        print("  docker compose up -d postgres qdrant")
        print("  python scripts/ingest_demo_data.py")
        print("\nWriting PENDING results structure to evaluation_results/.")

        # Write pending results
        pending_results = {
            "status": "pending",
            "message": "Evaluation pending document ingestion.",
            "metrics": {},
        }
        with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(pending_results, f, indent=2)
        return

    settings = get_settings()
    llm_available = settings.LLM_API_KEY and settings.LLM_API_KEY.strip()
    print(f"Workspace resolved: {workspace_id}")
    print(f"LLM Provider: {settings.LLM_PROVIDER} (Available: {bool(llm_available)})")

    # Initialize services
    dense_retriever = DenseRetriever()
    bm25_retriever = BM25Retriever()
    evaluator = LLMJudgeEvaluator()

    modes = ["dense", "hybrid"]
    all_results: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in modes}

    for mode in modes:
        print(f"\nEvaluating mode: [{mode.upper()}]")
        query_service = RAGQueryService(retrieval_mode=mode)

        for idx, q_data in enumerate(queries, start=1):
            question = q_data["question"]
            gt_doc = q_data["source_document"]
            gt_page = q_data.get("source_page")
            category = q_data["category"]

            print(f"  [{idx}/{len(queries)}] Querying: '{question[:40]}...'")

            # 1. Execute Query Pipeline
            try:
                result = await query_service.query(
                    question=question,
                    workspace_id=workspace_id,
                    top_k=5,
                )
                answer = result.answer
                latency_ms = result.total_latency_ms
                sources = result.sources
            except LLMProviderError as e:
                print(f"    LLM Provider Error: {e.message}. Marking metrics as pending.")
                answer = "LLM unavailable"
                latency_ms = 0.0
                sources = []

            # 2. Retrieve candidates directly for metrics calculations
            retriever = dense_retriever if mode == "dense" else query_service._hybrid_retriever
            retrieved = await retriever.retrieve(
                query=question,
                top_k=5,
                workspace_id=workspace_id,
            )

            # 3. Calculate Retrieval Metrics
            recall_5 = calculate_recall_at_k(retrieved, gt_doc, gt_page, k=5)
            precision_5 = calculate_precision_at_k(retrieved, gt_doc, gt_page, k=5)
            mrr = calculate_mrr(retrieved, gt_doc, gt_page)
            ndcg_5 = calculate_ndcg_at_k(retrieved, gt_doc, gt_page, k=5)

            # 4. Calculate Generation Metrics
            # If LLM key not available, mark faithfulness and relevances as pending
            if not llm_available or answer == "LLM unavailable":
                faithfulness = "pending"
                answer_relevance = "pending"
                context_relevance = "pending"
            else:
                context_str = "\n".join([c.content for c in retrieved])
                faithfulness = await evaluator.score_faithfulness(answer, context_str)
                answer_relevance = await evaluator.score_answer_relevance(question, answer)
                context_relevance = await evaluator.score_context_relevance(question, context_str)

            # Build a BuiltContext mock structure to verify citation metrics
            mock_context = type(
                "BuiltContext",
                (),
                {
                    "context_text": "\n\n".join(
                        [
                            f"[Evidence {i}]\nSource: {c.document_title}\n---\n{c.content}"
                            for i, c in enumerate(retrieved, start=1)
                        ]
                    ),
                    "sources": retrieved,
                },
            )()
            citation_correctness = evaluator.score_citation_correctness(answer, mock_context)
            citation_completeness = evaluator.score_citation_completeness(answer)

            # Accumulate item metrics
            record = {
                "question": question,
                "category": category,
                "recall@5": recall_5,
                "precision@5": precision_5,
                "mrr": mrr,
                "ndcg@5": ndcg_5,
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "context_relevance": context_relevance,
                "citation_correctness": citation_correctness,
                "citation_completeness": citation_completeness,
                "latency_ms": latency_ms,
                "answer": answer,
            }
            all_results[mode].append(record)

    # Calculate aggregate summaries
    summary: Dict[str, Any] = {"status": "completed", "runs": {}}

    for mode in modes:
        records = all_results[mode]
        count = len(records)

        def avg(key: str) -> Any:
            vals = [r[key] for r in records if r[key] != "pending"]
            return round(sum(vals) / len(vals), 4) if vals else "pending"

        summary["runs"][mode] = {
            "avg_recall@5": avg("recall@5"),
            "avg_precision@5": avg("precision@5"),
            "avg_mrr": avg("mrr"),
            "avg_ndcg@5": avg("ndcg@5"),
            "avg_faithfulness": avg("faithfulness"),
            "avg_answer_relevance": avg("answer_relevance"),
            "avg_context_relevance": avg("context_relevance"),
            "avg_citation_correctness": avg("citation_correctness"),
            "avg_citation_completeness": avg("citation_completeness"),
            "avg_latency_ms": avg("latency_ms"),
        }

    # Write summary report JSON
    with open(RESULTS_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write detailed CSV results
    for mode in modes:
        csv_file = RESULTS_DIR / f"{mode}_detailed_results.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write Header
            writer.writerow(
                [
                    "Question",
                    "Category",
                    "Recall@5",
                    "Precision@5",
                    "MRR",
                    "NDCG@5",
                    "Faithfulness",
                    "Answer Relevance",
                    "Context Relevance",
                    "Citation Correctness",
                    "Citation Completeness",
                    "Latency (ms)",
                    "Answer",
                ]
            )
            # Write rows
            for r in all_results[mode]:
                writer.writerow(
                    [
                        r["question"],
                        r["category"],
                        r["recall@5"],
                        r["precision@5"],
                        r["mrr"],
                        r["ndcg@5"],
                        r["faithfulness"],
                        r["answer_relevance"],
                        r["context_relevance"],
                        r["citation_correctness"],
                        r["citation_completeness"],
                        r["latency_ms"],
                        r["answer"],
                    ]
                )

    print("\n" + "=" * 70)
    print(" Evaluation Complete. Results written to evaluation_results/")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run_evaluation())
