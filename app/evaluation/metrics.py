"""Retrieval metrics calculation module."""

import math
from typing import List
from app.rag.retrieval import RetrievedChunk


def calculate_recall_at_k(
    retrieved: List[RetrievedChunk],
    ground_truth_doc: str,
    ground_truth_page: int | None = None,
    k: int = 5,
) -> float:
    """Recall@K: 1.0 if at least one hit is present in top K retrieved chunks, else 0.0."""
    sub_list = retrieved[:k]
    for chunk in sub_list:
        if chunk.filename.lower() == ground_truth_doc.lower():
            if ground_truth_page is None or chunk.page_number == ground_truth_page:
                return 1.0
    return 0.0


def calculate_precision_at_k(
    retrieved: List[RetrievedChunk],
    ground_truth_doc: str,
    ground_truth_page: int | None = None,
    k: int = 5,
) -> float:
    """Precision@K: Fraction of retrieved chunks in top K that are hits."""
    if not retrieved or k <= 0:
        return 0.0
    effective_k = min(len(retrieved), k)
    hits = 0
    for chunk in retrieved[:effective_k]:
        if chunk.filename.lower() == ground_truth_doc.lower():
            if ground_truth_page is None or chunk.page_number == ground_truth_page:
                hits += 1
    return hits / k


def calculate_mrr(
    retrieved: List[RetrievedChunk],
    ground_truth_doc: str,
    ground_truth_page: int | None = None,
) -> float:
    """MRR: Reciprocal of the rank of the first hit."""
    for rank_idx, chunk in enumerate(retrieved, start=1):
        if chunk.filename.lower() == ground_truth_doc.lower():
            if ground_truth_page is None or chunk.page_number == ground_truth_page:
                return 1.0 / rank_idx
    return 0.0


def calculate_ndcg_at_k(
    retrieved: List[RetrievedChunk],
    ground_truth_doc: str,
    ground_truth_page: int | None = None,
    k: int = 5,
) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain in top K."""
    effective_k = min(len(retrieved), k)
    if effective_k == 0:
        return 0.0

    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for rank_idx, chunk in enumerate(retrieved[:effective_k], start=1):
        is_hit = 0.0
        if chunk.filename.lower() == ground_truth_doc.lower():
            if ground_truth_page is None or chunk.page_number == ground_truth_page:
                is_hit = 1.0
        dcg += is_hit / math.log2(rank_idx + 1)

    # Calculate Ideal DCG (IDCG) - assuming at least one ideal hit exists at rank 1
    # since we evaluate queries with one primary ground truth page/doc source.
    idcg = 1.0 / math.log2(1 + 1)  # 1.0

    return dcg / idcg
