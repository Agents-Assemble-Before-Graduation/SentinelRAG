# SentinelRAG System Specification

## 1. System Overview
SentinelRAG is a self-improving multi-agent retrieval-augmented generation platform. Unlike single-pass RAG pipelines, SentinelRAG decomposes user queries into multi-hop retrieval plans, critiques candidate responses, extracts atomic propositions, verifies evidence against retrieved source chunks, and triggers dynamic repair loops or safe termination when evidence is lacking.

## 2. Hybrid Retrieval Architecture
The retrieval layer combines dense semantic vector retrieval via Qdrant and sparse lexical retrieval using BM25. Results are merged via Reciprocal Rank Fusion (RRF) and scored through a cross-encoder reranker to guarantee high precision top-K passage selection.

## 3. Claim Extraction and Verification
Every generated answer is parsed into atomic claims by the Claim Extractor. Each claim is evaluated against the grounded retrieved text by the Evidence Verifier. A confidence score between 0.0 and 1.0 is computed for each claim.

## 4. Safe Termination Engine
When total confidence falls below the strict safety threshold or critical evidence is missing, the system terminates execution with a safe informative refusal rather than outputting potentially hallucinated claims.
