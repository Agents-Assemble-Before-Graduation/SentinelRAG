# SentinelRAG — Development Status

## Current Status: Phase 5 (Complete)

**Last Updated:** August 2026
**Target Architecture:** Local-First Self-Improving Multi-Agent RAG

---

## 📊 Phase Breakdown

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Foundation, Architecture & Local Development Environment | **Completed ✅** |
| **Phase 2** | Document Ingestion, Chunking, Embeddings, PostgreSQL & Qdrant | **Completed ✅** |
| **Phase 3** | Conventional RAG Baseline (Retrieval, Context, Generation & Query API) | **Completed ✅** |
| **Phase 4** | Hybrid Retrieval, Reranking & Research Evaluation Baseline | **Completed ✅** |
| **Phase 5** | Multi-Agent Reasoning Loops (Planner & Generator orchestration via LangGraph) | **Completed ✅** |
| **Phase 6** | Multi-Agent Refinement (Critic, Claim Extractor, Evidence Verifier, Repair) | **Next Up ⏳** |
| **Phase 7** | Experience Memory, Self-Improvement Loop & Continuous Evaluation | Planned ⏳ |

---

## ✅ Completed in Phase 5

1. **Typed Agent State Graph (`app/agents/state.py`):**
   - Created `AgentState` TypedDict to capture data across all pipeline steps: question, classification, plan, subquestions, candidate document pools, context text, generation, and telemetry.

2. **Planner Agent (`app/agents/planner.py`):**
   - Implemented `PlannerAgent` performing query classification (factual, definition, comparison, etc.) and selecting optimal retrieval strategies (dense, BM25, hybrid) via structured JSON prompting or local rule-based heuristics fallback.

3. **Generator Agent (`app/agents/generator.py`):**
   - Implemented `GeneratorAgent` wrapping the existing grounded `RAGGenerator` engine without duplicating retrieval or prompt logic.

4. **LangGraph Orchestrator Workflow (`app/agents/graph.py`):**
   - Constructed the sequential state graph: `START -> Planner -> Retrieval -> Reranking -> Context Builder -> Generator -> END`.
   - Enabled flexible dependency injection via `RunnableConfig` for test isolation and runtime configuration overrides.

5. **Playground UI Timeline (`frontend/main.py`):**
   - Rendered a horizontal Flexbox pipeline execution timeline showing Planning, Retrieval, Reranking, and Generation stages along with their exact latencies.

6. **Agent Unit Test Suite (`tests/unit/test_agents.py`):**
   - Added tests covering planner rule heuristics, LLM JSON parsing, generator mappings, node transitions, and full graph execution pipelines (76 tests passing).

---

## ✅ Completed in Phase 4

1. **BM25 Keyword Retrieval (`app/rag/retrieval/bm25.py`):**
   - Implemented `BM25Retriever` executing exact keyword search using in-memory BM25 engines over document chunks.
   - Built automatic database chunk caching keyed by `(workspace_id, db_chunk_count)` for zero-compute query reuse.

2. **Weighted Score Fusion (`app/rag/retrieval/hybrid.py`):**
   - Implemented `HybridRetriever` combining dense and BM25 result pools.
   - Built a Min-Max score normalizer to translate different search score scales before weighted linear combination.

3. **Cross-Encoder Reranking (`app/rag/retrieval/reranker.py`):**
   - Integrated `FastEmbedReranker` running local ONNX cross-encoders (`Xenova/ms-marco-MiniLM-L-6-v2`) for re-scoring candidates.
   - Created offline-safe `MockReranker` using term-overlaps, avoiding network calls during tests.

4. **Retriever Interfaces & Ablation Support:**
   - Established unified `BaseRetriever` ABC and `BaseReranker` ABC interfaces to support clean modular swapping for research ablation experiments.

5. **Benchmark Evaluation Dataset (`data/evaluation/benchmark.json`):**
   - Created a seed dataset with 8 questions covering: factual, definition, comparison, multi-hop, summarization, numerical, ambiguous, and insufficient evidence categories.

6. **Retrieval Metrics Suite (`app/evaluation/metrics.py`):**
   - Built calculators for standard search quality metrics: Recall@K, Precision@K, MRR, and NDCG@K.

7. **Generation Metrics & Citations (`app/evaluation/generation_metrics.py`):**
   - Implemented LLM-as-judge prompts for faithfulness and relevance metrics (clearly documenting judge limitations).
   - Built heuristic verifiers for sentence-level citation correctness and citation completeness.

8. **Automated Experiment Runner (`scripts/evaluate_rag.py`):**
   - Created an automated evaluation script that compares RAG modes on the benchmark and saves detailed CSV results to `evaluation_results/`.

9. **Playground UI Controls:**
   - Added selectboxes for dense/BM25/hybrid modes, checkboxes for Cross-Encoder Reranking, and detailed score telemetry cards to Streamlit.

---

## ✅ Completed in Phase 3

1. **LLM Abstraction Layer (`app/llm/`):**
   - Implemented provider-agnostic `BaseLLMProvider` interface.
   - Built direct REST API-based `OpenAIProvider` using `httpx` (requires zero extra client library dependencies).
   - Created `MockLLMProvider` for isolated, repeatable, zero-cost unit and E2E testing.
   - Built a provider-resolving `get_llm_provider` singleton factory.

2. **Dense Retriever (`app/rag/retrieval/`):**
   - Built `DenseRetriever` to embed queries and perform ANN search on Qdrant.
   - Preserves complete document provenance metadata (page number, section headings, document ID, score).
   - Configurable per-query overrides for `top_k` and `score_threshold`.

3. **Context Builder (`app/rag/context/`):**
   - Created `ContextBuilder` that removes duplicate chunks based on ID and content fingerprints.
   - Sorts chunks by similarity score, formats them into structured evidence blocks, and enforces strict character limits.

4. **Grounded Generator (`app/rag/generation/`):**
   - Configured prompt templates (`prompts/rag_system.txt`, `prompts/rag_user.txt`) that treat retrieved passages as data.
   - Short-circuits empty context queries directly to a safe "insufficient evidence" response to prevent hallucinations.
   - Standardized temperature to `0.0` for maximum grounding.

5. **FastAPI Query API (`POST /api/v1/query`):**
   - Implemented query endpoint return schemas containing the answer, source citations, latency breakdowns, and context stats.
   - Automatically writes audit telemetry records to the PostgreSQL `QueryLog` table.

6. **Streamlit UI Integration:**
   - Updated the Query Playground tab in `frontend/main.py` to allow live queries.
   - Displays query answers, source files, page numbers, section headings, and retrieval/generation latency metrics.

7. **Verification & Test Suite:**
   - Created 18 new automated unit, integration, and E2E pipeline tests.
   - Total test suite counts 61 passing tests with 100% success rate.

---

## ✅ Completed in Phase 2

1. **Document Validation & Security:**
   - Multi-format validation supporting PDF, Markdown, Plain Text, and Word DOCX (`app/rag/ingestion/validator.py`).
   - Strict filename sanitization preventing path traversal (`../`, control chars, null bytes).
   - Magic bytes verification for binary formats (`%PDF-`, `PK\x03\x04`).

2. **Multi-Format Extraction:**
   - `PDFExtractor`: Page-by-page extraction preserving page numbers and page-level metadata using `pypdf`.
   - `MarkdownExtractor`: Preserves heading hierarchy (`#`, `##`, `###`) and structure breadcrumbs.
   - `PlainTextExtractor`: Normalizes paragraphs and text formatting.
   - `DocxExtractor`: Parses Word document headings and paragraphs using `python-docx`.

3. **Structure-Aware Chunking:**
   - `StructureAwareChunker` (`app/rag/ingestion/chunker.py`) with configurable `chunk_size`, `chunk_overlap`, and `min_chunk_size`.
   - Splits on paragraph and sentence boundaries, preserving page number and section heading metadata on every chunk.

4. **SHA-256 Content Deduplication:**
   - Generates deterministic SHA-256 hashes across extracted document content (`app/rag/ingestion/deduplication.py`).
   - Prevents duplicate document storage and redundant embedding computation within workspaces.

5. **Embedding Providers:**
   - `BaseEmbeddingProvider` abstraction (`app/rag/embeddings/base.py`).
   - `FastEmbedEmbeddingProvider`: Local ONNX-based embedding engine using `BAAI/bge-small-en-v1.5` (384 dims, fast CPU execution, zero API keys).
   - `DeterministicEmbeddingProvider`: Zero-download deterministic provider for test isolation.
   - Factory pattern provider resolution (`app/rag/embeddings/factory.py`).

6. **PostgreSQL Relational Schema:**
   - `DocumentRecord`: File metadata, title, source type, SHA-256 hash, file size, chunk counts, and status (`app/models/base.py`).
   - `DocumentChunkRecord`: Chunk indices, text contents, token estimates, page numbers, section headings, and JSONB metadata.

7. **Qdrant Vector Indexing:**
   - Dynamic collection management with configurable vector dimensions (`sentinel_chunks`).
   - Point insertion with payload metadata (`chunk_id`, `document_id`, `workspace_id`, `page_number`, `section_heading`, `content`).
   - Document-level vector deletion filtering on `document_id`.

8. **Ingestion Coordinator & REST APIs:**
   - `DocumentIngestionService` (`app/services/ingestion_service.py`): Full pipeline coordinator.
   - FastAPI Document CRUD routes (`app/api/v1/documents.py`):
     - `POST /api/v1/documents`: Ingest uploaded files.
     - `GET /api/v1/documents`: List indexed documents.
     - `GET /api/v1/documents/{id}`: Detailed document metadata and chunk breakdown.
     - `DELETE /api/v1/documents/{id}`: Cascade deletion from PostgreSQL & Qdrant.

9. **Demo Data & Testing:**
   - Real sample documents (`data/raw/sample_paper.pdf`, `data/raw/system_spec.md`, `data/raw/release_notes.txt`).
   - CLI demo script (`scripts/ingest_demo_data.py`).
   - 43 automated unit and integration tests passing with 100% success rate.

---

## 🎯 Recommended Next Phase (Phase 4)

**Phase 4: Multi-Agent Reasoning Loops (Planner, Critic, Claim Extractor, Evidence Verifier, Repair)**
- Planner Agent that decomposes complex query objectives.
- Critic Agent reviewing logical coherence, completeness, and unsupported assertions.
- Claim Extractor decomposing generated candidate answers into atomic propositions.
- Evidence Verifier cross-checking atomic claims against retrieved chunks with confidence scoring.
- Repair Agent executing context expansion or query rewrite iterations when claims fail verification.

