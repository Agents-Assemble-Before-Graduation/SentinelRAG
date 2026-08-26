# SentinelRAG 🛡️

> **Self-Improving Multi-Agent RAG with Continuous Evidence Verification, Self-Critique, Dynamic Repair, and Safe Termination.**

---

## 📌 Project Overview & Objective

**SentinelRAG** is a serious research-grade AI/ML system designed to overcome critical failure modes of traditional Retrieval-Augmented Generation (RAG)—namely hallucinations, missing evidence, ungrounded extrapolations, and silent failures.

Instead of a single-pass prompt-and-generate approach, SentinelRAG orchestrates an active multi-agent pipeline:

```
User Query
    │
    ▼
[ Planner Agent ]
    │
    ▼
[ Hybrid Retrieval Engine ] ──── (Dense + Sparse BM25 / Qdrant)
    │
    ▼
[ Cross-Encoder Reranker ]
    │
    ▼
[ Candidate Generator ]
    │
    ▼
[ Critic Agent ] ────────────── (Critiques coherence, coverage, logic)
    │
    ▼
[ Claim Extractor ] ─────────── (Decomposes output into verifiable atomic claims)
    │
    ▼
[ Evidence Verifier ] ───────── (Cross-examines claims against retrieved chunks)
    │
   ┌┴────────────────────────┐
   ▼                         ▼
[ Verified ✅ ]        [ Unverified / Flawed ❌ ]
   │                         │
   │               ┌─────────┴─────────┐
   │               ▼                   ▼
   │      [ Retries Left ]     [ Insufficient Evidence ]
   │               │                   │
   │               ▼                   ▼
   │      [ Repair Agent ]     [ Safe Termination / KILL 🛑 ]
   │               │
   │               └───► (Loop back to Generator)
   ▼
[ Final Judge ]
   │
   ▼
[ Experience Memory ] ──────── (Stores successful patterns & failure lessons)
    │
    ▼
Final Verified Answer
```

---

## 🚦 Current Phase & Status

- **Current Phase:** `Phase 6: Critic, Claim Verification, Repair Loop & Fail-Safe Kill`
- **Status:** **Completed & Operational ✅**
- **Highlights:**
  - Automated atomic claim extraction and fact verification against context excerpts with structured evaluation statuses.
  - Multi-agent Critic loops assessing answer completeness, support, contradictions, and choosing ACCEPT/REPAIR/KILL pathways.
  - Multi-turn repair node executing query rewrites, retrieval expansion (top-k), and retrieval strategy shifts.
  - Deterministic loop protection limiting retries to `MAX_REPAIR_ATTEMPTS = 2` to prevent infinite LLM execution.
  - Fail-safe Kill node refusing answer generation safely in insufficient-evidence scenarios to eliminate hallucination risk.
  - Real-time Guardrails & Verification Streamlit telemetry panel exposing Critic PASS/FAIL, Evidence coverage %, System confidence estimate %, Repair attempts, and Final decisions.
  - Interactive demonstration runner (`scripts/demo_failsafe_kill.py`) showcasing execution aborts on insufficient-evidence queries.
  - Comprehensive unit test coverage covering loop successes, loop failures, empty searches, contradiction kills, and prompt injection data isolation (81 tests passing).

---

## 🏗️ High-Level Architecture

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **API Backend** | FastAPI / Uvicorn | Async REST API, document ingestion, orchestration |
| **Application UI** | Streamlit | Real-time diagnostics dashboard and document management |
| **Ingestion Pipeline**| PyPDF, Python-Docx | Structure-preserving text extractors |
| **Chunking Engine**| StructureAwareChunker | Section/page-preserving overlapping text segmentation |
| **Local Embeddings** | FastEmbed (ONNX) | Fast local vector embeddings (BAAI/bge-small-en-v1.5) |
| **Relational DB** | PostgreSQL 16 | Metadata: workspaces, documents, chunks, audit logs |
| **Vector DB** | Qdrant | Approximate nearest neighbor vector search (`sentinel_chunks`) |
| **ORM & Migrations**| SQLAlchemy 2.0 + Alembic | Type-safe database mapping and schema revisions |
| **Testing** | Pytest / Pytest-Asyncio | Automated unit and integration testing |

---

## 🛠️ Local Development Setup

### Prerequisites

- **Python**: 3.11 or higher
- **Docker & Docker Compose**: For local PostgreSQL and Qdrant services

### 1. Clone & Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd "SentinelRAG II"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

---

## 🐳 Docker Infrastructure Setup

To launch PostgreSQL and Qdrant locally using Docker Compose:

```bash
# Start background services
docker compose up -d postgres qdrant

# Check service health
docker compose ps
```

---

## 🚀 Running SentinelRAG

### 1. Start FastAPI Backend

```bash
./scripts/run_backend.sh
```

- **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Liveness Probe:** `GET http://localhost:8000/health`
- **Readiness Probe:** `GET http://localhost:8000/ready`
- **Document Management:** `/api/v1/documents`

### 2. Ingest Sample Demo Documents

```bash
python scripts/ingest_demo_data.py
```

### 3. Start Streamlit Frontend

```bash
./scripts/run_frontend.sh
```

- **Streamlit Web UI:** [http://localhost:8501](http://localhost:8501)

---

## 📄 Document Ingestion API Examples

### Upload a Document
```bash
curl -X POST http://localhost:8000/api/v1/documents \
     -F "file=@data/raw/system_spec.md"
```

### List Ingested Documents
```bash
curl -X GET http://localhost:8000/api/v1/documents
```

### Get Document Details & Chunks
```bash
curl -X GET http://localhost:8000/api/v1/documents/<DOCUMENT_UUID>
```

### Delete Document & Vectors
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/<DOCUMENT_UUID>
```

---

## 💬 RAG Query API Examples

### Execute Cited Query
```bash
curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is retrieval augmented generation?", "top_k": 3, "score_threshold": 0.25}'
```

#### Example Output:
```json
{
  "answer": "Retrieval Augmented Generation (RAG) is a technique that combines retrieval models with generative LLMs [Evidence 1].",
  "sources": [
    {
      "document_title": "SentinelRAG Spec",
      "filename": "sys_spec.md",
      "page_number": 1,
      "section_heading": "Introduction",
      "chunk_index": 0,
      "score": 0.8654,
      "document_id": "8a8342db-4fdc-4a37-97eb-30fbe8f1a141"
    }
  ],
  "retrieval_latency_ms": 25.4,
  "generation_latency_ms": 845.2,
  "total_latency_ms": 870.6,
  "model_used": "gpt-4o",
  "chunks_retrieved": 1,
  "context_chars": 234,
  "request_id": "ea3bc12",
  "grounded": true
}
```

---

## 📊 Running Evaluations

The Phase 4 evaluation pipeline benchmarks RAG performance (comparing `dense` and `hybrid` modes) across 8 query categories (factual, definition, comparison, multi-hop, summarization, numerical, ambiguous, and insufficient evidence).

### Run Benchmark Evaluation
Ensure Postgres and Qdrant are running and sample documents have been ingested, then execute:
```bash
python scripts/evaluate_rag.py
```

Results are stored in the following files:
- `evaluation_results/evaluation_summary.json` (contains aggregate run statistics)
- `evaluation_results/dense_detailed_results.csv` (detailed per-query metrics for dense retrieval)
- `evaluation_results/hybrid_detailed_results.csv` (detailed per-query metrics for hybrid retrieval)

---

## 🧪 Running Tests

```bash
# Run all unit and integration tests
pytest

# Run tests with code coverage
pytest --cov=app tests/

# Run code linter
ruff check .
```
