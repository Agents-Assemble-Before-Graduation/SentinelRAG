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

- **Current Phase:** `Phase 2: Document Ingestion, Chunking, Embeddings, PostgreSQL & Qdrant`
- **Status:** **Completed & Operational ✅**
- **Highlights:**
  - Complete ingestion pipeline supporting PDF, Markdown, Plain Text, and Word DOCX.
  - Structure-aware chunking preserving section headings, outline trails, and page numbers.
  - Deterministic SHA-256 content deduplication preventing redundant storage and vector compute.
  - Local ONNX embedding provider (`FastEmbed` with `BAAI/bge-small-en-v1.5`, 384 dims, zero API keys required).
  - PostgreSQL relational models (`DocumentRecord` + `DocumentChunkRecord`).
  - Qdrant vector database collection management and chunk indexing.
  - FastAPI document CRUD API (`POST`, `GET`, `DELETE` on `/api/v1/documents`).
  - Streamlit UI with live document uploader, corpus browser, and diagnostics.
  - 43 automated unit and integration tests passing with 100% success rate.

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

## 🧪 Running Tests

```bash
# Run all unit and integration tests
pytest

# Run tests with code coverage
pytest --cov=app tests/

# Run code linter
ruff check .
```
