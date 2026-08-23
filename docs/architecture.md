# SentinelRAG System Architecture Specification

## 1. Executive Architecture Summary

SentinelRAG is engineered as a local-first, multi-agent AI/ML platform that transforms standard passive RAG into an active, self-correcting verification and reasoning loop.

```
                      ┌─────────────────────────────────────────┐
                      │              Streamlit UI               │
                      │         (Diagnostics & Queries)         │
                      └────────────────────┬────────────────────┘
                                           │ HTTP
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │             FastAPI Backend             │
                      │    - Lifespan & Middleware Layer        │
                      │    - REST Endpoints (/health, /ready)   │
                      │    - Request ID Tracing & Redaction     │
                      └────────────┬───────────────┬────────────┘
                                   │               │
                     SQLAlchemy /  │               │ Async REST
                     Asyncpg       │               │
                                   ▼               ▼
                      ┌──────────────────┐   ┌──────────────────┐
                      │   PostgreSQL 16  │   │  Qdrant Vectors  │
                      │   - Workspaces   │   │  - Collections   │
                      │   - Document DB  │   │  - Embeddings    │
                      │   - Audit Logs   │   │  - Payloads      │
                      └──────────────────┘   └──────────────────┘
```

---

## 2. Component Design

### 2.1 Core & Configuration
- **Settings (`app/core/config.py`)**: Environment-driven with Pydantic Settings. Supports hot configuration for database URLs, Qdrant cluster hosts, LLM models, and log levels.
- **Logging (`app/core/logging.py`)**: Structured logs with `contextvars.ContextVar` Request ID binding. Sensitive keys (`api_key`, `token`, `password`, `Bearer`) are stripped before writing.

### 2.2 Database Layer
- **Engine (`app/database/session.py`)**: Asynchronous connection pool (`asyncpg`) with health check probes (`SELECT 1`).
- **Data Models (`app/models/base.py`)**:
  - `workspaces`: Multitenant/corpus isolation.
  - `documents`: Document ingestion metadata, source tracking, chunk counts.
  - `agent_runs`: Multi-agent pipeline execution runs, termination reasons, and audit metrics.
  - `query_logs`: Query telemetry, latency records, and verification decisions.
- **Migrations (`migrations/`)**: Managed by Alembic.

### 2.3 Vector Store Abstraction
- **Interface (`app/services/base_vector_store.py`)**: Defines `health_check()`, `create_collection()`, `collection_exists()`, `delete_collection()`, and `close()`.
- **Driver (`app/services/vector_store.py`)**: Implements `QdrantVectorStore` wrapping `AsyncQdrantClient`. Shields downstream application services from raw database client dependencies.

### 2.4 FastAPI Service
- **Liveness Probe (`/health`)**: Immediate 200 OK verifying the Python event loop and HTTP server are responsive.
- **Readiness Probe (`/ready`)**: Evaluates dependent services (PostgreSQL, Qdrant) concurrently, outputting detailed latency and connection statuses.

---

## 3. Future Multi-Agent Pipeline (Phases 2-5)

When fully implemented, the pipeline executes the following loop:
1. **Planner Agent**: Analyzes user query, decomposes into atomic retrieval sub-tasks, and generates search strategies.
2. **Hybrid Retrieval + Cross-Encoder Reranker**: Retrieves dense embeddings from Qdrant and sparse keywords via BM25, scoring top results through a cross-encoder reranker.
3. **Candidate Generator**: Synthesizes a preliminary answer strictly conditioned on retrieved context.
4. **Critic Agent**: Reviews logical coherence, completeness, and potential ungrounded claims.
5. **Claim Extractor**: Deconstructs candidate response into atomic propositions.
6. **Evidence Verifier**: Validates each claim against retrieved evidence chunks with confidence scores.
7. **Dynamic Repair Loop / Safe Termination**:
   - If claims fail verification and retries remain: activates **Repair Agent** to refine the answer.
   - If evidence is irrecoverably missing: triggers **Safe Termination (KILL)** to prevent hallucinations.
8. **Experience Memory**: Logs run trajectories, query embeddings, and critique lessons for continuous self-improvement.
