"""SentinelRAG Streamlit User Interface — Phase 2 Ingestion & Diagnostics."""

import os
from typing import Any

import httpx
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="SentinelRAG | Self-Improving Multi-Agent RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        background-color: #1e293b;
        border: 1px solid #334155;
        margin-bottom: 0.75rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #0f172a;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_backend_url() -> str:
    """Retrieve backend API base URL from environment or session state."""
    return st.session_state.get(
        "backend_url", os.getenv("BACKEND_API_URL", "http://localhost:8000")
    )


def fetch_backend_health(base_url: str) -> dict[str, Any] | None:
    """Fetch backend liveness status."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url}/health")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def fetch_backend_readiness(base_url: str) -> dict[str, Any] | None:
    """Fetch downstream infrastructure readiness status."""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{base_url}/ready")
            return resp.json()
    except Exception:
        pass
    return None


def fetch_documents(base_url: str) -> list[dict[str, Any]]:
    """Fetch indexed documents list from backend."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/api/v1/documents")
            if resp.status_code == 200:
                return resp.json().get("documents", [])
    except Exception:
        pass
    return []


# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ SentinelRAG")
    st.markdown("<span class='badge-pill'>PHASE 5: LANGGRAPH AGENT ACTIVE</span>", unsafe_allow_html=True)
    st.markdown("---")

    backend_url = st.text_input("Backend API Endpoint", value=get_backend_url())
    st.session_state["backend_url"] = backend_url

    if st.button("🔄 Refresh System Status", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📡 System Status")

    health_data = fetch_backend_health(backend_url)
    readiness_data = fetch_backend_readiness(backend_url)

    if health_data:
        st.success(f"Backend: **{health_data.get('status', 'healthy').upper()}**")
    else:
        st.error("Backend: **OFFLINE / UNREACHABLE**")

    if readiness_data:
        components = readiness_data.get("components", {})
        db_info = components.get("database", {})
        qdrant_info = components.get("vector_store", {})

        db_icon = "🟢" if db_info.get("connected") else "🔴"
        qdrant_icon = "🟢" if qdrant_info.get("connected") else "🔴"

        st.markdown(
            f"**PostgreSQL**: {db_icon} `{db_info.get('status', 'unknown')}` "
            f"({db_info.get('latency_ms', '-')} ms)"
        )
        st.markdown(
            f"**Qdrant Vector DB**: {qdrant_icon} `{qdrant_info.get('status', 'unknown')}` "
            f"({qdrant_info.get('latency_ms', '-')} ms)"
        )
    else:
        st.caption("Waiting for backend connection to probe infrastructure...")

    st.markdown("---")
    st.caption("SentinelRAG Research Environment — Local-First")


# Main Page Content
st.markdown("<div class='main-title'>🛡️ SentinelRAG Platform</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Self-Improving Multi-Agent RAG with Continuous Verification, Critique, and Dynamic Repair</div>",
    unsafe_allow_html=True,
)

tab_overview, tab_documents, tab_query_preview, tab_architecture = st.tabs(
    [
        "📊 System & Diagnostics",
        "📚 Ingestion & Documents",
        "💬 Query Playground (Preview)",
        "🗺️ Architecture & Roadmap",
    ]
)

# TAB 1: System & Diagnostics
with tab_overview:
    st.markdown("### 🔍 Infrastructure & Service Diagnostics")

    col1, col2, col3 = st.columns(3)
    with col1:
        if health_data:
            st.metric("FastAPI Backend", "Online", f"v{health_data.get('version', '0.1.0')}")
        else:
            st.metric("FastAPI Backend", "Offline", "Check service", delta_color="inverse")

    with col2:
        if readiness_data and readiness_data.get("components", {}).get("database", {}).get("connected"):
            lat = readiness_data["components"]["database"].get("latency_ms", 0)
            st.metric("PostgreSQL Database", "Connected", f"{lat} ms")
        else:
            st.metric("PostgreSQL Database", "Disconnected", "Docker service required", delta_color="inverse")

    with col3:
        if readiness_data and readiness_data.get("components", {}).get("vector_store", {}).get("connected"):
            lat = readiness_data["components"]["vector_store"].get("latency_ms", 0)
            st.metric("Qdrant Vector DB", "Connected", f"{lat} ms")
        else:
            st.metric("Qdrant Vector DB", "Disconnected", "Docker service required", delta_color="inverse")

    st.markdown("---")
    st.markdown("#### Live Diagnostic Response")
    if readiness_data:
        st.json(readiness_data)
    else:
        st.warning(
            "Could not connect to FastAPI backend at "
            f"`{backend_url}`. Start the backend with: `./scripts/run_backend.sh`"
        )


# TAB 2: Document Ingestion & Corpus
with tab_documents:
    st.markdown("### 📚 Document Ingestion & Corpus Explorer")

    col_up, col_list = st.columns([1, 2])

    with col_up:
        st.markdown("#### 📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=["pdf", "md", "txt", "docx"],
            help="Supported formats: PDF, Markdown, Plain Text, Word DOCX",
        )

        if uploaded_file and st.button("🚀 Ingest Document", type="primary", use_container_width=True):
            with st.spinner("Processing, chunking, and embedding..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.post(f"{backend_url}/api/v1/documents", files=files)
                        if resp.status_code == 201:
                            data = resp.json()
                            if data.get("status") == "duplicate":
                                st.warning(f"🔁 Duplicate detected: {data.get('message')}")
                            else:
                                st.success(
                                    f"✅ Ingested '{data.get('filename')}' ({data.get('chunk_count')} chunks)"
                                )
                            st.json(data)
                        else:
                            st.error(f"Ingestion failed ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")

    with col_list:
        st.markdown("#### 🗄️ Indexed Documents")
        docs = fetch_documents(backend_url)

        if docs:
            st.caption(f"Total documents: {len(docs)}")
            for doc in docs:
                with st.expander(f"📄 {doc.get('title')} ({doc.get('source_type', '').upper()})"):
                    st.markdown(f"**Filename**: `{doc.get('filename')}`")
                    st.markdown(f"**Document ID**: `{doc.get('id')}`")
                    st.markdown(f"**Chunks**: `{doc.get('chunk_count')}`")
                    st.markdown(f"**Content Hash**: `{doc.get('content_hash')}`")
                    st.markdown(f"**Ingested At**: `{doc.get('created_at')}`")
        else:
            st.info("No documents currently indexed. Ingest a document or run `python scripts/ingest_demo_data.py`.")


# TAB 3: Query Playground
with tab_query_preview:
    st.markdown("### 💬 Sentinel Query Playground (Baseline RAG)")

    st.info(
        "ℹ️ **Phase 3: Conventional RAG Baseline Active.** "
        "Direct retrieval, context construction, and LLM grounded generation are operational."
    )

    query_text = st.text_area(
        "Enter research query:",
        placeholder="e.g., What are the performance trade-offs of speculative decoding in MoE models?",
        height=100,
    )

    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
    with col_opt1:
        top_k = st.slider("Top K Chunks:", min_value=1, max_value=20, value=5)
    with col_opt2:
        score_threshold = st.slider("Threshold:", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    with col_opt3:
        retrieval_mode = st.selectbox("Retrieval Mode:", ["dense", "bm25", "hybrid"], index=0)
    with col_opt4:
        rerank_enabled = st.checkbox("Cross-Encoder Reranking", value=False)

    execute_btn = st.button("🚀 Run Query", type="primary")

    if execute_btn:
        if not query_text.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Executing RAG Pipeline..."):
                try:
                    payload = {
                        "question": query_text.strip(),
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "retrieval_mode": retrieval_mode,
                        "rerank_enabled": rerank_enabled,
                    }
                    with httpx.Client(timeout=60.0) as client:
                        resp = client.post(f"{backend_url}/api/v1/query", json=payload)
                        if resp.status_code == 200:
                            data = resp.json()
                            st.markdown("#### 📝 Answer")
                            st.write(data.get("answer"))

                            # Latency breakdown
                            st.markdown("---")
                            st.markdown("#### ⚡ Telemetry")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Retrieval Latency", f"{data.get('retrieval_latency_ms')} ms")
                            col2.metric("Generation Latency", f"{data.get('generation_latency_ms')} ms")
                            col3.metric("Total Latency", f"{data.get('total_latency_ms')} ms")
                            
                            meta = data.get("metadata", {})
                            col4.metric("Retrieval Mode", meta.get("retrieval_mode", retrieval_mode).upper())
                            
                            rerank_status = "ACTIVE" if meta.get("reranked", rerank_enabled) else "DISABLED"
                            col5.metric("Reranking", rerank_status)
                            
                            st.caption(f"Model used: `{data.get('model_used')}` | Chunks retrieved: `{data.get('chunks_retrieved')}` | Context size: `{data.get('context_chars')} chars` | Request ID: `{data.get('request_id')}`")

                            # Execution timeline
                            latencies = meta.get("latency_breakdown", {})
                            st.markdown("##### 🤖 Multi-Agent Execution Timeline")
                            stages_html = f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; background-color: #f0f2f6; padding: 12px; border-radius: 8px; margin-bottom: 15px;">
                                <div style="text-align: center; flex: 1;"><strong>🧠 Planning</strong><br><span style="color: #666; font-size: 11px;">{latencies.get('planning', 0.0)} ms</span></div>
                                <div style="color: #999;">➡️</div>
                                <div style="text-align: center; flex: 1;"><strong>🔍 Retrieval</strong><br><span style="color: #666; font-size: 11px;">{latencies.get('retrieval', 0.0)} ms</span></div>
                                <div style="color: #999;">➡️</div>
                                <div style="text-align: center; flex: 1;"><strong>⚡ Reranking</strong><br><span style="color: #666; font-size: 11px;">{latencies.get('reranking', 0.0)} ms</span></div>
                                <div style="color: #999;">➡️</div>
                                <div style="text-align: center; flex: 1;"><strong>✍️ Generation</strong><br><span style="color: #666; font-size: 11px;">{latencies.get('generation', 0.0)} ms</span></div>
                            </div>
                            """
                            st.markdown(stages_html, unsafe_allow_html=True)

                            # Multi-Agent Verification telemetry
                            st.markdown("##### 🛡️ Multi-Agent Guardrails & Verification")
                            
                            critic_decision = str(meta.get("final_decision", "ACCEPT")).upper()
                            critic_status = "🟢 PASS" if critic_decision == "ACCEPT" else "🔴 FAIL"
                            
                            verifications = meta.get("verification") or []
                            total_claims = len(verifications)
                            supported_claims = sum(1 for v in verifications if v.get("status") == "SUPPORTED")
                            partial_claims = sum(1 for v in verifications if v.get("status") == "PARTIALLY_SUPPORTED")
                            
                            coverage_pct = int(((supported_claims + 0.5 * partial_claims) / total_claims) * 100) if total_claims > 0 else 100
                            confidence_pct = int(meta.get("confidence", 1.0) * 100)
                            attempts = meta.get("retry_count", 0)
                            
                            col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
                            col_g1.metric("Critic Status", critic_status)
                            col_g2.metric("Evidence Coverage", f"{coverage_pct}%")
                            col_g3.metric("System Confidence", f"{confidence_pct}%")
                            col_g4.metric("Repair Attempts", f"{attempts}")
                            col_g5.metric("Final Decision", critic_decision)

                            # Sources and page numbers
                            st.markdown("---")
                            st.markdown("#### 📚 Sources")
                            sources = data.get("sources", [])
                            if sources:
                                for idx, src in enumerate(sources, start=1):
                                    page_info = f"Page {src.get('page_number')}" if src.get('page_number') is not None else "N/A"
                                    sec_info = src.get('section_heading') or "N/A"
                                    st.markdown(
                                        f"**{idx}. {src.get('document_title')}** "
                                        f"(File: `{src.get('filename')}`, {page_info}, Section: *{sec_info}*, Score: `{src.get('score')}`)"
                                    )
                            else:
                                st.info("No sources cited (possibly insufficient evidence).")
                        elif resp.status_code == 503:
                            detail = resp.json().get("detail", {})
                            st.error(f"LLM Provider Error: {detail.get('message')}")
                            st.info(detail.get("hint"))
                        else:
                            st.error(f"Query failed ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")


# TAB 4: Architecture & Roadmap
with tab_architecture:
    st.markdown("### 🗺️ Multi-Agent Architecture Pipeline")
    st.markdown(
        """
        ```mermaid
        graph TD
            User([User Query]) --> Planner[Planner Agent]
            Planner --> Retrieval[Hybrid Dense + Sparse Retrieval]
            Retrieval --> Reranker[Cross-Encoder Reranker]
            Reranker --> Generator[Candidate Generator]
            Generator --> Critic[Critic Agent]
            Critic --> Claims[Claim Extractor]
            Claims --> Verifier[Evidence Verifier]
            Verifier --> Decision{Claims Supported?}
            Decision -- No & Retries Left --> Repair[Repair Agent]
            Repair --> Generator
            Decision -- No & Insufficient Evidence --> Kill[Safe Termination / Kill]
            Decision -- Yes --> Judge[Final Judge]
            Judge --> Memory[Experience Memory]
            Memory --> Answer([Final Verified Response])
        ```
        """
    )

    st.markdown("#### 📋 Phase Roadmap")
    st.table(
        [
            {"Phase": "Phase 1", "Objective": "Foundation, Architecture & Local Dev Environment", "Status": "Completed ✅"},
            {"Phase": "Phase 2 (Current)", "Objective": "Document Ingestion, Chunking, Embeddings, PostgreSQL & Qdrant", "Status": "Completed ✅"},
            {"Phase": "Phase 3", "Objective": "Candidate Generator, Critic & Claim Verification Agents", "Status": "Planned ⏳"},
            {"Phase": "Phase 4", "Objective": "Repair Loop, Final Judge & Safe Termination Engine", "Status": "Planned ⏳"},
            {"Phase": "Phase 5", "Objective": "Experience Memory, Self-Improvement & Benchmarking", "Status": "Planned ⏳"},
        ]
    )
