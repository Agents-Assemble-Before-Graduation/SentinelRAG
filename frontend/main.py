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
    st.markdown("<span class='badge-pill'>PHASE 2: INGESTION ACTIVE</span>", unsafe_allow_html=True)
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


# TAB 3: Query Playground (Preview)
with tab_query_preview:
    st.markdown("### 💬 Sentinel Query Playground")

    st.info(
        "ℹ️ **Phase 2 Complete:** Document ingestion, chunking, deduplication, and embeddings are active. "
        "Multi-agent reasoning loops (Planner, Critic, Claim Verification, Repair) will be integrated in Phase 3 & 4."
    )

    query_text = st.text_area(
        "Enter test research query:",
        placeholder="e.g., What are the performance trade-offs of speculative decoding in MoE models?",
        height=100,
    )

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        mode = st.selectbox(
            "Execution Pipeline Mode:",
            [
                "Full Sentinel Multi-Agent Pipeline (Phase 3+)",
                "Direct Hybrid Retrieval (Phase 3+)",
                "Direct LLM Baseline",
            ],
            index=0,
        )
    with col_opt2:
        confidence_threshold = st.slider("Verification Confidence Threshold:", 0.0, 1.0, 0.85, 0.05)

    execute_btn = st.button("🚀 Run Pipeline", type="primary", disabled=True)

    st.caption("🔒 Execution disabled during Phase 2 Ingestion mode.")


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
