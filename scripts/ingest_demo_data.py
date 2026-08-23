#!/usr/bin/env python3
"""Interactive CLI demo for ingesting sample documents and testing deduplication."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.database.session import AsyncSessionLocal, check_database_health
from app.rag.embeddings.factory import get_embedding_provider
from app.services.ingestion_service import DocumentIngestionService
from app.services.vector_store import get_vector_store


async def main() -> None:
    print("=" * 70)
    print("🛡️  SentinelRAG — Document Ingestion & Deduplication Demo")
    print("=" * 70)

    # 1. Check Infrastructure
    print("\n[1/4] Checking backing infrastructure...")
    db_health = await check_database_health()
    print(f"  • PostgreSQL Status: {db_health.get('status')} (Connected: {db_health.get('connected')})")

    vector_store = get_vector_store()
    vs_health = await vector_store.health_check()
    print(f"  • Qdrant Status:     {vs_health.get('status')} (Connected: {vs_health.get('connected')})")

    # 2. Initialize Ingestion Service
    embedding_provider = get_embedding_provider()
    print(f"  • Embedding Model:   {embedding_provider.model_name} ({embedding_provider.dimension} dims)")

    ingestion_service = DocumentIngestionService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    # 3. Locate Sample Files
    data_dir = ROOT_DIR / "data" / "raw"
    sample_files = [
        data_dir / "system_spec.md",
        data_dir / "release_notes.txt",
        data_dir / "sample_paper.pdf",
    ]

    print("\n[2/4] Ingesting sample documents from data/raw/ ...")
    async with AsyncSessionLocal() as db:
        workspace = await ingestion_service.get_or_create_workspace(db, name="demo_workspace")
        print(f"  • Using Workspace: '{workspace.name}' (ID: {workspace.id})")

        for file_path in sample_files:
            if not file_path.exists():
                print(f"  ⚠️ File not found: {file_path.name}")
                continue

            with open(file_path, "rb") as f:
                content = f.read()

            print(f"\n  📄 Processing: {file_path.name} ({len(content)} bytes)")
            result = await ingestion_service.ingest_document(
                db=db,
                filename=file_path.name,
                content=content,
                workspace_id=workspace.id,
            )

            status_symbol = "✅" if result["status"] == "indexed" else "🔁"
            print(f"     Status: {status_symbol} {result['status'].upper()}")
            print(f"     Doc ID: {result.get('document_id')}")
            print(f"     Title:  {result.get('title')}")
            print(f"     Chunks: {result.get('chunk_count')}")
            print(f"     Hash:   {result.get('content_hash')[:16]}...")

        # 4. Demonstrate Deduplication
        print("\n[3/4] Testing Content Deduplication (re-ingesting system_spec.md)...")
        spec_path = data_dir / "system_spec.md"
        with open(spec_path, "rb") as f:
            dup_content = f.read()

        dup_result = await ingestion_service.ingest_document(
            db=db,
            filename="system_spec_copy.md",
            content=dup_content,
            workspace_id=workspace.id,
        )
        print(f"  • Deduplication Result: {dup_result['status'].upper()}")
        print(f"  • Matched Doc ID:       {dup_result.get('document_id')}")
        print(f"  • Message:              {dup_result.get('message')}")

        # 5. List Ingested Documents
        print("\n[4/4] Listing all workspace documents...")
        docs = await ingestion_service.list_documents(db, workspace_id=workspace.id)
        for i, d in enumerate(docs, 1):
            print(f"  {i}. [{d.source_type.upper()}] {d.title} (Chunks: {d.chunk_count}, ID: {d.id})")

    print("\n" + "=" * 70)
    print("✨ Ingestion demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
