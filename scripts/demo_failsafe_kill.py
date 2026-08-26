"""Demo script demonstrating the SentinelRAG Phase 6 Fail-Safe Kill mechanism on insufficient evidence queries."""

import asyncio
import uuid
from unittest.mock import AsyncMock

from app.services.query_service import RAGQueryService
from app.rag.retrieval.retriever import RetrievedChunk
from app.rag.generation.generator import GenerationResult
from app.llm.mock_provider import MockLLMProvider
from app.rag.retrieval.base import BaseRetriever


class DemoMockRetriever(BaseRetriever):
    """Mock retriever returning a document block that lacks the requested security threshold info."""
    async def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0, workspace_id: str | None = None) -> list:
        # Returns general system spec info but lacks safety threshold values
        return [
            RetrievedChunk(
                chunk_id="chunk_spec_1",
                content="SentinelRAG incorporates a Safe Termination Engine that monitors system confidence. If confidence drops, execution terminates.",
                score=0.9,
                document_id="doc_spec",
                document_title="SentinelRAG System Specification",
                filename="system_spec.md",
                page_number=1
            )
        ]


async def run_demo():
    print("=" * 80)
    print(" SentinelRAG Phase 6 Demo: Fail-Safe Kill Guardrail Activation")
    print("=" * 80)
    print("Query: 'What is the security threshold value for the Safe Termination Engine?'")
    print("Scenario: The document contains general info about the engine but lacks the threshold value.")
    print("-" * 80)

    # 1. Setup services with mock LLM generator representing a model attempting to guess/hallucinate
    mock_llm = MockLLMProvider(
        response_text="The security threshold value for the Safe Termination Engine is 0.85."
    )
    
    # We inject the demo mock retriever that lacks the threshold details
    demo_retriever = DemoMockRetriever()
    
    query_service = RAGQueryService(
        embedding_provider=None,
        vector_store=None,
        llm_provider=mock_llm,
    )
    query_service._dense_retriever = demo_retriever
    query_service._bm25_retriever = demo_retriever
    query_service._hybrid_retriever = demo_retriever

    # Create mock DB session
    mock_db = AsyncMock()

    print("🚀 Invoking Multi-Agent State Graph...")
    result = await query_service.query(
        question="What is the security threshold value for the Safe Termination Engine?",
        db=mock_db,
        workspace_id=str(uuid.uuid4()),
        retrieval_mode="dense"
    )

    print("\n🏁 Execution Pipeline Completed!")
    print("-" * 80)
    print(f"Final Decision:   {result.metadata.get('final_decision').upper()}")
    print(f"Critic Score:     {int(result.metadata.get('critic_score') * 100)}%")
    print(f"System Confidence:{int(result.metadata.get('confidence') * 100)}%")
    print(f"Repair Attempts:  {result.metadata.get('retry_count')}")
    print(f"Issues Logged:    {result.metadata.get('issues')}")
    print("-" * 80)
    print("Answer Returned to User:")
    print(f"\033[91m{result.answer}\033[0m")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_demo())
