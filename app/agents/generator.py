"""Generator Agent: uses the existing grounded RAG generation system within the multi-agent workflow."""

from typing import Dict, Any
from app.agents.state import AgentState
from app.rag.generation.generator import RAGGenerator
from app.rag.context.builder import BuiltContext, SourceCitation
from app.core.logging import get_logger

logger = get_logger(__name__)


class GeneratorAgent:
    """Agent that wraps the existing RAG generation engine."""

    def __init__(self, generator: RAGGenerator | None = None) -> None:
        self.generator = generator or RAGGenerator()

    async def generate(self, state: AgentState) -> Dict[str, Any]:
        """Generate answer from state evidence context."""
        question = state["question"]
        context_text = state.get("context", "")
        documents = state.get("reranked_documents") or state.get("retrieved_documents") or []

        # Map document chunk info to expected SourceCitation objects
        citations = []
        for chunk in documents:
            citations.append(
                SourceCitation(
                    document_title=chunk.document_title,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                    document_id=chunk.document_id,
                )
            )

        # Build context object for generator call
        built_context = BuiltContext(
            context_text=context_text,
            sources=citations,
            total_chunks=len(documents),
            included_chunks=len(citations),
            total_chars=len(context_text),
            was_truncated=False,
        )

        logger.info(
            "Generator agent executing for question '%s...' with %d source documents.",
            question[:40],
            len(documents),
        )

        result = await self.generator.generate(question, built_context)

        # Convert citations back to serialisable dictionaries for QuerySource mapping
        serialised_sources = []
        for citation in result.sources:
            serialised_sources.append(
                {
                    "document_title": citation.document_title,
                    "filename": citation.filename,
                    "page_number": citation.page_number,
                    "section_heading": citation.section_heading,
                    "chunk_index": citation.chunk_index,
                    "score": round(citation.score, 4),
                    "document_id": citation.document_id,
                }
            )

        # Decide final answer and decision based on whether LLM grounded the response
        final_decision = "accept" if result.grounded else "refuse"
        confidence = 1.0 if result.grounded else 0.0

        return {
            "draft_answer": result.answer,
            "final_answer": result.answer,
            "sources": serialised_sources,
            "final_decision": final_decision,
            "confidence": confidence,
            "cost": 0.0,  # Telemetry tracking
        }
