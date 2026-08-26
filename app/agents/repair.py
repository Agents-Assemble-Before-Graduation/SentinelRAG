"""Repair Engine: implements RAG query rewrites, expansions, and retrieval strategy shifts."""

import re
from typing import Any, Dict, List

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider, Message, MessageRole
from app.llm.factory import get_llm_provider

logger = get_logger(__name__)

REWRITE_SYSTEM_PROMPT = """You are the SentinelRAG Query Reformulator. Your task is to analyze the original query and the verification issues identified, and generate a new, optimized search query to retrieve the missing information.

Focus on extracting the key entities, missing facts, and technical keywords. Do not include questions words like "how", "what", or "why".

Output only the new search query. Do not include markdown formatting or extra conversational text.
"""


class RepairEngine:
    """Agent that performs query rewrites, expansion, and weight adjustments."""

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm = llm_provider or get_llm_provider()

    def _heuristics_rewrite(self, question: str, issues: List[str]) -> str:
        """Basic keyword append fallback when LLM is unavailable or in mock mode."""
        if not issues:
            return question

        # Extract words from issues to append as query terms
        all_terms = " ".join(issues).lower()
        cleaned_terms = re.findall(r"\b[a-zA-Z]{4,}\b", all_terms)
        
        # Stop words list
        stop_words = {
            "found",
            "claim",
            "claims",
            "issue",
            "issues",
            "unsupported",
            "contradicted",
            "context",
            "evidence",
            "about",
            "there",
            "directly",
            "where",
            "verify",
        }
        keywords = [t for t in cleaned_terms if t not in stop_words]
        
        unique_keywords = list(dict.fromkeys(keywords))[:3]
        if unique_keywords:
            return f"{question} {' '.join(unique_keywords)}"
        return question

    async def rewrite_query(self, question: str, issues: List[str]) -> str:
        """Use LLM to reformulate query keywords based on the critic's issues."""
        if not issues:
            return question

        if not self.llm or not self.llm.is_available() or self.llm.provider_name == "mock":
            logger.info("Using local query rewrite heuristics (LLM unavailable/mock).")
            return self._heuristics_rewrite(question, issues)

        try:
            issues_str = "\n".join(f"- {i}" for i in issues)
            messages = [
                Message(role=MessageRole.SYSTEM, content=REWRITE_SYSTEM_PROMPT),
                Message(
                    role=MessageRole.USER,
                    content=f"Original Query: '{question}'\nIssues identified:\n{issues_str}",
                ),
            ]

            resp = await self.llm.chat_complete(messages, temperature=0.0)
            return resp.content.strip()
        except Exception as e:
            logger.warning("Query Reformulator encountered error: %s. Using heuristics.", str(e))
            return self._heuristics_rewrite(question, issues)

    def shift_strategy(self, current_strategy: str) -> str:
        """Cycle the retrieval strategy to try different retrieval modalities."""
        strategies = ["dense", "hybrid", "bm25"]
        try:
            idx = strategies.index(current_strategy.lower())
            next_idx = (idx + 1) % len(strategies)
            return strategies[next_idx]
        except ValueError:
            return "hybrid"
