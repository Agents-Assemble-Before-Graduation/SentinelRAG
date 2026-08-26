"""Critic Agent: evaluates answer quality, support, and decides next actions (accept, repair, kill)."""

import json
import re
from typing import Any, Dict, List

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider, Message, MessageRole
from app.llm.factory import get_llm_provider

logger = get_logger(__name__)

CRITIC_SYSTEM_PROMPT = """You are the SentinelRAG Critic Agent. Your task is to evaluate the generated response, its verification audit, and decide if the response is fully supported and ready to return.

You must evaluate:
1. Factual support - Are there any unsupported claims?
2. Evidence coverage - Does the answer cover the required facts?
3. Relevance - Is the answer direct and helpful?
4. Hallucination - Are there any facts introduced not in the context?
5. Contradictions - Does the answer contradict itself or the context?

Select one of the decisions:
- `ACCEPT`: The answer is fully correct, supported, cited, and complete.
- `REPAIR`: The answer has minor issues, missing details, or lacks evidence for some claims, but can be improved by running a repair loop.
- `KILL`: The query is ambiguous, contradictory, or lacks any matching context whatsoever, and should terminate with a refusal to generate.

If you choose `REPAIR`, you must recommend one of the repair strategies:
- `QUERY_REWRITE`: Rewrite the search query to retrieve better documents.
- `RETRIEVAL_EXPANSION`: Retrieve more documents (increase top-k).
- `RETRIEVAL_WEIGHT_CHANGE`: Shift strategy weights (e.g. dense vs BM25).
- `SUBQUESTION_DECOMPOSITION`: Decompose the question into subquestions.
- `CONTEXT_REBUILD`: Format and select different chunks.
- `GENERATION_REPAIR`: Instruct the generator to rewrite without new retrieval.

Format your response strictly as a JSON object matching this schema:
{
  "decision": "ACCEPT | REPAIR | KILL",
  "score": 0.85,
  "issues": ["Issue 1 description"],
  "recommended_repair": "QUERY_REWRITE | RETRIEVAL_EXPANSION | RETRIEVAL_WEIGHT_CHANGE | SUBQUESTION_DECOMPOSITION | CONTEXT_REBUILD | GENERATION_REPAIR"
}

Do not write markdown formatting or extra text. Output only valid JSON.
"""


class CriticAgent:
    """Agent that critiques answers and advises on graph loop routing."""

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm = llm_provider or get_llm_provider()

    def _heuristics_critique(self, verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rule-based decision scoring fallback when LLM is unavailable or in mock mode."""
        if not verifications:
            return {
                "decision": "KILL",
                "score": 0.0,
                "issues": ["No claims extracted or context was completely empty."],
                "recommended_repair": "QUERY_REWRITE"
            }

        total_claims = len(verifications)
        supported_claims = sum(1 for v in verifications if v["status"] == "SUPPORTED")
        contradicted_claims = sum(1 for v in verifications if v["status"] == "CONTRADICTED")
        unsupported_claims = sum(1 for v in verifications if v["status"] == "UNSUPPORTED")

        coverage = supported_claims / total_claims
        score = coverage

        # Hard criteria mapping
        if contradicted_claims > 0:
            decision = "KILL"
            issues = [f"Found {contradicted_claims} claims directly contradicted by evidence."]
            repair = "QUERY_REWRITE"
        elif coverage == 1.0:
            decision = "ACCEPT"
            issues = []
            repair = "GENERATION_REPAIR"
        elif coverage >= 0.5:
            decision = "REPAIR"
            issues = [f"Only {supported_claims}/{total_claims} claims are fully supported."]
            repair = "RETRIEVAL_EXPANSION"
        else:
            # Low support
            decision = "REPAIR"
            issues = [f"Low evidence coverage: {supported_claims}/{total_claims} claims supported."]
            # Recommend rewriting if it's very bad
            repair = "QUERY_REWRITE"

        return {
            "decision": decision,
            "score": round(score, 2),
            "issues": issues,
            "recommended_repair": repair
        }

    async def critique(self, answer: str, verifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Critique the answer text and verification results."""
        if not self.llm or not self.llm.is_available() or self.llm.provider_name == "mock":
            logger.info("Using local critic heuristics (LLM unavailable/mock).")
            return self._heuristics_critique(verifications)

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=CRITIC_SYSTEM_PROMPT),
                Message(
                    role=MessageRole.USER,
                    content=f"Answer: '{answer}'\n\nVerifications: {json.dumps(verifications)}",
                ),
            ]

            resp = await self.llm.chat_complete(messages, temperature=0.0)

            content = resp.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)

            data = json.loads(content)
            return {
                "decision": data.get("decision", "REPAIR"),
                "score": max(0.0, min(1.0, float(data.get("score", 0.5)))),
                "issues": data.get("issues", []),
                "recommended_repair": data.get("recommended_repair", "QUERY_REWRITE"),
            }
        except Exception as e:
            logger.warning("Critic Agent execution error: %s. Falling back to heuristics.", str(e))
            return self._heuristics_critique(verifications)
