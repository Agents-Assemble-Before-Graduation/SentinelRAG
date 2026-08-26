"""Claim Extractor Agent: extracts atomic factual claims from generated text."""

import json
import re
from typing import Any, Dict, List

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider, Message, MessageRole
from app.llm.factory import get_llm_provider

logger = get_logger(__name__)

CLAIM_EXTRACTOR_SYSTEM_PROMPT = """You are the SentinelRAG Claim Extractor. Your task is to analyze the generated answer and extract all atomic, distinct factual claims.

An atomic claim is a single statement that can be verified as true or false against source text. It should not contain conjunctions like "and", "or", or "but" that merge multiple facts.

Format your response strictly as a JSON object containing a list of claims, where each claim has a unique ID ("claim_1", "claim_2", etc.) and the claim text:
{
  "claims": [
    {"id": "claim_1", "text": "Factual assertion 1"},
    {"id": "claim_2", "text": "Factual assertion 2"}
  ]
}

Do not write markdown formatting or extra text. Output only valid JSON.
"""


class ClaimExtractorAgent:
    """Agent that decomposes answers into atomic claims for verification."""

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm = llm_provider or get_llm_provider()

    def _heuristics_extract(self, answer: str) -> List[Dict[str, Any]]:
        """Sentence-splitting fallback when LLM is unavailable or in mock mode."""
        if not answer or answer == "LLM unavailable":
            return []

        # Split into sentences using simple regex
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.?!])\s+", answer)
            if s.strip() and not s.startswith("[")
        ]
        
        claims = []
        for idx, sent in enumerate(sentences, start=1):
            # Clean up inline citation tags
            clean_sent = re.sub(r"\[Evidence \d+\]", "", sent).strip()
            if clean_sent:
                claims.append({"id": f"claim_{idx}", "text": clean_sent})
        return claims

    async def extract_claims(self, answer: str) -> List[Dict[str, Any]]:
        """Decompose the answer text into atomic verifiable claims."""
        if not answer or answer.strip() == "LLM unavailable":
            return []

        if not self.llm or not self.llm.is_available() or self.llm.provider_name == "mock":
            logger.info("Using local claim extraction heuristics (LLM unavailable/mock).")
            return self._heuristics_extract(answer)

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=CLAIM_EXTRACTOR_SYSTEM_PROMPT),
                Message(
                    role=MessageRole.USER,
                    content=f"Extract atomic claims from: '{answer}'",
                ),
            ]

            resp = await self.llm.chat_complete(messages, temperature=0.0)

            content = resp.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)

            data = json.loads(content)
            return data.get("claims", [])
        except Exception as e:
            logger.warning("ClaimExtractor Agent encountered error: %s. Falling back to sentence splitting.", str(e))
            return self._heuristics_extract(answer)
