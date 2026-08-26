"""Evidence Verifier Agent: verifies extracted claims against the retrieved context."""

import json
import re
from typing import Any, Dict, List

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider, Message, MessageRole
from app.llm.factory import get_llm_provider

logger = get_logger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are the SentinelRAG Evidence Verifier. Your task is to evaluate the support status of a set of atomic claims against the provided context.

Context:
{context}

For each claim, you must determine its verification status:
- `SUPPORTED`: The claim is directly and fully stated in the context.
- `PARTIALLY_SUPPORTED`: Part of the claim is supported, or it is implicitly supported, but lacks direct full alignment.
- `UNSUPPORTED`: The context does not mention this claim or provide evidence for it.
- `CONTRADICTED`: The context directly states the opposite or contradicts the claim.
- `UNCERTAIN`: The evidence is ambiguous or insufficient to verify.

Respond strictly with a JSON object in this format:
{
  "verifications": [
    {
      "id": "claim_1",
      "status": "SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED | UNCERTAIN",
      "reason": "Brief reason explaining the status",
      "citations": ["filename.txt"]
    }
  ]
}

Do not write markdown formatting or extra text. Output only valid JSON.
"""


class EvidenceVerifierAgent:
    """Agent that verifies claims against evidence and calculates coverage."""

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm = llm_provider or get_llm_provider()

    def _heuristics_verify(self, claims: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
        """Keyword overlap matching fallback when LLM is unavailable or in mock mode."""
        if not claims:
            return []

        verifications = []
        ctx_lower = context.lower()

        # Extract source filenames from context blocks to assign citations
        filenames = list(set(re.findall(r"Source: [^\n|]+\| File: ([^\s,]+)", context)))
        if not filenames:
            filenames = list(set(re.findall(r"File: `([^`]+)`", context)))
        if not filenames:
            filenames = ["source_document"]

        for claim in claims:
            claim_text = claim["text"]
            claim_lower = claim_text.lower()

            # Clean claim to get keywords
            words = set(re.findall(r"\w{4,}", claim_lower))
            if not words:
                verifications.append({
                    "id": claim["id"],
                    "status": "UNSUPPORTED",
                    "reason": "Claim too short to verify.",
                    "citations": []
                })
                continue

            # Check if contradictions exist
            contradicted = False
            if "not" in claim_lower or "never" in claim_lower:
                # If claim asserts negation but context has positive overlap
                contradicted = False  # heuristic refinement below
            
            # Simple keyword match ratio
            matches = [w for w in words if w in ctx_lower]
            overlap_ratio = len(matches) / len(words)

            if overlap_ratio >= 0.7:
                status = "SUPPORTED"
                reason = f"Keyword overlap of {int(overlap_ratio * 100)}% found in context."
                citations = filenames
            elif overlap_ratio >= 0.3:
                status = "PARTIALLY_SUPPORTED"
                reason = f"Partial overlap of {int(overlap_ratio * 100)}% found in context."
                citations = filenames
            else:
                status = "UNSUPPORTED"
                reason = "No matching key assertions found in context."
                citations = []

            verifications.append({
                "id": claim["id"],
                "status": status,
                "reason": reason,
                "citations": citations
            })

        return verifications

    async def verify_claims(self, claims: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
        """Verify each claim against the context and return status mappings."""
        if not claims:
            return []

        if not context or context.strip() == "":
            return [
                {
                    "id": c["id"],
                    "status": "UNSUPPORTED",
                    "reason": "Context is empty.",
                    "citations": []
                }
                for c in claims
            ]

        if not self.llm or not self.llm.is_available() or self.llm.provider_name == "mock":
            logger.info("Using local evidence verification heuristics (LLM unavailable/mock).")
            return self._heuristics_verify(claims, context)

        try:
            prompt = VERIFIER_SYSTEM_PROMPT.format(context=context)
            messages = [
                Message(role=MessageRole.SYSTEM, content=prompt),
                Message(
                    role=MessageRole.USER,
                    content=f"Verify these claims: {json.dumps(claims)}",
                ),
            ]

            resp = await self.llm.chat_complete(messages, temperature=0.0)

            content = resp.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)

            data = json.loads(content)
            return data.get("verifications", [])
        except Exception as e:
            logger.warning("EvidenceVerifier Agent encountered error: %s. Falling back to heuristics.", str(e))
            return self._heuristics_verify(claims, context)
