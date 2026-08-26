"""Lesson extractor: derives generalised orchestration lessons from notable query episodes."""

from typing import Any

from app.core.logging import get_logger
from app.llm.factory import get_llm_provider
from app.llm.base import BaseLLMProvider, Message, MessageRole

logger = get_logger(__name__)

# Minimum episode interestingness to merit lesson extraction
# An episode is "notable" if any of these conditions hold:
#   - was_killed is True
#   - repair_attempts >= 1
#   - confidence < 0.6
NOTABLE_CONFIDENCE_THRESHOLD = 0.6


def _is_notable_episode(episode_data: dict[str, Any]) -> bool:
    """Return True if the episode is interesting enough to extract lessons from."""
    return (
        episode_data.get("was_killed", False)
        or (episode_data.get("repair_attempts", 0) >= 1)
        or (episode_data.get("confidence", 1.0) < NOTABLE_CONFIDENCE_THRESHOLD)
    )


EXTRACTOR_SYSTEM_PROMPT = """You are a Retrieval Strategy Analyst for SentinelRAG.

Your job: given a query execution summary, extract ONE generalised lesson about retrieval orchestration.

Rules:
- The lesson must be actionable and general, not specific to the exact query text.
- Describe what retrieval behaviour should change for similar future queries.
- Keep the lesson under 2 sentences.
- Assign a category from: retrieval_strategy, query_rewriting, evidence_gap, contradiction, verification_depth.
- Assign a confidence score (0.0-1.0) for how reliable this lesson is.

Reply ONLY with valid JSON:
{
  "lesson": "<generalised lesson>",
  "category": "<category>",
  "confidence": <0.0-1.0>
}"""


class LessonExtractor:
    """Extracts structured, generalised lessons from notable query episodes.

    Uses the LLM when available, falling back to deterministic rule-based
    template generation for offline/test mode.
    """

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm = llm_provider or get_llm_provider()

    def _rule_based_lessons(self, episode_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate deterministic template lessons when LLM is unavailable."""
        lessons: list[dict[str, Any]] = []
        query_type = episode_data.get("query_type", "factual")
        strategy = episode_data.get("retrieval_strategy", "dense")
        was_killed = episode_data.get("was_killed", False)
        repair_attempts = episode_data.get("repair_attempts", 0)
        issues = episode_data.get("issues", [])
        confidence = episode_data.get("confidence", 1.0)

        if was_killed:
            lessons.append({
                "lesson": (
                    f"Queries of type '{query_type}' using strategy '{strategy}' may have insufficient "
                    "evidence coverage in the corpus; consider increasing retrieval depth or flagging as "
                    "evidence-gap queries."
                ),
                "category": "evidence_gap",
                "confidence": 0.75,
            })

        if repair_attempts >= 1 and not was_killed:
            lessons.append({
                "lesson": (
                    f"For '{query_type}' queries, an initial repair loop was needed with strategy '{strategy}'. "
                    "Switching to hybrid retrieval earlier may reduce repair cycles."
                ),
                "category": "retrieval_strategy",
                "confidence": 0.65,
            })

        if repair_attempts >= 1 and query_type in ("numerical", "factual"):
            lessons.append({
                "lesson": (
                    "Exact-match numerical and factual queries benefit from BM25 lexical retrieval "
                    "as primary strategy before falling back to semantic search."
                ),
                "category": "retrieval_strategy",
                "confidence": 0.70,
            })

        if confidence < 0.4:
            lessons.append({
                "lesson": (
                    f"Low confidence ({confidence:.2f}) encountered on '{query_type}' queries suggests "
                    "that retrieval depth should be increased (top-k expansion) for this query class."
                ),
                "category": "verification_depth",
                "confidence": 0.60,
            })

        if any("rewrite" in str(i).lower() for i in issues):
            lessons.append({
                "lesson": (
                    "Query rewriting was flagged as a remediation action. "
                    "Original query reformulation should be attempted proactively for ambiguous queries."
                ),
                "category": "query_rewriting",
                "confidence": 0.65,
            })

        return lessons

    async def extract(self, episode_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract lessons from a completed episode.

        Only processes notable episodes (killed, repaired, or low-confidence).
        Uses LLM for richer extraction when available.

        Args:
            episode_data: Dict with episode fields (question, query_type,
                retrieval_strategy, confidence, repair_attempts, was_killed, issues).

        Returns:
            List of lesson dicts with keys: lesson, category, confidence.
            Returns empty list for unremarkable episodes.
        """
        if not _is_notable_episode(episode_data):
            logger.debug("[LessonExtractor] Episode not notable; skipping extraction.")
            return []

        # Prefer LLM extraction for richer output
        if (
            self.llm
            and self.llm.is_available()
            and self.llm.provider_name != "mock"
        ):
            return await self._llm_extract(episode_data)

        return self._rule_based_lessons(episode_data)

    async def _llm_extract(self, episode_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Use the LLM to generate a structured lesson from the episode summary."""
        import json
        import re

        summary = (
            f"Question type: {episode_data.get('query_type', 'unknown')}\n"
            f"Retrieval strategy: {episode_data.get('retrieval_strategy', 'unknown')}\n"
            f"Repair attempts: {episode_data.get('repair_attempts', 0)}\n"
            f"Was killed: {episode_data.get('was_killed', False)}\n"
            f"Confidence: {episode_data.get('confidence', 0.0):.2f}\n"
            f"Issues: {', '.join(episode_data.get('issues', [])) or 'none'}\n"
        )

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=EXTRACTOR_SYSTEM_PROMPT),
                Message(
                    role=MessageRole.USER,
                    content=f"Extract a lesson from this episode:\n{summary}",
                ),
            ]
            resp = await self.llm.chat_complete(messages, temperature=0.0)
            content = resp.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
            data = json.loads(content)
            confidence = float(data.get("confidence", 0.5))
            return [
                {
                    "lesson": data.get("lesson", ""),
                    "category": data.get("category", "retrieval_strategy"),
                    "confidence": confidence,
                }
            ]
        except Exception as exc:
            logger.warning("[LessonExtractor] LLM extraction failed: %s — using rule-based fallback", str(exc))
            return self._rule_based_lessons(episode_data)
