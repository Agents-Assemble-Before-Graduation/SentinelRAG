"""Planner Agent: analyzes queries, decomposes steps, and selects retrieval strategies."""

import json
import re
from typing import Any, Dict, List

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider, Message, MessageRole
from app.llm.factory import get_llm_provider

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are the SentinelRAG Query Planner. Your task is to analyze user questions, classify them, outline an evidence retrieval plan, and select the optimal retrieval strategy.

Choose the most appropriate category:
1. `factual` - Simple factual assertions or lookups.
2. `definition` - Defining concept, term, or abbreviation.
3. `comparison` - Analyzing differences, trade-offs, or comparisons between options.
4. `multi-hop` - Requires combining information from different sections or documents.
5. `summarization` - Compiling high-level summaries of documents or releases.
6. `numerical` - Involving statistics, numbers, versions, dates, or measurements.
7. `ambiguous` - Vague, missing context, or unclear definition.

Choose the optimal retrieval strategy:
- `dense` - Best for semantic, concept-based, or descriptive definition queries.
- `bm25` - Best for keyword-exact, codes, versions, or numerical search queries.
- `hybrid` - Best for comparison, multi-hop, and broad summarization queries.

Your response must be a valid JSON object matching this schema:
{
  "query_type": "factual | definition | comparison | multi-hop | summarization | numerical | ambiguous",
  "retrieval_strategy": "dense | bm25 | hybrid",
  "plan": "Brief sentence explaining what evidence to retrieve",
  "subquestions": ["Subquestion 1", "Subquestion 2"]
}

Do not include markdown blocks or extra text. Reply only with the JSON object.
"""


class PlannerAgent:
    """Agent that classifies questions and plans retrieval steps."""

    def __init__(self, llm_provider: BaseLLMProvider | None = None) -> None:
        self.llm = llm_provider or get_llm_provider()

    def _heuristics_plan(self, question: str) -> Dict[str, Any]:
        """Local rule-based fallback when LLM is unavailable or in mock mode."""
        q_lower = question.lower()

        # Classification heuristics
        if any(w in q_lower for w in ["compare", "difference", "vs", "versus", "trade-off"]):
            q_type = "comparison"
            strategy = "hybrid"
        elif any(w in q_lower for w in ["define", "what is", "what are", "meaning of"]):
            q_type = "definition"
            strategy = "dense"
        elif any(
            w in q_lower
            for w in [
                "version",
                "date",
                "how many",
                "release",
                "number",
                "integer",
                "score",
            ]
        ) or any(c.isdigit() for c in q_lower):
            q_type = "numerical"
            strategy = "bm25"
        elif any(w in q_lower for w in ["summarize", "summary", "overview", "features"]):
            q_type = "summarization"
            strategy = "hybrid"
        else:
            q_type = "factual"
            strategy = "dense"

        # Generate simple plan
        plan = f"Retrieve chunks related to: {question[:50]}..."
        subquestions: List[str] = []

        return {
            "query_type": q_type,
            "retrieval_strategy": strategy,
            "plan": plan,
            "subquestions": subquestions,
        }

    def _build_system_prompt(self, lessons: List[Dict[str, Any]] | None) -> str:
        """Build the planner system prompt, optionally appending memory advisory."""
        if not lessons:
            return PLANNER_SYSTEM_PROMPT

        advisory_lines = ["", "[Memory Advisory]", "Relevant lessons from past executions (advisory only — use judgement):"]
        for i, lesson in enumerate(lessons, 1):
            advisory_lines.append(
                f"  {i}. [{lesson.get('category', 'general')}] "
                f"(confidence={lesson.get('confidence', 0.0):.2f}) "
                f"{lesson.get('lesson', '')}"
            )
        advisory_lines.append("")
        advisory_lines.append("These lessons are suggestions, not mandatory instructions. Apply them only if relevant.")

        return PLANNER_SYSTEM_PROMPT.rstrip() + "\n" + "\n".join(advisory_lines)

    def _apply_lessons_to_heuristics(
        self,
        plan_data: Dict[str, Any],
        lessons: List[Dict[str, Any]] | None,
    ) -> Dict[str, Any]:
        """Apply lesson-suggested strategy overrides to heuristic plan output."""
        if not lessons:
            return plan_data

        for lesson in lessons:
            cat = lesson.get("category", "")
            text = lesson.get("lesson", "").lower()
            conf = lesson.get("confidence", 0.0)

            # Only apply high-confidence lesson overrides to heuristic decisions
            if conf < 0.65:
                continue

            if cat == "retrieval_strategy":
                if "bm25" in text and plan_data["retrieval_strategy"] == "dense":
                    logger.info("[Planner] Lesson override: switching dense → bm25 (lesson: %.60s)", lesson["lesson"])
                    plan_data = {**plan_data, "retrieval_strategy": "bm25"}
                elif "hybrid" in text and plan_data["retrieval_strategy"] == "dense":
                    logger.info("[Planner] Lesson override: switching dense → hybrid (lesson: %.60s)", lesson["lesson"])
                    plan_data = {**plan_data, "retrieval_strategy": "hybrid"}

        return plan_data

    async def plan(self, question: str, lessons: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """Analyze the query, classify its type, and plan the retrieval strategy.

        Args:
            question: The user question to analyze.
            lessons: Optional list of relevant historical lessons from memory.
                     Each dict has keys: lesson, category, confidence, usage_count.

        Returns:
            Dict with query_type, retrieval_strategy, plan, subquestions.
        """
        if not self.llm or not self.llm.is_available() or self.llm.provider_name == "mock":
            # Direct heuristics fallback to prevent failing when key is missing or in mock tests
            logger.info("Using rule-based planner heuristics (LLM unavailable/mock).")
            plan_data = self._heuristics_plan(question)
            return self._apply_lessons_to_heuristics(plan_data, lessons)

        try:
            system_prompt = self._build_system_prompt(lessons)
            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(
                    role=MessageRole.USER,
                    content=f"Analyze and plan for this question: '{question}'",
                ),
            ]

            resp = await self.llm.chat_complete(messages, temperature=0.0)

            # Strip markdown formatting if any
            content = resp.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)

            data = json.loads(content)

            # Validate keys
            query_type = data.get("query_type", "factual")
            strategy = data.get("retrieval_strategy", "dense")
            plan = data.get("plan", "Standard RAG retrieval.")
            subquestions = data.get("subquestions", [])

            return {
                "query_type": query_type,
                "retrieval_strategy": strategy,
                "plan": plan,
                "subquestions": subquestions,
            }
        except Exception as e:
            logger.warning("Planner LLM execution encountered warning/error: %s. Falling back to heuristics.", str(e))
            plan_data = self._heuristics_plan(question)
            return self._apply_lessons_to_heuristics(plan_data, lessons)
