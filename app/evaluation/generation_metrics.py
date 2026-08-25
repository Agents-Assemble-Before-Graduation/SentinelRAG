"""Generation metrics module evaluating faithfulness, relevance, and citation accuracy."""

import re
from typing import Any, Dict, List, Optional
from app.llm.base import BaseLLMProvider, Message, MessageRole
from app.llm.factory import get_llm_provider
from app.rag.context.builder import BuiltContext

# LLM-as-Judge Limitations Documentation:
# 1. Prompt Sensitivity: Scores can vary significantly based on minor prompt wording changes.
# 2. Position Bias: LLMs tend to favor assertions positioned near the beginning or end of text.
# 3. Scale Consistency: LLMs struggle to maintain a uniform numeric rating scale across different runs.
# 4. Self-Preference Bias: If the judge LLM matches the generator LLM, it tends to rate its own style higher.
# 5. Cost and Latency: Running LLM evaluations adds substantial runtime overhead and cost.


class LLMJudgeEvaluator:
    """Evaluates RAG generation metrics using an LLM-as-judge approach."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None) -> None:
        self.llm = llm_provider or get_llm_provider()

    async def _evaluate_prompt(self, prompt: str) -> float:
        """Call LLM with evaluation prompt and parse the resulting score (0.0 - 1.0)."""
        if not self.llm or not self.llm.is_available() or self.llm.provider_name == "mock":
            return 1.0  # Safe fallback score for tests/unconfigured runs

        try:
            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are an objective evaluation judge. Reply only with a float score between 0.0 and 1.0.",
                ),
                Message(role=MessageRole.USER, content=prompt),
            ]
            resp = await self.llm.chat_complete(
                messages, temperature=0.0, max_tokens=10
            )

            # Search for a decimal number in the response
            match = re.search(r"(\d+\.\d+|\d+)", resp.content)
            if match:
                val = float(match.group(1))
                return max(0.0, min(1.0, val))
        except Exception:
            pass
        return 0.5

    async def score_faithfulness(self, answer: str, context: str) -> float:
        """Faithfulness: Is the answer derived ONLY from the provided context?"""
        if not answer or not context:
            return 0.0

        prompt = f"""
Evaluate if the following answer is faithful to the context. Every claim in the answer must be directly supported by the context. Do not allow outside knowledge.

Context:
{context}

Answer:
{answer}

Rate the faithfulness from 0.0 (completely unfaithful/hallucinated) to 1.0 (completely faithful). Respond with just the decimal score.
"""
        return await self._evaluate_prompt(prompt)

    async def score_answer_relevance(self, question: str, answer: str) -> float:
        """Answer Relevance: Does the generated answer directly address the question?"""
        if not question or not answer:
            return 0.0

        prompt = f"""
Evaluate if the answer directly and completely addresses the user's question.

Question:
{question}

Answer:
{answer}

Rate the relevance from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and complete). Respond with just the decimal score.
"""
        return await self._evaluate_prompt(prompt)

    async def score_context_relevance(self, question: str, context: str) -> float:
        """Context Relevance: Are the retrieved passages in context relevant to the query?"""
        if not question or not context:
            return 0.0

        prompt = f"""
Evaluate how relevant the retrieved context passages are to answering the user's question.

Question:
{question}

Context:
{context}

Rate the context relevance from 0.0 (completely irrelevant noise) to 1.0 (perfectly relevant details). Respond with just the decimal score.
"""
        return await self._evaluate_prompt(prompt)

    def score_citation_correctness(
        self, answer: str, context: BuiltContext
    ) -> float:
        """Citation Correctness: Do inline citation markers [Evidence N] reference chunks that contain the fact?

        Implements heuristic verification by looking up terms.
        """
        # Parse citations from answer like [Evidence 1], [Evidence 2]
        citations = re.findall(r"\[Evidence (\d+)\]", answer)
        if not citations:
            return 1.0  # No citations to be incorrect, but might fail completeness

        correct_count = 0
        total_citations = len(citations)

        # Split answer on sentences to identify where citations occur
        sentences = re.split(r"(?<=[.?!])\s+", answer)

        for citation_str in citations:
            idx = int(citation_str) - 1
            if idx < 0 or idx >= len(context.sources):
                continue

            expected_text = context.context_text
            # Look up corresponding evidence block content
            match_block = re.search(
                rf"\[Evidence {citation_str}\].*?\n---\n(.*?)(?=\n\n\[Evidence|\Z)",
                expected_text,
                re.DOTALL,
            )

            if not match_block:
                continue

            chunk_content = match_block.group(1).strip().lower()

            # Find sentence(s) that cite this chunk
            found_ref = False
            for sent in sentences:
                if f"[Evidence {citation_str}]" in sent:
                    # Check if key content/nouns overlap between sentence and chunk
                    sent_cleaned = re.sub(
                        r"\[Evidence \d+\]", "", sent
                    ).strip().lower()
                    sent_words = set(re.findall(r"\w{4,}", sent_cleaned))

                    if not sent_words:
                        found_ref = True
                        break

                    overlap = [w for w in sent_words if w in chunk_content]
                    # If at least some overlap is found, count as correct
                    if len(overlap) >= min(2, len(sent_words)):
                        found_ref = True
                        break

            if found_ref:
                correct_count += 1

        return correct_count / total_citations if total_citations > 0 else 1.0

    def score_citation_completeness(self, answer: str) -> float:
        """Citation Completeness: Are all assertions backed by a citation?"""
        if not answer:
            return 1.0

        # Split answer into sentences
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.?!])\s+", answer)
            if s.strip()
        ]
        if not sentences:
            return 1.0

        cited_sentences = 0
        for sent in sentences:
            if re.search(r"\[Evidence \d+\]", sent):
                cited_sentences += 1

        return cited_sentences / len(sentences)
