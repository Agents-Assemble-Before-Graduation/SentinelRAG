"""RAG answer generator: constructs grounded LLM responses from retrieved evidence."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider, Message, MessageRole
from app.llm.factory import get_llm_provider
from app.rag.context.builder import BuiltContext, SourceCitation

logger = get_logger(__name__)

# Paths to prompt templates
_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "rag_system.txt"
_USER_PROMPT_PATH = _PROMPTS_DIR / "rag_user.txt"

# Fallback inline prompts in case file loading fails
_FALLBACK_SYSTEM = (
    "You are a research assistant. Answer only from the provided evidence. "
    "Cite sources using [Evidence N]. If evidence is insufficient, say so."
)
_FALLBACK_USER = (
    "Evidence:\n{evidence_blocks}\n\nQuestion: {question}\n\nAnswer:"
)

_NO_EVIDENCE_RESPONSE = (
    "The provided documents do not contain sufficient information to answer this question. "
    "Please ingest relevant documents and try again."
)


@dataclass
class GenerationResult:
    """Structured output from the RAG generator."""

    answer: str
    sources: list[SourceCitation]
    model_used: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_used: int = 0
    finish_reason: str = "stop"
    grounded: bool = True                 # False when returned without LLM call
    metadata: dict = field(default_factory=dict)


def _load_prompt(path: Path, fallback: str) -> str:
    """Load prompt template from disk with fallback."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Could not load prompt from %s; using fallback.", path)
        return fallback


class RAGGenerator:
    """Generates grounded answers from evidence context using a configured LLM provider.

    Design principles:
    - Evidence is injected inside XML-like ``<evidence>`` tags, making it clear to
      the model that this content is DATA, not instructions.
    - An empty context immediately short-circuits to a "no evidence" response
      without ever calling the LLM (prevents hallucination by design).
    - Temperature is fixed at 0.0 for deterministic, citation-traceable output.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm_provider or get_llm_provider()
        self._max_tokens = max_tokens
        self._temperature = temperature

        # Load prompt templates once at construction time
        self._system_prompt = _load_prompt(_SYSTEM_PROMPT_PATH, _FALLBACK_SYSTEM)
        self._user_template = _load_prompt(_USER_PROMPT_PATH, _FALLBACK_USER)

    async def generate(self, question: str, context: BuiltContext) -> GenerationResult:
        """Generate a grounded answer for the question from the assembled context.

        Args:
            question: The user's raw question.
            context: BuiltContext produced by ContextBuilder.

        Returns:
            GenerationResult with answer, source citations, and token usage.

        Raises:
            LLMProviderError: If the LLM provider is unavailable or returns an error.
        """
        # Short-circuit: no evidence → refuse to generate without hallucinating
        if not context.context_text or not context.sources:
            logger.info(
                "No evidence context for question '%s...' — returning no-evidence response.",
                question[:60],
            )
            return GenerationResult(
                answer=_NO_EVIDENCE_RESPONSE,
                sources=[],
                model_used=self._llm.model_name,
                grounded=False,
                metadata={"reason": "no_evidence"},
            )

        # Build messages
        user_content = self._user_template.format(
            evidence_blocks=context.context_text,
            question=question.strip(),
        )

        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=self._system_prompt),
            Message(role=MessageRole.USER, content=user_content),
        ]

        logger.debug(
            "Calling LLM '%s' for question '%s...' (%d evidence chars, %d sources)",
            self._llm.model_name,
            question[:60],
            context.total_chars,
            len(context.sources),
        )

        llm_response = await self._llm.chat_complete(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        logger.info(
            "LLM generation complete: model=%s, tokens=%d, finish=%s",
            llm_response.model,
            llm_response.tokens_used,
            llm_response.finish_reason,
        )

        return GenerationResult(
            answer=llm_response.content.strip(),
            sources=context.sources,
            model_used=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            tokens_used=llm_response.tokens_used,
            finish_reason=llm_response.finish_reason,
            grounded=True,
            metadata=llm_response.metadata,
        )
