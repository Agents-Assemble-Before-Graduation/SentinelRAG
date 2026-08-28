"""Prompt injection detection and sanitization for SentinelRAG.

Treats all document content as UNTRUSTED DATA. Scans for known direct and
indirect prompt injection patterns before document content is used as RAG
context or stored in the vector store.

This module does NOT modify AI behaviour — it operates purely at the text
level before content reaches any LLM prompt.
"""

import re
from typing import NamedTuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Injection pattern registry
# ---------------------------------------------------------------------------
# Each pattern targets a known injection technique. Patterns are case-insensitive
# and use word boundaries where possible to avoid false positives on normal text.

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Role override / persona hijack
    (r"\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?\b", "role_override"),
    (r"\bdisregard\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?\b", "role_override"),
    (r"\bforget\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?\b", "role_override"),
    (r"\byou\s+are\s+now\b", "persona_hijack"),
    (r"\bact\s+as\s+(a|an|the)\s+\w+", "persona_hijack"),
    (r"\bpretend\s+(to\s+be|you\s+are)\b", "persona_hijack"),
    (r"\byour\s+new\s+instructions?\b", "instruction_override"),
    (r"\bnew\s+system\s+prompt\b", "system_override"),
    (r"\boverride\s+(the\s+)?(system\s+)?(prompt|instructions?)\b", "system_override"),
    # Secret exfiltration
    (r"\breveal\s+(the\s+)?(system\s+)?(prompt|instructions?|api\s+key|secret)\b", "exfiltration"),
    (r"\bprint\s+(the\s+)?(system\s+)?(prompt|instructions?)\b", "exfiltration"),
    (r"\brepeat\s+(the\s+)?(system\s+)?(prompt|instructions?)\b", "exfiltration"),
    (r"\bwhat\s+(are|is)\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b", "exfiltration"),
    # Jailbreak attempts
    (r"\bdo\s+anything\s+now\b", "jailbreak"),
    (r"\bdan\s+mode\b", "jailbreak"),
    (r"\bjailbreak\b", "jailbreak"),
    (r"\bno\s+restrictions?\b", "jailbreak"),
    (r"\bunrestricted\s+mode\b", "jailbreak"),
    # Context termination tricks
    (r"---+\s*(system|instruction|prompt)\b", "delimiter_injection"),
    (r"```\s*(system|instruction|prompt)\b", "delimiter_injection"),
    (r"\[system\]", "delimiter_injection"),
    (r"<\s*(system|instruction)\s*>", "delimiter_injection"),
    # Data extraction via role reset
    (r"\btranslate\s+everything\s+above\b", "context_extraction"),
    (r"\bsummarise\s+(all\s+)?(previous|prior)\s+(context|messages?|conversation)\b", "context_extraction"),
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE | re.MULTILINE), category)
    for pat, category in _INJECTION_PATTERNS
]


class InjectionMatch(NamedTuple):
    """Represents a single detected injection pattern."""
    pattern_text: str
    category: str
    span: tuple[int, int]


class PromptInjectionSanitizer:
    """Scans and sanitizes document text for prompt injection attempts.

    Usage modes:
    - ``scan_for_injection(text)`` — returns list of detected matches (non-destructive).
    - ``sanitize(text)`` — removes detected patterns and logs warnings.
    - ``raise_if_injection(text)`` — raises SecurityViolationError on any detection.

    All operations are purely text-level — no LLM calls are made.
    """

    def scan_for_injection(self, text: str) -> list[InjectionMatch]:
        """Scan text for prompt injection patterns.

        Args:
            text: Document content or user input to scan.

        Returns:
            List of InjectionMatch objects for each detection.
            Empty list means clean.
        """
        if not text:
            return []

        matches: list[InjectionMatch] = []
        for compiled, category in _COMPILED_PATTERNS:
            for m in compiled.finditer(text):
                matches.append(InjectionMatch(
                    pattern_text=m.group(0),
                    category=category,
                    span=(m.start(), m.end()),
                ))

        return matches

    def sanitize(self, text: str) -> str:
        """Remove detected injection phrases from text, logging each removal.

        This is the SOFT mode — content is still usable after sanitization.
        Use for user queries where you want to clean rather than reject.

        Args:
            text: Raw text to sanitize.

        Returns:
            Text with injection phrases replaced by ``[SANITIZED]``.
        """
        if not text:
            return text

        result = text
        for compiled, category in _COMPILED_PATTERNS:
            new_result = compiled.sub("[SANITIZED]", result)
            if new_result != result:
                logger.warning(
                    "[PromptInjectionSanitizer] Sanitized injection pattern "
                    "(category=%s) from input.", category
                )
                result = new_result

        return result

    def raise_if_injection(self, text: str, source: str = "document") -> None:
        """Raise SecurityViolationError if any injection pattern is detected.

        This is the HARD mode — use for uploaded document content where
        we want to reject the document rather than silently strip content.

        Args:
            text: Document content to validate.
            source: Descriptive label for error messages (e.g. 'document', 'filename').

        Raises:
            SecurityViolationError: If any injection pattern is found.
        """
        from app.core.exceptions import SecurityViolationError

        matches = self.scan_for_injection(text)
        if matches:
            categories = list({m.category for m in matches})
            logger.warning(
                "[PromptInjectionSanitizer] Injection attempt detected in %s. "
                "Categories: %s. Match count: %d.",
                source, categories, len(matches),
            )
            raise SecurityViolationError(
                f"Prompt injection attempt detected in {source}. "
                f"Patterns found: {categories}. "
                "Document rejected for safety.",
                details={"categories": categories, "match_count": len(matches)},
            )
