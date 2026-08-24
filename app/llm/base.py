"""Abstract base class and data models for LLM provider abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Standard message roles for chat-style LLM APIs."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a chat completion request."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to OpenAI-compatible dict."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    """Structured response from an LLM provider."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_used(self) -> int:
        """Total tokens consumed (prompt + completion)."""
        return self.total_tokens or (self.prompt_tokens + self.completion_tokens)


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers.

    Concrete providers must implement `chat_complete` and `is_available`.
    All providers are stateless — callers supply full message history each time.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'openai', 'mock')."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier used for generation (e.g. 'gpt-4o')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and reachable.

        This must be a cheap synchronous check (e.g. API key present).
        It should NOT make a network call.
        """

    @abstractmethod
    async def chat_complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Send a chat completion request and return a structured response.

        Args:
            messages: Ordered list of messages (system, user, assistant turns).
            temperature: Sampling temperature — 0.0 for deterministic output.
            max_tokens: Maximum tokens in the completion.

        Returns:
            LLMResponse with generated content and token usage.

        Raises:
            LLMProviderError: On API errors, timeouts, or invalid responses.
        """
