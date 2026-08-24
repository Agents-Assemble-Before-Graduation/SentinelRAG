"""OpenAI-compatible LLM provider using httpx (no openai SDK dependency)."""

import json
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.llm.base import BaseLLMProvider, LLMResponse, Message

logger = get_logger(__name__)

# OpenAI-compatible endpoint (works for OpenAI, Groq, Ollama, etc.)
_OPENAI_CHAT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(BaseLLMProvider):
    """LLM provider that speaks the OpenAI chat completions REST API.

    Uses httpx directly so no `openai` package is required.
    Works transparently with any OpenAI-compatible endpoint
    (OpenAI, Groq, local Ollama, etc.) by setting LLM_BASE_URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self._api_key: str | None = api_key or settings.LLM_API_KEY or None
        self._model: str = model or settings.LLM_MODEL
        self._base_url: str = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        """Return True only when a non-empty API key is configured."""
        return bool(self._api_key and self._api_key.strip())

    async def chat_complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Call the OpenAI chat completions endpoint via httpx.

        Raises:
            LLMProviderError: If API key is missing, network fails, or API returns an error.
        """
        if not self.is_available():
            raise LLMProviderError(
                "LLM provider is not configured. "
                "Set LLM_API_KEY in your .env file to enable generation.",
                details={"provider": self.provider_name, "model": self._model},
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        endpoint = f"{self._base_url}/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            logger.error("LLM request timed out after %.1fs: %s", self._timeout, str(exc))
            raise LLMProviderError(
                f"LLM request timed out after {self._timeout}s.",
                details={"model": self._model},
            ) from exc
        except httpx.RequestError as exc:
            logger.error("LLM network error: %s", str(exc))
            raise LLMProviderError(
                f"Network error reaching LLM provider: {str(exc)}",
                details={"model": self._model},
            ) from exc

        if resp.status_code != 200:
            body = resp.text[:500]
            logger.error("LLM API error %d: %s", resp.status_code, body)
            raise LLMProviderError(
                f"LLM API returned status {resp.status_code}.",
                details={"status_code": resp.status_code, "body": body, "model": self._model},
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                "LLM API returned malformed JSON response.",
                details={"model": self._model},
            ) from exc

        try:
            choice = data["choices"][0]
            content: str = choice["message"]["content"]
            finish_reason: str = choice.get("finish_reason", "stop")
            usage: dict[str, int] = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=data.get("model", self._model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=finish_reason,
                metadata={"id": data.get("id", "")},
            )
        except (KeyError, IndexError) as exc:
            raise LLMProviderError(
                "LLM API response has unexpected structure.",
                details={"model": self._model, "keys": list(data.keys())},
            ) from exc
