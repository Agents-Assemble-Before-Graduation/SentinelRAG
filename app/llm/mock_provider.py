"""Mock LLM provider for test isolation — never calls any real API."""

from app.llm.base import BaseLLMProvider, LLMResponse, Message


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock LLM provider for unit and integration testing.

    Returns a configurable canned response without any network calls.
    Tracks call history for assertion in tests.
    """

    def __init__(
        self,
        response_text: str = "This is a mock answer based on the provided evidence.",
        model: str = "mock-model-1.0",
        available: bool = True,
    ) -> None:
        self._response_text = response_text
        self._model = model
        self._available = available
        self.call_count: int = 0
        self.last_messages: list[Message] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return self._available

    async def chat_complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return a canned response and record call metadata."""
        self.call_count += 1
        self.last_messages = list(messages)

        # Estimate fake token counts
        total_input_chars = sum(len(m.content) for m in messages)
        prompt_tokens = max(1, total_input_chars // 4)
        completion_tokens = max(1, len(self._response_text) // 4)

        return LLMResponse(
            content=self._response_text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
            metadata={"mock": True},
        )
