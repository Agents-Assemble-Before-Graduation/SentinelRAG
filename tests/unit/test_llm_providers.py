"""Unit tests for LLM providers and factory."""

import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.core.exceptions import LLMProviderError
from app.llm.base import Message, MessageRole
from app.llm.mock_provider import MockLLMProvider
from app.llm.openai_provider import OpenAIProvider
from app.llm.factory import get_llm_provider, reset_llm_provider_cache


def test_message_to_dict():
    """Verify Message serialize logic."""
    msg = Message(role=MessageRole.USER, content="Hello World")
    assert msg.to_dict() == {"role": "user", "content": "Hello World"}


@pytest.mark.asyncio
async def test_mock_llm_provider():
    """Verify MockLLMProvider returns correct responses and tracks call history."""
    provider = MockLLMProvider(response_text="Custom Mock Answer", model="mock-test-1.0")
    assert provider.provider_name == "mock"
    assert provider.model_name == "mock-test-1.0"
    assert provider.is_available() is True
    
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helper."),
        Message(role=MessageRole.USER, content="Ask something.")
    ]
    response = await provider.chat_complete(messages=messages)
    assert response.content == "Custom Mock Answer"
    assert response.model == "mock-test-1.0"
    assert response.tokens_used > 0
    assert provider.call_count == 1
    assert provider.last_messages == messages


@pytest.mark.asyncio
async def test_openai_provider_unavailable():
    """Verify OpenAIProvider is unavailable if API key is missing."""
    provider = OpenAIProvider(api_key="")
    assert provider.is_available() is False

    with pytest.raises(LLMProviderError, match="LLM provider is not configured"):
        await provider.chat_complete([Message(role=MessageRole.USER, content="hello")])


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_openai_provider_success(mock_post):
    """Verify OpenAIProvider parses successful API responses correctly."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Grounded answer from docs."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="fake-key", model="gpt-4o", base_url="https://api.openai.com/v1")
    assert provider.is_available() is True

    messages = [Message(role=MessageRole.USER, content="hello")]
    response = await provider.chat_complete(messages)

    assert response.content == "Grounded answer from docs."
    assert response.model == "gpt-4o"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 15
    assert response.total_tokens == 25
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_openai_provider_api_error(mock_post):
    """Verify OpenAIProvider raises LLMProviderError on API errors."""
    mock_response = AsyncMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized API key"
    mock_post.return_value = mock_response

    provider = OpenAIProvider(api_key="invalid-key")
    with pytest.raises(LLMProviderError, match="LLM API returned status 401"):
        await provider.chat_complete([Message(role=MessageRole.USER, content="hello")])


def test_llm_factory():
    """Verify LLM factory creates correct provider instance based on settings."""
    reset_llm_provider_cache()
    
    mock_provider = get_llm_provider("mock")
    assert isinstance(mock_provider, MockLLMProvider)

    openai_provider = get_llm_provider("openai")
    assert isinstance(openai_provider, OpenAIProvider)
