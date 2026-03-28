import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.adapters.outbound.gemini_fallback_adapter import GeminiAdapter
from src.domain.entities.llm_message import LLMRequest
from src.domain.exceptions.llm_exceptions import CreditExhaustionError


@patch("google.genai.Client")
def test_gemini_fallback_adapter_credit_exhaustion(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate credit exhaustion error
    mock_client.models.generate_content.side_effect = Exception(
        "User location is not supported or credit balance is low"
    )

    adapter = GeminiAdapter(model_name="gemini-3-flash-preview", api_key="fake-key")
    request = LLMRequest(prompt="Hello")

    with pytest.raises(CreditExhaustionError):
        adapter.generate_response(request)


@patch("litellm.completion")
def test_litellm_fallback_adapter_credit_exhaustion(mock_completion):
    # Simulate credit exhaustion error from litellm (often 402 Payment Required)
    mock_completion.side_effect = Exception("LiteLLM Error: 402 Payment Required")

    from src.infrastructure.adapters.outbound.litellm_fallback_adapter import (
        LiteLLMAdapter,
    )

    adapter = LiteLLMAdapter(model_name="openai/gpt-5-nano")
    request = LLMRequest(prompt="Hello")

    with pytest.raises(CreditExhaustionError):
        adapter.generate_response(request)
