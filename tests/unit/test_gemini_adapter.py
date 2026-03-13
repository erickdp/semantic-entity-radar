import pytest
from tenacity import wait_fixed
from unittest.mock import MagicMock, patch
from src.infrastructure.adapters.outbound.gemini_fallback_adapter import GeminiAdapter
from src.domain.entities.llm_message import LLMRequest
from src.domain.exceptions.llm_exceptions import RateLimitError


@patch("google.genai.Client")
def test_gemini_fallback_adapter_rate_limit_retry(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Simulate rate limit error twice, then success
    mock_response = MagicMock()
    mock_response.text = "Success"

    mock_client.models.generate_content.side_effect = [
        Exception("429 Rate limit reached"),
        Exception("429 Rate limit reached"),
        mock_response,
    ]

    adapter = GeminiAdapter(model_name="gemini-3-flash-preview", api_key="fake-key")
    request = LLMRequest(prompt="Hello")

    # We expect tenacity to retry and eventually succeed
    response = adapter.generate_response(request)

    assert response.content == "Success"
    assert mock_client.models.generate_content.call_count == 3


@patch("google.genai.Client")
def test_gemini_fallback_adapter_exhausts_retries(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_client.models.generate_content.side_effect = Exception(
        "429 Rate limit reached"
    )

    adapter = GeminiAdapter(model_name="gemini-3-flash-preview", api_key="fake-key")
    # Reduce wait for faster test
    with patch(
        "src.infrastructure.adapters.outbound.gemini_fallback_adapter.RETRY_WAIT_SEQUENCE",
        [wait_fixed(0.1), wait_fixed(0.1), wait_fixed(0.1)],
    ):
        request = LLMRequest(prompt="Hello")
        with pytest.raises(RateLimitError):
            adapter.generate_response(request)
