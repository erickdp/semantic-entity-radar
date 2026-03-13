from unittest.mock import MagicMock
from src.domain.services.llm_orchestrator import LLMOrchestrator
from src.domain.entities.llm_message import LLMRequest, LLMResponse
from src.domain.exceptions.llm_exceptions import RateLimitError


def test_orchestrator_fallbacks_on_failure():
    mock_gemini = MagicMock()
    mock_litellm = MagicMock()

    # Gemini fails with RateLimitError (after its own retries)
    mock_gemini.generate_response.side_effect = RateLimitError("Gemini failed")

    # LiteLLM succeeds
    expected_response = LLMResponse(
        content="LiteLLM response",
        raw_response={},
        provider_name="LiteLLM",
        latency_ms=100,
    )
    mock_litellm.generate_response.return_value = expected_response

    orchestrator = LLMOrchestrator(providers=[mock_gemini, mock_litellm])
    request = LLMRequest(prompt="Hello")

    response = orchestrator.generate_response(request)

    assert response.content == "LiteLLM response"
    assert response.provider_name == "LiteLLM"
    mock_gemini.generate_response.assert_called_once()
    mock_litellm.generate_response.assert_called_once()
