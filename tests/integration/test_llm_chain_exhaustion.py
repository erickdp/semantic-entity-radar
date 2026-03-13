import pytest
from unittest.mock import MagicMock
from src.domain.services.llm_orchestrator import LLMOrchestrator
from src.domain.entities.llm_message import LLMRequest
from src.domain.exceptions.llm_exceptions import (
    ChainExhaustedError,
    ProviderUnavailableError,
)


def test_orchestrator_exhausts_all_providers():
    mock_1 = MagicMock()
    mock_2 = MagicMock()

    mock_1.generate_response.side_effect = ProviderUnavailableError("P1 down")
    mock_2.generate_response.side_effect = ProviderUnavailableError("P2 down")

    orchestrator = LLMOrchestrator(providers=[mock_1, mock_2])
    request = LLMRequest(prompt="Hello")

    with pytest.raises(ChainExhaustedError) as excinfo:
        orchestrator.generate_response(request)

    assert "All providers failed" in str(excinfo.value)
    assert mock_1.generate_response.call_count == 1
    assert mock_2.generate_response.call_count == 1
