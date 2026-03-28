import pytest
from src.domain.ports.outbound.llm_provider_port import LLMProviderPort


def test_llm_provider_port_is_abstract():
    with pytest.raises(TypeError):
        LLMProviderPort()


class MockProvider(LLMProviderPort):
    def generate_response(self, request):
        return "ok"


def test_mock_provider_implementation():
    provider = MockProvider()
    assert provider.generate_response(None) == "ok"
