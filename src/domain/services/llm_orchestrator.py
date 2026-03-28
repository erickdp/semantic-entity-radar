import logging
from typing import List
from src.domain.ports.outbound.llm_provider_port import LLMProviderPort
from src.domain.entities.llm_message import LLMRequest, LLMResponse
from src.domain.exceptions.llm_exceptions import ChainExhaustedError

logger = logging.getLogger(__name__)


class LLMOrchestrator(LLMProviderPort):
    def __init__(self, providers: List[LLMProviderPort]):
        self.providers = providers

    def generate_response(self, request: LLMRequest) -> LLMResponse:
        last_exception = None
        for provider in self.providers:
            try:
                logger.info(
                    f"Attempting LLM request with {provider.__class__.__name__}"
                )
                return provider.generate_response(request)
            except Exception as e:
                logger.warning(
                    f"Provider {provider.__class__.__name__} failed: {e}. Trying next..."
                )
                last_exception = e
                continue

        raise ChainExhaustedError(f"All providers failed. Last error: {last_exception}")
