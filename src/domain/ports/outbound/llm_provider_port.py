from abc import ABC, abstractmethod
from src.domain.entities.llm_message import LLMRequest, LLMResponse


class LLMProviderPort(ABC):
    @abstractmethod
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Sends a request to the LLM and returns a normalized response.
        Should raise specific exceptions for failures.
        """
        pass
