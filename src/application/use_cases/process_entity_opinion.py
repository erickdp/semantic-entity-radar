from src.domain.ports.outbound.llm_provider_port import LLMProviderPort
from src.domain.entities.llm_message import LLMRequest


class ProcessEntityOpinionUseCase:
    def __init__(self, llm_provider: LLMProviderPort):
        self.llm = llm_provider

    def execute(self, text: str):
        # Example implementation
        request = LLMRequest(
            prompt=f"Analyze the opinions in this text: {text}", temperature=0.7
        )
        return self.llm.generate_response(request)
