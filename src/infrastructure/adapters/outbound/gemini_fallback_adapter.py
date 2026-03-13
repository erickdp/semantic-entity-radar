from google import genai
from google.genai import types
import time
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from src.domain.ports.outbound.llm_provider_port import LLMProviderPort
from src.domain.entities.llm_message import LLMRequest, LLMResponse
from src.domain.exceptions.llm_exceptions import (
    RateLimitError,
    ProviderUnavailableError,
    CreditExhaustionError,
)

RETRY_WAIT_SEQUENCE = [wait_fixed(1), wait_fixed(3), wait_fixed(5)]


class GeminiAdapter(LLMProviderPort):
    def __init__(self, model_name: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.provider_name = "Gemini"

    @retry(
        stop=stop_after_attempt(3), wait=wait_chain(*RETRY_WAIT_SEQUENCE), reraise=True
    )
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=request.prompt,
                config=types.GenerateContentConfig(
                    temperature=request.temperature,
                    system_instruction=request.system_instruction,
                ),
            )
            latency = int((time.time() - start_time) * 1000)
            return LLMResponse(
                content=response.text,
                raw_response={"usage": getattr(response, "usage_metadata", {})},
                provider_name=self.provider_name,
                latency_ms=latency,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                raise RateLimitError(str(e))
            if "credit" in err_msg or "balance" in err_msg or "location" in err_msg:
                raise CreditExhaustionError(str(e))
            raise ProviderUnavailableError(str(e))
