import litellm
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


class LiteLLMAdapter(LLMProviderPort):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.provider_name = f"LiteLLM({model_name})"

    @retry(
        stop=stop_after_attempt(3), wait=wait_chain(*RETRY_WAIT_SEQUENCE), reraise=True
    )
    def generate_response(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
            )
            latency = int((time.time() - start_time) * 1000)
            return LLMResponse(
                content=response.choices[0].message.content,
                raw_response=response.dict(),
                provider_name=self.provider_name,
                latency_ms=latency,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "rate limit" in err_msg:
                raise RateLimitError(str(e))
            if "402" in err_msg or "credit" in err_msg or "balance" in err_msg:
                raise CreditExhaustionError(str(e))
            raise ProviderUnavailableError(str(e))
