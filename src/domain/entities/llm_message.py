from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class LLMRequest(BaseModel):
    prompt: str
    system_instruction: Optional[str] = None
    schema_dict: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    temperature: float = 0.0

    class Config:
        populate_by_name = True


class LLMResponse(BaseModel):
    content: str
    raw_response: Dict[str, Any]
    provider_name: str
    latency_ms: int
