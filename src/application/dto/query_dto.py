from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequestDTO(BaseModel):
    query_text: str = Field(min_length=3, max_length=500)
    intent: Literal["opinion", "summary", "search"]
    language: str = Field(default="es")
    max_sources: int = Field(default=8, ge=1, le=20)


class CitationDTO(BaseModel):
    source_url: str
    source_network: str
    document_id: str
    chunk_id: str
    relevance_score: float
    metadata: dict[str, str]


class StructuredResponseDTO(BaseModel):
    query_id: str
    response_type: Literal["opinion", "summary", "search"]
    professional_text: str
    key_points: list[str]
    confidence_level: Literal["low", "medium", "high"]
    generated_at: datetime
    citations: list[CitationDTO]
