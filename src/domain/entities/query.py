from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class QueryIntent(StrEnum):
    OPINION = "opinion"
    SUMMARY = "summary"
    SEARCH = "search"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class QueryTopic:
    query_id: str
    query_text: str
    intent: QueryIntent
    language: str = "es"
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Citation:
    source_url: str
    source_network: str
    document_id: str
    chunk_id: str
    relevance_score: float
    metadata: dict[str, str]


@dataclass(frozen=True)
class StructuredResponse:
    query_id: str
    response_type: QueryIntent
    professional_text: str
    key_points: list[str]
    citations: list[Citation]
    confidence_level: ConfidenceLevel
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
