from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    document_id: str
    chunk_text: str
    source_url: str
    source_network: str
    metadata: dict[str, str] = field(default_factory=dict)
    dense_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    query_id: str
    chunk: EvidenceChunk
    fused_score: float
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
