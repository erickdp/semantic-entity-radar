from typing import Protocol

from src.domain.entities.evidence import EvidenceChunk
from src.domain.entities.query import QueryIntent


class TextGenerationPort(Protocol):
    def generate_structured_response(
        self,
        query_text: str,
        ranked_evidence: list[EvidenceChunk],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]: ...
