from src.domain.entities.evidence import EvidenceChunk
from src.domain.entities.query import QueryIntent
from src.domain.ports.outbound.generation import TextGenerationPort


class TextGenerationService:
    def __init__(
        self, primary_adapter: TextGenerationPort, fallback_adapter: TextGenerationPort
    ) -> None:
        self.primary_adapter = primary_adapter
        self.fallback_adapter = fallback_adapter

    def generate(
        self,
        query_text: str,
        ranked_evidence: list[EvidenceChunk],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        try:
            return self.primary_adapter.generate_structured_response(
                query_text=query_text,
                ranked_evidence=ranked_evidence,
                intent=intent,
            )
        except Exception:
            return self.fallback_adapter.generate_structured_response(
                query_text=query_text,
                ranked_evidence=ranked_evidence,
                intent=intent,
            )
