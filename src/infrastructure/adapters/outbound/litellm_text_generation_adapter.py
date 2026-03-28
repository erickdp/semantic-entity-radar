from src.domain.entities.evidence import EvidenceChunk
from src.domain.entities.query import QueryIntent
from src.domain.ports.outbound.generation import TextGenerationPort


class LiteLLMTextGenerationAdapter(TextGenerationPort):
    def generate_structured_response(
        self,
        query_text: str,
        ranked_evidence: list[EvidenceChunk],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        _ = intent
        evidence_count = len(ranked_evidence)
        text = (
            f"Analisis profesional sobre '{query_text}': se identifican "
            f"{evidence_count} evidencias relevantes de entidades publicas."
        )
        key_points = [
            "Se identifican posturas favorables y criticas en la conversacion publica.",
            "La evidencia se encuentra trazada con metadata por fuente.",
        ]
        return text, key_points
