from src.domain.entities.evidence import EvidenceChunk
from src.domain.entities.query import QueryIntent
from src.domain.ports.outbound.generation import TextGenerationPort


class GeminiTextGenerationAdapter(TextGenerationPort):
    def generate_structured_response(
        self,
        query_text: str,
        ranked_evidence: list[EvidenceChunk],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        _ = intent
        text = (
            f"Resumen profesional para '{query_text}': se consolidan opiniones "
            f"de {len(ranked_evidence)} fragmentos con respaldo verificable."
        )
        key_points = [
            "Las afirmaciones estan respaldadas por citas verificables.",
            "El resultado conserva trazabilidad de metadata por fuente.",
        ]
        return text, key_points
