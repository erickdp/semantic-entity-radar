from dataclasses import replace

from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import RerankerPort


class HuggingFaceRerankerAdapter(RerankerPort):
    def rerank(
        self, query_text: str, candidates: list[EvidenceChunk], top_n: int
    ) -> list[EvidenceChunk]:
        _ = query_text
        ranked = sorted(
            candidates,
            key=lambda item: item.dense_score + item.lexical_score,
            reverse=True,
        )
        selected = ranked[:top_n]
        return [
            replace(item, rerank_score=item.dense_score + item.lexical_score)
            for item in selected
        ]
