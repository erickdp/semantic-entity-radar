from src.domain.ports.outbound.retrieval import EmbeddingPort


class FastEmbedAdapter(EmbeddingPort):
    def embed_query(self, query_text: str) -> list[float]:
        length = max(len(query_text), 1)
        return [length / 100.0, (length % 7) / 10.0, 0.42]
