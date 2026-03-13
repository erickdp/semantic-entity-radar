from src.domain.ports.outbound.retrieval import EmbeddingPort


class ExternalModelEmbeddingAdapter(EmbeddingPort):
    def embed_query(self, query_text: str) -> list[float]:
        checksum = sum(ord(char) for char in query_text) % 100
        return [0.11, checksum / 100.0, 0.73]
