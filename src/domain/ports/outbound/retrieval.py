from typing import Protocol

from src.domain.entities.evidence import EvidenceChunk


class EmbeddingPort(Protocol):
    def embed_query(self, query_text: str) -> list[float]: ...


class VectorStorePort(Protocol):
    def semantic_search(
        self, query_embedding: list[float], k: int
    ) -> list[EvidenceChunk]: ...


class LexicalSearchPort(Protocol):
    def keyword_search(self, query_text: str, k: int) -> list[EvidenceChunk]: ...


class RerankerPort(Protocol):
    def rerank(
        self, query_text: str, candidates: list[EvidenceChunk], top_n: int
    ) -> list[EvidenceChunk]: ...
