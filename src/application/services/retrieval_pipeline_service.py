from dataclasses import replace

from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import (
    EmbeddingPort,
    LexicalSearchPort,
    RerankerPort,
    VectorStorePort,
)


class RetrievalPipelineService:
    def __init__(
        self,
        embedding_port: EmbeddingPort,
        vector_store_port: VectorStorePort,
        lexical_search_port: LexicalSearchPort,
        reranker_port: RerankerPort,
    ) -> None:
        self.embedding_port = embedding_port
        self.vector_store_port = vector_store_port
        self.lexical_search_port = lexical_search_port
        self.reranker_port = reranker_port

    def retrieve(self, query_text: str, max_sources: int) -> list[EvidenceChunk]:
        query_embedding = self.embedding_port.embed_query(query_text)
        dense_candidates = self.vector_store_port.semantic_search(
            query_embedding, k=max_sources * 2
        )
        lexical_candidates = self.lexical_search_port.keyword_search(
            query_text, k=max_sources * 2
        )

        fused = self._fuse(dense_candidates, lexical_candidates)
        return self.reranker_port.rerank(
            query_text=query_text, candidates=fused, top_n=max_sources
        )

    @staticmethod
    def _fuse(
        dense: list[EvidenceChunk], lexical: list[EvidenceChunk]
    ) -> list[EvidenceChunk]:
        merged: dict[str, EvidenceChunk] = {item.chunk_id: item for item in dense}
        for item in lexical:
            if item.chunk_id in merged:
                current = merged[item.chunk_id]
                merged[item.chunk_id] = replace(
                    current,
                    lexical_score=item.lexical_score,
                    dense_score=max(current.dense_score, item.dense_score),
                )
            else:
                merged[item.chunk_id] = item
        return list(merged.values())
