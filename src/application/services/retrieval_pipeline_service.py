from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import (
    RerankerPort,
    VectorStorePort,
)


class RetrievalPipelineService:
    def __init__(
        self,
        vector_store_port: VectorStorePort,
        reranker_port: RerankerPort,
    ) -> None:
        self.vector_store_port = vector_store_port
        self.reranker_port = reranker_port

    def retrieve(self, query_text: str, max_sources: int) -> list[EvidenceChunk]:
        fused = self.vector_store_port.semantic_search(
            query_text=query_text, k=max_sources * 2
        )
        return self.reranker_port.rerank(
            query_text=query_text, candidates=fused, top_n=max_sources
        )
