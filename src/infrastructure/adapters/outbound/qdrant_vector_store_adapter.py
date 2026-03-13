from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import VectorStorePort


class QdrantVectorStoreAdapter(VectorStorePort):
    def semantic_search(
        self, query_embedding: list[float], k: int
    ) -> list[EvidenceChunk]:
        _ = query_embedding
        samples = [
            EvidenceChunk(
                chunk_id="chunk-001",
                document_id="doc-001",
                chunk_text="La entidad publica A expreso apoyo al tema en X.",
                source_url="https://x.com/source/1",
                source_network="x",
                metadata={"author": "@source1", "published_at": "2026-03-01"},
                dense_score=0.92,
            ),
            EvidenceChunk(
                chunk_id="chunk-002",
                document_id="doc-002",
                chunk_text="La entidad publica B presento una postura critica.",
                source_url="https://x.com/source/2",
                source_network="x",
                metadata={"author": "@source2", "published_at": "2026-03-02"},
                dense_score=0.87,
            ),
        ]
        return samples[:k]
