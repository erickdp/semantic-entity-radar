from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import LexicalSearchPort


class BM25LexicalAdapter(LexicalSearchPort):
    def keyword_search(self, query_text: str, k: int) -> list[EvidenceChunk]:
        _ = query_text
        samples = [
            EvidenceChunk(
                chunk_id="chunk-001",
                document_id="doc-001",
                chunk_text="La entidad publica A expreso apoyo al tema en X.",
                source_url="https://x.com/source/1",
                source_network="x",
                metadata={"author": "@source1", "published_at": "2026-03-01"},
                lexical_score=0.94,
            ),
            EvidenceChunk(
                chunk_id="chunk-003",
                document_id="doc-003",
                chunk_text="La entidad publica C mantiene una opinion neutral.",
                source_url="https://x.com/source/3",
                source_network="x",
                metadata={"author": "@source3", "published_at": "2026-03-03"},
                lexical_score=0.81,
            ),
        ]
        return samples[:k]
