from src.domain.entities.evidence import EvidenceChunk
from src.domain.entities.query import Citation


class CitationMappingService:
    @staticmethod
    def from_chunks(chunks: list[EvidenceChunk]) -> list[Citation]:
        return [
            Citation(
                source_url=chunk.source_url,
                source_network=chunk.source_network,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                relevance_score=chunk.rerank_score,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
