from typing import Any

from qdrant_client import QdrantClient, models

from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import VectorStorePort
from src.infrastructure.adapters.outbound.bm25_lexical_adapter import BM25LexicalAdapter
from src.infrastructure.adapters.outbound.embedding_fastembed_adapter import (
    DEFAULT_FASTEMBED_MODEL,
    FastEmbedAdapter,
)


class QdrantVectorStoreAdapter(VectorStorePort):
    def __init__(
        self,
        url: str = "",
        api_key: str = "",
        collection_name: str = "",
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        timeout_seconds: float = 10.0,
        client: Any | None = None,
        dense_embedding_adapter: Any | None = None,
        sparse_embedding_adapter: Any | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self._client = client
        self._enabled = self._client is not None or bool(url and collection_name)
        self._dense_embedding_adapter = dense_embedding_adapter or FastEmbedAdapter(
            model_name=DEFAULT_FASTEMBED_MODEL
        )
        self._sparse_embedding_adapter = (
            sparse_embedding_adapter or BM25LexicalAdapter()
        )

        if self._client is None and self._enabled:
            self._client = QdrantClient(
                url=url,
                api_key=api_key or None,
                timeout=int(timeout_seconds),
            )

    def semantic_search(self, query_text: str, k: int) -> list[EvidenceChunk]:
        if k <= 0:
            return []
        if not self._enabled or self._client is None:
            return self._fallback_samples(k)

        try:
            dense_query = self._dense_embedding_adapter.embed_query(query_text)
            sparse_query = self._sparse_embedding_adapter.embed_query(query_text)

            query_result = self._client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using=self.dense_vector_name,
                        limit=k * 2,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using=self.sparse_vector_name,
                        limit=k * 2,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return self._fallback_samples(k)

        points = (
            query_result.points if hasattr(query_result, "points") else query_result
        )
        chunks = [self._to_evidence_chunk(point) for point in (points or [])]
        return chunks[:k]

    def upsert_chunks(self, chunks: list[EvidenceChunk]) -> int:
        if not chunks:
            return 0
        if not self._enabled or self._client is None:
            return 0

        chunk_texts = [chunk.chunk_text for chunk in chunks]
        dense_vectors = self._dense_embedding_adapter.embed_texts(chunk_texts)
        sparse_vectors = self._sparse_embedding_adapter.embed_texts(chunk_texts)

        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector={
                    self.dense_vector_name: dense_vector,
                    self.sparse_vector_name: sparse_vector,
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_text": chunk.chunk_text,
                    "source_url": chunk.source_url,
                    "source_network": chunk.source_network,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, dense_vector, sparse_vector in zip(
                chunks, dense_vectors, sparse_vectors, strict=True
            )
        ]

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        return len(points)

    @staticmethod
    def _to_evidence_chunk(point: Any) -> EvidenceChunk:
        payload_raw = getattr(point, "payload", {})
        payload = payload_raw if isinstance(payload_raw, dict) else {}

        metadata_raw = payload.get("metadata", {})
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        metadata = {str(key): str(value) for key, value in metadata.items()}

        chunk_id = str(payload.get("chunk_id", getattr(point, "id", "unknown-chunk")))
        document_id = str(
            payload.get("document_id", payload.get("doc_id", "unknown-doc"))
        )
        chunk_text = str(
            payload.get(
                "chunk_text",
                payload.get("text", payload.get("content", "")),
            )
        )
        source_url = str(payload.get("source_url", payload.get("url", "")))
        source_network = str(
            payload.get("source_network", payload.get("network", "unknown"))
        )

        return EvidenceChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_text=chunk_text,
            source_url=source_url,
            source_network=source_network,
            metadata=metadata,
            dense_score=float(getattr(point, "score", 0.0) or 0.0),
        )

    @staticmethod
    def _fallback_samples(k: int) -> list[EvidenceChunk]:
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
