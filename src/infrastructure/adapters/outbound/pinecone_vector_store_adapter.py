from dataclasses import replace
from typing import Any

from pinecone import Pinecone

from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import VectorStorePort


class PineconeHybridVectorStoreAdapter(VectorStorePort):
    def __init__(
        self,
        api_key: str = "",
        dense_index_name: str = "",
        sparse_index_name: str = "",
        dense_index_host: str = "",
        sparse_index_host: str = "",
        namespace: str = "default",
        text_field: str = "chunk_text",
        dense_model: str = "multilingual-e5-large",
        sparse_model: str = "pinecone-sparse-english-v0",
        cloud: str = "aws",
        region: str = "us-east-1",
        search_multiplier: int = 3,
        auto_create_indexes: bool = False,
        pc: Any | None = None,
        dense_index: Any | None = None,
        sparse_index: Any | None = None,
    ) -> None:
        self.dense_index_name = dense_index_name
        self.sparse_index_name = sparse_index_name
        self.dense_index_host = dense_index_host
        self.sparse_index_host = sparse_index_host
        self.namespace = namespace
        self.text_field = text_field
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.cloud = cloud
        self.region = region
        self.search_multiplier = max(search_multiplier, 1)
        self.auto_create_indexes = auto_create_indexes
        self._pc = pc
        self._dense_index = dense_index
        self._sparse_index = sparse_index
        self._enabled = bool(
            (api_key or pc is not None)
            and (dense_index_name or dense_index_host)
            and (sparse_index_name or sparse_index_host)
        ) or (dense_index is not None and sparse_index is not None)

        if (
            self._pc is None
            and self._enabled
            and (self._dense_index is None or self._sparse_index is None)
        ):
            self._pc = Pinecone(api_key=api_key)

    def semantic_search(self, query_text: str, k: int) -> list[EvidenceChunk]:
        if k <= 0:
            return []

        dense_index = self._get_dense_index()
        sparse_index = self._get_sparse_index()
        if dense_index is None or sparse_index is None:
            return self._fallback_samples(k)

        limit = max(k * self.search_multiplier, k)

        try:
            dense_response = dense_index.search_records(
                namespace=self.namespace,
                query=self._build_search_query(query_text, limit),
                fields=self._result_fields(),
            )
            sparse_response = sparse_index.search_records(
                namespace=self.namespace,
                query=self._build_search_query(query_text, limit),
                fields=self._result_fields(),
            )
        except Exception:
            return self._fallback_samples(k)

        dense_hits = self._extract_hits(dense_response)
        sparse_hits = self._extract_hits(sparse_response)
        merged: dict[str, EvidenceChunk] = {}
        fused_scores: dict[str, float] = {}

        self._merge_hits(
            merged=merged,
            fused_scores=fused_scores,
            hits=dense_hits,
            score_field="dense_score",
        )
        self._merge_hits(
            merged=merged,
            fused_scores=fused_scores,
            hits=sparse_hits,
            score_field="lexical_score",
        )

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                fused_scores.get(item.chunk_id, 0.0),
                item.dense_score + item.lexical_score,
            ),
            reverse=True,
        )
        return ranked[:k]

    def upsert_chunks(self, chunks: list[EvidenceChunk]) -> int:
        if not chunks:
            return 0

        dense_index = self._get_dense_index()
        sparse_index = self._get_sparse_index()
        if dense_index is None or sparse_index is None:
            return 0

        records = [self._to_record(chunk) for chunk in chunks]
        dense_index.upsert_records(namespace=self.namespace, records=records)
        sparse_index.upsert_records(namespace=self.namespace, records=records)
        return len(records)

    def _get_dense_index(self) -> Any | None:
        if self._dense_index is not None:
            return self._dense_index
        if not self._enabled or self._pc is None:
            return None

        try:
            host = self.dense_index_host or self._resolve_index_host(
                index_name=self.dense_index_name,
                model_name=self.dense_model,
            )
            self._dense_index = self._pc.Index(host=host)
        except Exception:
            return None

        return self._dense_index

    def _get_sparse_index(self) -> Any | None:
        if self._sparse_index is not None:
            return self._sparse_index
        if not self._enabled or self._pc is None:
            return None

        try:
            host = self.sparse_index_host or self._resolve_index_host(
                index_name=self.sparse_index_name,
                model_name=self.sparse_model,
            )
            self._sparse_index = self._pc.Index(host=host)
        except Exception:
            return None

        return self._sparse_index

    def _resolve_index_host(self, index_name: str, model_name: str) -> str:
        if self._pc is None:
            raise RuntimeError("Pinecone client is not configured.")

        if self._pc.has_index(index_name):
            return str(self._pc.describe_index(name=index_name).host)

        if not self.auto_create_indexes:
            raise RuntimeError(f"Pinecone index '{index_name}' was not found.")

        description = self._pc.create_index_for_model(
            name=index_name,
            cloud=self.cloud,
            region=self.region,
            embed={"model": model_name, "field_map": {"text": self.text_field}},
        )
        return str(description.host)

    def _build_search_query(self, query_text: str, top_k: int) -> dict[str, Any]:
        return {"inputs": {"text": query_text}, "top_k": top_k}

    def _result_fields(self) -> list[str]:
        return [
            self.text_field,
            "document_id",
            "source_url",
            "source_network",
            "metadata",
        ]

    def _merge_hits(
        self,
        merged: dict[str, EvidenceChunk],
        fused_scores: dict[str, float],
        hits: list[Any],
        score_field: str,
    ) -> None:
        for rank, hit in enumerate(hits, start=1):
            chunk = self._to_evidence_chunk(hit=hit, score_field=score_field)
            existing = merged.get(chunk.chunk_id)
            merged[chunk.chunk_id] = self._merge_chunk(existing, chunk)
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (60 + rank)
            )

    @staticmethod
    def _merge_chunk(
        existing: EvidenceChunk | None, incoming: EvidenceChunk
    ) -> EvidenceChunk:
        if existing is None:
            return incoming

        return replace(
            existing,
            document_id=existing.document_id or incoming.document_id,
            chunk_text=existing.chunk_text or incoming.chunk_text,
            source_url=existing.source_url or incoming.source_url,
            source_network=existing.source_network or incoming.source_network,
            metadata={**incoming.metadata, **existing.metadata},
            dense_score=max(existing.dense_score, incoming.dense_score),
            lexical_score=max(existing.lexical_score, incoming.lexical_score),
        )

    def _to_record(self, chunk: EvidenceChunk) -> dict[str, Any]:
        return {
            "_id": chunk.chunk_id,
            self.text_field: chunk.chunk_text,
            "document_id": chunk.document_id,
            "source_url": chunk.source_url,
            "source_network": chunk.source_network,
            "metadata": chunk.metadata,
        }

    def _to_evidence_chunk(self, hit: Any, score_field: str) -> EvidenceChunk:
        hit_id = str(self._get_value(hit, "_id", "id", default="unknown-chunk"))
        score = float(self._get_value(hit, "_score", "score", default=0.0) or 0.0)
        fields_raw = self._get_value(hit, "fields", default={})
        fields = fields_raw if isinstance(fields_raw, dict) else {}
        if isinstance(hit, dict):
            fields = {
                **{
                    key: value
                    for key, value in hit.items()
                    if key not in {"_id", "id", "_score", "score", "fields"}
                },
                **fields,
            }

        metadata_raw = fields.get("metadata", {})
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
        normalized_metadata = {str(key): str(value) for key, value in metadata.items()}

        chunk = EvidenceChunk(
            chunk_id=hit_id,
            document_id=str(fields.get("document_id", hit_id)),
            chunk_text=str(fields.get(self.text_field, "")),
            source_url=str(fields.get("source_url", "")),
            source_network=str(fields.get("source_network", "unknown")),
            metadata=normalized_metadata,
        )
        return replace(chunk, **{score_field: score})

    @staticmethod
    def _extract_hits(response: Any) -> list[Any]:
        if response is None:
            return []

        result = getattr(response, "result", None)
        if result is not None:
            hits = getattr(result, "hits", None)
            if hits is not None:
                return list(hits)
            if isinstance(result, dict):
                dict_hits = result.get("hits") or result.get("matches") or []
                return list(dict_hits)

        for attr_name in ("hits", "matches", "records"):
            hits = getattr(response, attr_name, None)
            if hits is not None:
                return list(hits)

        if isinstance(response, dict):
            dict_hits = (
                response.get("hits")
                or response.get("matches")
                or response.get("records")
                or []
            )
            return list(dict_hits)

        return []

    @staticmethod
    def _get_value(item: Any, *keys: str, default: Any = None) -> Any:
        for key in keys:
            if isinstance(item, dict) and key in item:
                return item[key]
            if hasattr(item, key):
                return getattr(item, key)
        return default

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
