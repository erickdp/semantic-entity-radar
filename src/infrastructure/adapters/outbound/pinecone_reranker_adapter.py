from dataclasses import replace
from typing import Any

from pinecone import Pinecone

from src.domain.entities.evidence import EvidenceChunk
from src.domain.ports.outbound.retrieval import RerankerPort


class PineconeRerankerAdapter(RerankerPort):
    def __init__(
        self,
        api_key: str = "",
        model_name: str = "bge-reranker-v2-m3",
        rank_field: str = "chunk_text",
        pc: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.rank_field = rank_field
        self._pc = pc
        self._enabled = bool(api_key or pc is not None)

        if self._pc is None and self._enabled:
            self._pc = Pinecone(api_key=api_key)

    def rerank(
        self, query_text: str, candidates: list[EvidenceChunk], top_n: int
    ) -> list[EvidenceChunk]:
        if top_n <= 0 or not candidates:
            return []

        if not self._enabled or self._pc is None:
            return self._fallback_rank(candidates, top_n)

        documents = [self._to_document(candidate) for candidate in candidates]

        try:
            response = self._pc.inference.rerank(
                model=self.model_name,
                query=query_text,
                documents=documents,
                rank_fields=[self.rank_field],
                top_n=top_n,
                return_documents=True,
            )
        except Exception:
            return self._fallback_rank(candidates, top_n)

        reranked = self._extract_reranked_candidates(
            response=response, candidates=candidates
        )
        return reranked[:top_n] if reranked else self._fallback_rank(candidates, top_n)

    def _to_document(self, candidate: EvidenceChunk) -> dict[str, Any]:
        return {
            "chunk_id": candidate.chunk_id,
            self.rank_field: candidate.chunk_text,
            "document_id": candidate.document_id,
            "source_url": candidate.source_url,
            "source_network": candidate.source_network,
            "metadata": candidate.metadata,
        }

    def _extract_reranked_candidates(
        self, response: Any, candidates: list[EvidenceChunk]
    ) -> list[EvidenceChunk]:
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if not data:
            return []

        by_chunk_id = {candidate.chunk_id: candidate for candidate in candidates}
        ranked: list[EvidenceChunk] = []

        for item in data:
            score = float(self._get_value(item, "score", default=0.0) or 0.0)
            index = self._get_value(item, "index")
            document = self._get_value(item, "document", default={}) or {}
            chunk_id = self._extract_chunk_id(
                document=document, index=index, candidates=candidates
            )
            candidate = by_chunk_id.get(chunk_id)

            if candidate is None:
                continue

            ranked.append(replace(candidate, rerank_score=score))

        return ranked

    @staticmethod
    def _extract_chunk_id(
        document: Any, index: Any, candidates: list[EvidenceChunk]
    ) -> str:
        if isinstance(document, dict):
            if "chunk_id" in document:
                return str(document["chunk_id"])
            if "id" in document:
                return str(document["id"])

        if isinstance(index, int) and 0 <= index < len(candidates):
            return candidates[index].chunk_id

        return ""

    @staticmethod
    def _fallback_rank(
        candidates: list[EvidenceChunk], top_n: int
    ) -> list[EvidenceChunk]:
        ranked = sorted(
            candidates,
            key=lambda item: item.dense_score + item.lexical_score,
            reverse=True,
        )
        selected = ranked[:top_n]
        return [
            replace(item, rerank_score=item.dense_score + item.lexical_score)
            for item in selected
        ]

    @staticmethod
    def _get_value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)
