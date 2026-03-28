from dataclasses import dataclass
from typing import Any

from src.domain.entities.evidence import EvidenceChunk
from src.infrastructure.adapters.outbound.pinecone_vector_store_adapter import (
    PineconeHybridVectorStoreAdapter,
)


@dataclass
class FakeSearchHit:
    _id: str
    _score: float
    fields: dict[str, Any]


class FakeIndex:
    def __init__(self, hits: list[FakeSearchHit] | None = None) -> None:
        self.hits = hits or []
        self.last_search_args: dict[str, Any] = {}
        self.last_upsert_args: dict[str, Any] = {}

    def search_records(self, **kwargs: Any) -> Any:
        self.last_search_args = kwargs
        result = type("FakeSearchResult", (), {"hits": self.hits})()
        return type("FakeSearchResponse", (), {"result": result})()

    def upsert_records(self, **kwargs: Any) -> Any:
        self.last_upsert_args = kwargs
        return {"upserted_count": len(kwargs.get("records", []))}


def _create_adapter(
    dense_hits: list[FakeSearchHit] | None = None,
    sparse_hits: list[FakeSearchHit] | None = None,
) -> tuple[PineconeHybridVectorStoreAdapter, FakeIndex, FakeIndex]:
    dense_index = FakeIndex(hits=dense_hits)
    sparse_index = FakeIndex(hits=sparse_hits)
    adapter = PineconeHybridVectorStoreAdapter(
        namespace="entity-opinion",
        text_field="chunk_text",
        dense_index=dense_index,
        sparse_index=sparse_index,
        search_multiplier=2,
    )
    return adapter, dense_index, sparse_index


def test_semantic_search_merges_dense_and_sparse_hits() -> None:
    adapter, dense_index, sparse_index = _create_adapter(
        dense_hits=[
            FakeSearchHit(
                _id="chunk-1",
                _score=0.91,
                fields={
                    "chunk_text": "Texto 1",
                    "document_id": "doc-1",
                    "source_url": "https://x.com/1",
                    "source_network": "x",
                    "metadata": {"author": "@one"},
                },
            ),
            FakeSearchHit(
                _id="chunk-2",
                _score=0.73,
                fields={
                    "chunk_text": "Texto 2",
                    "document_id": "doc-2",
                    "source_url": "https://x.com/2",
                    "source_network": "x",
                    "metadata": {"author": "@two"},
                },
            ),
        ],
        sparse_hits=[
            FakeSearchHit(
                _id="chunk-2",
                _score=0.88,
                fields={
                    "chunk_text": "Texto 2",
                    "document_id": "doc-2",
                    "source_url": "https://x.com/2",
                    "source_network": "x",
                    "metadata": {"author": "@two"},
                },
            ),
            FakeSearchHit(
                _id="chunk-3",
                _score=0.69,
                fields={
                    "chunk_text": "Texto 3",
                    "document_id": "doc-3",
                    "source_url": "https://x.com/3",
                    "source_network": "x",
                    "metadata": {"author": "@three"},
                },
            ),
        ],
    )

    result = adapter.semantic_search("opinion publica", k=2)

    assert len(result) == 2
    assert dense_index.last_search_args["namespace"] == "entity-opinion"
    assert sparse_index.last_search_args["query"]["top_k"] == 4
    assert result[0].chunk_id == "chunk-2"
    assert result[0].dense_score == 0.73
    assert result[0].lexical_score == 0.88


def test_semantic_search_returns_fallback_when_not_configured() -> None:
    adapter = PineconeHybridVectorStoreAdapter()

    result = adapter.semantic_search("opinion publica", k=2)

    assert len(result) == 2
    assert result[0].source_url.startswith("https://")


def test_upsert_chunks_writes_records_to_both_indexes() -> None:
    adapter, dense_index, sparse_index = _create_adapter()
    chunks = [
        EvidenceChunk(
            chunk_id="chunk-555",
            document_id="doc-555",
            chunk_text="Texto de prueba",
            source_url="https://x.com/source/555",
            source_network="x",
            metadata={"author": "@tester"},
        )
    ]

    stored = adapter.upsert_chunks(chunks)

    assert stored == 1
    assert dense_index.last_upsert_args["namespace"] == "entity-opinion"
    assert sparse_index.last_upsert_args["records"][0]["_id"] == "chunk-555"
    assert (
        sparse_index.last_upsert_args["records"][0]["chunk_text"] == "Texto de prueba"
    )
