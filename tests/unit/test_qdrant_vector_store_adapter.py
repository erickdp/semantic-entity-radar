from dataclasses import dataclass
from typing import Any

from qdrant_client import models

from src.domain.entities.evidence import EvidenceChunk
from src.infrastructure.adapters.outbound.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)


@dataclass
class FakePoint:
    id: str
    score: float
    payload: dict[str, Any]


class FakeDenseEmbeddingAdapter:
    def embed_query(self, query_text: str) -> list[float]:
        _ = query_text
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeSparseEmbeddingAdapter:
    def embed_query(self, query_text: str) -> models.SparseVector:
        _ = query_text
        return models.SparseVector(indices=[1, 7], values=[0.8, 0.5])

    def embed_texts(self, texts: list[str]) -> list[models.SparseVector]:
        return [models.SparseVector(indices=[1, 7], values=[0.8, 0.5]) for _ in texts]


class FakeQdrantClient:
    def __init__(
        self, points: list[FakePoint] | None = None, raise_error: bool = False
    ) -> None:
        self._points = points or []
        self._raise_error = raise_error
        self.last_query_args: dict[str, Any] = {}
        self.last_upsert_args: dict[str, Any] = {}

    def query_points(self, **kwargs: Any) -> Any:
        self.last_query_args = kwargs
        if self._raise_error:
            raise RuntimeError("qdrant not available")
        return type("FakeQueryResponse", (), {"points": self._points})()

    def upsert(self, **kwargs: Any) -> Any:
        self.last_upsert_args = kwargs
        return type("FakeUpsertResult", (), {"status": "ok"})()


def _create_adapter(fake_client: FakeQdrantClient) -> QdrantVectorStoreAdapter:
    return QdrantVectorStoreAdapter(
        collection_name="entity_opinions",
        dense_vector_name="dense",
        sparse_vector_name="sparse",
        client=fake_client,
        dense_embedding_adapter=FakeDenseEmbeddingAdapter(),
        sparse_embedding_adapter=FakeSparseEmbeddingAdapter(),
    )


def test_semantic_search_uses_hybrid_prefetch() -> None:
    fake_client = FakeQdrantClient(
        points=[
            FakePoint(
                id="point-1",
                score=0.88,
                payload={
                    "chunk_id": "chunk-123",
                    "document_id": "doc-123",
                    "chunk_text": "Texto de evidencia",
                    "source_url": "https://x.com/source/123",
                    "source_network": "x",
                    "metadata": {"author": "@demo"},
                },
            )
        ]
    )
    adapter = _create_adapter(fake_client)

    result = adapter.semantic_search("opinion publica", k=1)

    assert len(result) == 1
    assert fake_client.last_query_args["collection_name"] == "entity_opinions"
    assert isinstance(fake_client.last_query_args["query"], models.FusionQuery)
    assert fake_client.last_query_args["query"].fusion == models.Fusion.RRF
    assert len(fake_client.last_query_args["prefetch"]) == 2
    assert fake_client.last_query_args["prefetch"][0].using == "dense"
    assert fake_client.last_query_args["prefetch"][1].using == "sparse"
    assert result[0].chunk_id == "chunk-123"
    assert result[0].dense_score == 0.88


def test_semantic_search_returns_fallback_when_not_configured() -> None:
    adapter = QdrantVectorStoreAdapter()

    result = adapter.semantic_search("opinion publica", k=2)

    assert len(result) == 2
    assert result[0].source_url.startswith("https://")


def test_semantic_search_returns_fallback_on_client_error() -> None:
    fake_client = FakeQdrantClient(raise_error=True)
    adapter = _create_adapter(fake_client)

    result = adapter.semantic_search("opinion publica", k=1)

    assert len(result) == 1
    assert result[0].chunk_id == "chunk-001"


def test_upsert_chunks_stores_dense_and_sparse_vectors() -> None:
    fake_client = FakeQdrantClient()
    adapter = _create_adapter(fake_client)
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
    assert fake_client.last_upsert_args["collection_name"] == "entity_opinions"
    points = fake_client.last_upsert_args["points"]
    assert len(points) == 1
    assert points[0].id == "chunk-555"
    assert set(points[0].vector.keys()) == {"dense", "sparse"}
