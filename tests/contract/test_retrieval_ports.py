import math

from qdrant_client import models

from src.infrastructure.adapters.outbound.bm25_lexical_adapter import BM25LexicalAdapter
from src.infrastructure.adapters.outbound.embedding_fastembed_adapter import (
    FastEmbedAdapter,
)
from src.infrastructure.adapters.outbound.hf_reranker_adapter import (
    HuggingFaceRerankerAdapter,
)
from src.infrastructure.adapters.outbound.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)


class FakeFastEmbedModel:
    def embed(self, texts: list[str]):
        for _ in texts:
            yield [3.0, 4.0, 0.0]


class FakeSparseEmbedding:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class FakeSparseModel:
    def query_embed(self, texts: list[str]):
        for _ in texts:
            yield FakeSparseEmbedding(indices=[1, 8], values=[0.9, 0.2])

    def embed(self, texts: list[str]):
        for _ in texts:
            yield FakeSparseEmbedding(indices=[2, 6], values=[0.7, 0.5])


def test_embedding_port_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        FastEmbedAdapter, "_get_model", lambda self: FakeFastEmbedModel()
    )
    adapter = FastEmbedAdapter()
    vector = adapter.embed_query("opinion publica")

    assert isinstance(vector, list)
    assert all(isinstance(value, float) for value in vector)
    assert len(vector) > 0
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)


def test_sparse_embedding_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        BM25LexicalAdapter, "_get_model", lambda self: FakeSparseModel()
    )
    adapter = BM25LexicalAdapter()

    query_vector = adapter.embed_query("opinion publica")
    document_vectors = adapter.embed_texts(["texto 1", "texto 2"])

    assert isinstance(query_vector, models.SparseVector)
    assert query_vector.indices
    assert query_vector.values
    assert len(document_vectors) == 2


def test_vector_store_contract(
    qdrant_vector_store_adapter: QdrantVectorStoreAdapter,
) -> None:
    dense = qdrant_vector_store_adapter.semantic_search("opinion publica", k=2)

    assert dense
    assert dense[0].source_url.startswith("https://")
    assert dense[0].source_network == "x"


def test_reranker_port_contract(
    qdrant_vector_store_adapter: QdrantVectorStoreAdapter,
) -> None:
    candidates = qdrant_vector_store_adapter.semantic_search("opinion publica", k=2)
    ranked = HuggingFaceRerankerAdapter().rerank("opinion publica", candidates, top_n=2)

    assert len(ranked) <= 2
    assert ranked[0].rerank_score >= 0
