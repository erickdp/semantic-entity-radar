import math
from dataclasses import dataclass

from qdrant_client import models

from src.infrastructure.adapters.outbound.bm25_lexical_adapter import BM25LexicalAdapter
from src.infrastructure.adapters.outbound.embedding_fastembed_adapter import (
    FastEmbedAdapter,
)


class FakeFastEmbedModel:
    def embed(self, texts: list[str]):
        for _ in texts:
            yield [1.0, 2.0, 2.0]


@dataclass
class FakeSparseEmbedding:
    indices: list[int]
    values: list[float]


class FakeSparseTextEmbeddingModel:
    def query_embed(self, texts: list[str]):
        for _ in texts:
            yield FakeSparseEmbedding(indices=[1, 3, 9], values=[0.8, 0.4, 0.2])

    def embed(self, texts: list[str]):
        for _ in texts:
            yield FakeSparseEmbedding(indices=[2, 5], values=[0.7, 0.3])


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def test_fastembed_adapter_returns_normalized_query_vector(monkeypatch) -> None:
    monkeypatch.setattr(
        FastEmbedAdapter,
        "_get_model",
        lambda self: FakeFastEmbedModel(),
    )
    adapter = FastEmbedAdapter()

    result = adapter.embed_query("consulta")

    assert isinstance(result, list)
    assert math.isclose(_norm(result), 1.0)


def test_fastembed_adapter_returns_normalized_batch_vectors(monkeypatch) -> None:
    monkeypatch.setattr(
        FastEmbedAdapter,
        "_get_model",
        lambda self: FakeFastEmbedModel(),
    )
    adapter = FastEmbedAdapter()

    result = adapter.embed_texts(["consulta 1", "consulta 2"])

    assert len(result) == 2
    assert all(math.isclose(_norm(vector), 1.0) for vector in result)


def test_bm25_adapter_returns_sparse_query_vector(monkeypatch) -> None:
    monkeypatch.setattr(
        BM25LexicalAdapter,
        "_get_model",
        lambda self: FakeSparseTextEmbeddingModel(),
    )
    adapter = BM25LexicalAdapter()

    result = adapter.embed_query("consulta")

    assert isinstance(result, models.SparseVector)
    assert result.indices == [1, 3, 9]
    assert result.values == [0.8, 0.4, 0.2]


def test_bm25_adapter_returns_sparse_batch_vectors(monkeypatch) -> None:
    monkeypatch.setattr(
        BM25LexicalAdapter,
        "_get_model",
        lambda self: FakeSparseTextEmbeddingModel(),
    )
    adapter = BM25LexicalAdapter()

    result = adapter.embed_texts(["doc 1", "doc 2"])

    assert len(result) == 2
    assert all(isinstance(vector, models.SparseVector) for vector in result)
    assert result[0].indices == [2, 5]
    assert result[0].values == [0.7, 0.3]
