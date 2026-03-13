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


def test_embedding_port_contract() -> None:
    adapter = FastEmbedAdapter()
    vector = adapter.embed_query("opinion publica")

    assert isinstance(vector, list)
    assert all(isinstance(value, float) for value in vector)
    assert len(vector) > 0


def test_vector_store_and_lexical_contracts() -> None:
    dense = QdrantVectorStoreAdapter().semantic_search([0.1, 0.2, 0.3], k=2)
    lexical = BM25LexicalAdapter().keyword_search("opinion publica", k=2)

    assert dense
    assert lexical
    assert dense[0].source_url.startswith("https://")
    assert lexical[0].source_network == "x"


def test_reranker_port_contract() -> None:
    candidates = QdrantVectorStoreAdapter().semantic_search([0.1, 0.2, 0.3], k=2)
    ranked = HuggingFaceRerankerAdapter().rerank("opinion publica", candidates, top_n=2)

    assert len(ranked) <= 2
    assert ranked[0].rerank_score >= 0
