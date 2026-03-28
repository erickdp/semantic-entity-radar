from dataclasses import dataclass

from src.domain.entities.evidence import EvidenceChunk
from src.infrastructure.adapters.outbound.pinecone_reranker_adapter import (
    PineconeRerankerAdapter,
)


@dataclass
class FakeRerankItem:
    index: int
    score: float
    document: dict[str, str]


class FakeInference:
    def __init__(self, data: list[FakeRerankItem]) -> None:
        self.data = data
        self.last_args: dict[str, object] = {}

    def rerank(self, **kwargs):
        self.last_args = kwargs
        return type("FakeRerankResponse", (), {"data": self.data})()


class FakePinecone:
    def __init__(self, data: list[FakeRerankItem]) -> None:
        self.inference = FakeInference(data=data)


def test_rerank_uses_pinecone_inference() -> None:
    pc = FakePinecone(
        data=[
            FakeRerankItem(index=1, score=0.95, document={"chunk_id": "chunk-2"}),
            FakeRerankItem(index=0, score=0.81, document={"chunk_id": "chunk-1"}),
        ]
    )
    adapter = PineconeRerankerAdapter(
        model_name="bge-reranker-v2-m3",
        rank_field="chunk_text",
        pc=pc,
    )
    candidates = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_text="Texto 1",
            source_url="https://x.com/1",
            source_network="x",
            dense_score=0.7,
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            chunk_text="Texto 2",
            source_url="https://x.com/2",
            source_network="x",
            dense_score=0.6,
            lexical_score=0.8,
        ),
    ]

    ranked = adapter.rerank("opinion publica", candidates, top_n=2)

    assert [item.chunk_id for item in ranked] == ["chunk-2", "chunk-1"]
    assert ranked[0].rerank_score == 0.95
    assert pc.inference.last_args["model"] == "bge-reranker-v2-m3"


def test_rerank_falls_back_to_score_sorting_when_not_configured() -> None:
    adapter = PineconeRerankerAdapter()
    candidates = [
        EvidenceChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_text="Texto 1",
            source_url="https://x.com/1",
            source_network="x",
            dense_score=0.5,
        ),
        EvidenceChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            chunk_text="Texto 2",
            source_url="https://x.com/2",
            source_network="x",
            dense_score=0.4,
            lexical_score=0.7,
        ),
    ]

    ranked = adapter.rerank("opinion publica", candidates, top_n=1)

    assert len(ranked) == 1
    assert ranked[0].chunk_id == "chunk-2"
    assert ranked[0].rerank_score == 1.1
