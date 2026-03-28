from typing import Any

from src.domain.ports.outbound.retrieval import EmbeddingPort
from src.infrastructure.adapters.outbound.embedding_utils import normalize_l2


DEFAULT_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedAdapter(EmbeddingPort):
    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is not installed. Install qdrant-client[fastembed]."
                ) from exc
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_query(self, query_text: str) -> list[float]:
        return self.embed_texts([query_text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        normalized_texts = [text.strip() or text or " " for text in texts]
        vectors = list(model.embed(normalized_texts))
        if len(vectors) != len(normalized_texts):
            raise RuntimeError("FastEmbed returned an unexpected number of embeddings.")

        return [
            normalize_l2(vector.tolist() if hasattr(vector, "tolist") else list(vector))
            for vector in vectors
        ]
