from typing import Any

from qdrant_client import models


DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class BM25LexicalAdapter:
    def __init__(self, model_name: str = DEFAULT_SPARSE_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise RuntimeError(
                    "fastembed is not installed. Install qdrant-client[fastembed]."
                ) from exc
            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def embed_query(self, query_text: str) -> models.SparseVector:
        text = query_text.strip() or query_text or " "
        model = self._get_model()

        if hasattr(model, "query_embed"):
            query_vectors = list(model.query_embed([text]))
        else:
            query_vectors = list(model.embed([text]))

        if not query_vectors:
            raise RuntimeError(
                "SparseTextEmbedding did not return any query embedding."
            )
        return self._to_sparse_vector(query_vectors[0])

    def embed_texts(self, texts: list[str]) -> list[models.SparseVector]:
        model = self._get_model()
        normalized_texts = [text.strip() or text or " " for text in texts]
        vectors = list(model.embed(normalized_texts))

        if len(vectors) != len(normalized_texts):
            raise RuntimeError(
                "SparseTextEmbedding returned an unexpected number of vectors."
            )

        return [self._to_sparse_vector(vector) for vector in vectors]

    @staticmethod
    def _to_sparse_vector(vector: Any) -> models.SparseVector:
        indices_raw = getattr(vector, "indices", [])
        values_raw = getattr(vector, "values", [])

        indices = (
            indices_raw.tolist()
            if hasattr(indices_raw, "tolist")
            else [int(index) for index in indices_raw]
        )
        values = (
            values_raw.tolist()
            if hasattr(values_raw, "tolist")
            else [float(value) for value in values_raw]
        )

        return models.SparseVector(indices=indices, values=values)
