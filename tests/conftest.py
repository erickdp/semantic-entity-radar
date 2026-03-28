import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.infrastructure.adapters.outbound.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)
from src.infrastructure.adapters.outbound.embedding_fastembed_adapter import (
    FastEmbedAdapter,
)
from src.infrastructure.adapters.outbound.bm25_lexical_adapter import BM25LexicalAdapter
from src.infrastructure.settings.app_settings import AppSettings


@pytest.fixture(scope="session")
def app_settings() -> AppSettings:
    return AppSettings()


@pytest.fixture(scope="session")
def qdrant_vector_store_adapter(app_settings: AppSettings) -> QdrantVectorStoreAdapter:
    if not app_settings.qdrant_url or not app_settings.qdrant_collection_name:
        pytest.skip(
            "Qdrant integration tests require QDRANT_URL and QDRANT_COLLECTION_NAME"
        )

    return QdrantVectorStoreAdapter(
        url=app_settings.qdrant_url,
        api_key=app_settings.qdrant_api_key,
        collection_name=app_settings.qdrant_collection_name,
        dense_vector_name=app_settings.qdrant_dense_vector_name,
        sparse_vector_name=app_settings.qdrant_sparse_vector_name,
        timeout_seconds=app_settings.qdrant_timeout_seconds,
        dense_embedding_adapter=FastEmbedAdapter(
            model_name=app_settings.fastembed_model
        ),
        sparse_embedding_adapter=BM25LexicalAdapter(
            model_name=app_settings.qdrant_sparse_model
        ),
    )
