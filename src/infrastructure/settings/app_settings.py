from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    app_env: str = "dev"
    semantic_provider: str = "pinecone"
    fastembed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    qdrant_sparse_model: str = "Qdrant/bm25"
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = ""
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    qdrant_timeout_seconds: float = 10.0
    pinecone_api_key: str = ""
    pinecone_dense_index_name: str = ""
    pinecone_sparse_index_name: str = ""
    pinecone_dense_index_host: str = ""
    pinecone_sparse_index_host: str = ""
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_text_field: str = "chunk_text"
    pinecone_dense_model: str = "multilingual-e5-large"
    pinecone_sparse_model: str = "pinecone-sparse-english-v0"
    pinecone_rerank_model: str = "bge-reranker-v2-m3"
    pinecone_search_multiplier: int = 3
    pinecone_auto_create_indexes: bool = False
    litellm_api_key: str = ""
    google_api_key: str = ""
    twikit_username: str = ""
    twikit_password: str = ""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
