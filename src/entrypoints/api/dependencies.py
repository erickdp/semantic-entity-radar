from src.application.services.retrieval_pipeline_service import RetrievalPipelineService
from src.application.services.text_generation_service import TextGenerationService
from src.application.use_cases.query_opinion_use_case import QueryOpinionUseCase
from src.infrastructure.adapters.outbound.bm25_lexical_adapter import BM25LexicalAdapter
from src.infrastructure.adapters.outbound.embedding_fastembed_adapter import (
    FastEmbedAdapter,
)
from src.infrastructure.adapters.outbound.gemini_text_generation_adapter import (
    GeminiTextGenerationAdapter,
)
from src.infrastructure.adapters.outbound.hf_reranker_adapter import (
    HuggingFaceRerankerAdapter,
)
from src.infrastructure.adapters.outbound.litellm_text_generation_adapter import (
    LiteLLMTextGenerationAdapter,
)
from src.infrastructure.adapters.outbound.pinecone_reranker_adapter import (
    PineconeRerankerAdapter,
)
from src.infrastructure.adapters.outbound.pinecone_vector_store_adapter import (
    PineconeHybridVectorStoreAdapter,
)
from src.infrastructure.adapters.outbound.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)
from src.infrastructure.settings.app_settings import AppSettings


def get_query_opinion_use_case() -> QueryOpinionUseCase:
    settings = AppSettings()
    retrieval_pipeline = RetrievalPipelineService(
        vector_store_port=_build_vector_store(settings),
        reranker_port=_build_reranker(settings),
    )
    generation_service = TextGenerationService(
        primary_adapter=LiteLLMTextGenerationAdapter(),
        fallback_adapter=GeminiTextGenerationAdapter(),
    )
    return QueryOpinionUseCase(
        retrieval_pipeline=retrieval_pipeline,
        text_generation_service=generation_service,
    )


def _build_vector_store(settings: AppSettings):
    provider = settings.semantic_provider.strip().lower()
    if provider == "pinecone":
        return PineconeHybridVectorStoreAdapter(
            api_key=settings.pinecone_api_key,
            dense_index_name=settings.pinecone_dense_index_name,
            sparse_index_name=settings.pinecone_sparse_index_name,
            dense_index_host=settings.pinecone_dense_index_host,
            sparse_index_host=settings.pinecone_sparse_index_host,
            namespace=settings.pinecone_namespace,
            text_field=settings.pinecone_text_field,
            dense_model=settings.pinecone_dense_model,
            sparse_model=settings.pinecone_sparse_model,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
            search_multiplier=settings.pinecone_search_multiplier,
            auto_create_indexes=settings.pinecone_auto_create_indexes,
        )

    dense_embedding_adapter = FastEmbedAdapter(model_name=settings.fastembed_model)
    sparse_embedding_adapter = BM25LexicalAdapter(
        model_name=settings.qdrant_sparse_model
    )
    return QdrantVectorStoreAdapter(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
        dense_vector_name=settings.qdrant_dense_vector_name,
        sparse_vector_name=settings.qdrant_sparse_vector_name,
        timeout_seconds=settings.qdrant_timeout_seconds,
        dense_embedding_adapter=dense_embedding_adapter,
        sparse_embedding_adapter=sparse_embedding_adapter,
    )


def _build_reranker(settings: AppSettings):
    provider = settings.semantic_provider.strip().lower()
    if provider == "pinecone":
        return PineconeRerankerAdapter(
            api_key=settings.pinecone_api_key,
            model_name=settings.pinecone_rerank_model,
            rank_field=settings.pinecone_text_field,
        )

    return HuggingFaceRerankerAdapter()
