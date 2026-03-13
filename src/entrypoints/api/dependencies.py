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
from src.infrastructure.adapters.outbound.qdrant_vector_store_adapter import (
    QdrantVectorStoreAdapter,
)


def get_query_opinion_use_case() -> QueryOpinionUseCase:
    retrieval_pipeline = RetrievalPipelineService(
        embedding_port=FastEmbedAdapter(),
        vector_store_port=QdrantVectorStoreAdapter(),
        lexical_search_port=BM25LexicalAdapter(),
        reranker_port=HuggingFaceRerankerAdapter(),
    )
    generation_service = TextGenerationService(
        primary_adapter=LiteLLMTextGenerationAdapter(),
        fallback_adapter=GeminiTextGenerationAdapter(),
    )
    return QueryOpinionUseCase(
        retrieval_pipeline=retrieval_pipeline,
        text_generation_service=generation_service,
    )
