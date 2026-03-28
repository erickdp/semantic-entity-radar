from src.application.services.retrieval_pipeline_service import RetrievalPipelineService
from src.application.services.text_generation_service import TextGenerationService
from src.application.use_cases.query_opinion_use_case import QueryOpinionUseCase
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


def test_query_opinion_use_case_returns_grounded_response(
    qdrant_vector_store_adapter: QdrantVectorStoreAdapter,
) -> None:
    retrieval = RetrievalPipelineService(
        vector_store_port=qdrant_vector_store_adapter,
        reranker_port=HuggingFaceRerankerAdapter(),
    )
    generation = TextGenerationService(
        primary_adapter=LiteLLMTextGenerationAdapter(),
        fallback_adapter=GeminiTextGenerationAdapter(),
    )
    use_case = QueryOpinionUseCase(
        retrieval_pipeline=retrieval, text_generation_service=generation
    )

    result = use_case.execute(
        query_text="opinion sobre educacion",
        intent="opinion",
        language="es",
        max_sources=3,
    )

    assert result.response_type.value == "opinion"
    assert result.professional_text
    assert len(result.citations) > 0
    assert result.key_points
