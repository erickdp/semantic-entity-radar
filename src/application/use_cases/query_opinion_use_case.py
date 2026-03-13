from uuid import uuid4

from src.application.services.citation_mapping_service import CitationMappingService
from src.application.services.retrieval_pipeline_service import RetrievalPipelineService
from src.application.services.text_generation_service import TextGenerationService
from src.domain.entities.query import ConfidenceLevel, QueryIntent, StructuredResponse
from src.domain.ports.inbound.query_use_cases import QueryOpinionUseCasePort


class QueryOpinionUseCase(QueryOpinionUseCasePort):
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipelineService,
        text_generation_service: TextGenerationService,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.text_generation_service = text_generation_service

    def execute(
        self,
        query_text: str,
        intent: str,
        language: str = "es",
        max_sources: int = 8,
    ) -> StructuredResponse:
        _ = language
        query_intent = QueryIntent(intent)
        ranked_evidence = self.retrieval_pipeline.retrieve(
            query_text=query_text, max_sources=max_sources
        )
        professional_text, key_points = self.text_generation_service.generate(
            query_text=query_text,
            ranked_evidence=ranked_evidence,
            intent=query_intent,
        )
        citations = CitationMappingService.from_chunks(ranked_evidence)
        confidence = ConfidenceLevel.HIGH if citations else ConfidenceLevel.LOW

        return StructuredResponse(
            query_id=str(uuid4()),
            response_type=query_intent,
            professional_text=professional_text,
            key_points=key_points,
            citations=citations,
            confidence_level=confidence,
        )
