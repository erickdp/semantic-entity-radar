from typing import Annotated

from fastapi import APIRouter, Depends

from src.application.dto.query_dto import (
    CitationDTO,
    QueryRequestDTO,
    StructuredResponseDTO,
)
from src.domain.ports.inbound.query_use_cases import QueryOpinionUseCasePort
from src.entrypoints.api.dependencies import get_query_opinion_use_case

router = APIRouter(prefix="/v1")


@router.post("/queries", response_model=StructuredResponseDTO)
def create_query_response(
    payload: QueryRequestDTO,
    use_case: Annotated[QueryOpinionUseCasePort, Depends(get_query_opinion_use_case)],
) -> StructuredResponseDTO:
    result = use_case.execute(
        query_text=payload.query_text,
        intent=payload.intent,
        language=payload.language,
        max_sources=payload.max_sources,
    )

    return StructuredResponseDTO(
        query_id=result.query_id,
        response_type=result.response_type.value,
        professional_text=result.professional_text,
        key_points=result.key_points,
        confidence_level=result.confidence_level.value,
        generated_at=result.generated_at,
        citations=[
            CitationDTO(
                source_url=citation.source_url,
                source_network=citation.source_network,
                document_id=citation.document_id,
                chunk_id=citation.chunk_id,
                relevance_score=citation.relevance_score,
                metadata=citation.metadata,
            )
            for citation in result.citations
        ],
    )
