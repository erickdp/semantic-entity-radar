from typing import Protocol

from src.domain.entities.query import StructuredResponse


class QueryOpinionUseCasePort(Protocol):
    def execute(
        self,
        query_text: str,
        intent: str,
        language: str = "es",
        max_sources: int = 8,
    ) -> StructuredResponse: ...
