from fastapi.testclient import TestClient

from src.entrypoints.api.main import create_app


def test_opinion_query_flow_returns_citations_with_metadata() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/queries",
        json={
            "query_text": "opinion de entidades sobre salud publica",
            "intent": "opinion",
            "language": "es",
            "max_sources": 4,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["confidence_level"] in {"low", "medium", "high"}
    assert len(payload["citations"]) >= 1
    assert "author" in payload["citations"][0]["metadata"]
