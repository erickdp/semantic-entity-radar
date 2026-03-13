from fastapi.testclient import TestClient

from src.entrypoints.api.main import app


def test_post_queries_opinion_returns_structured_payload() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/queries",
        json={
            "query_text": "opinion sobre energia renovable",
            "intent": "opinion",
            "language": "es",
            "max_sources": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "opinion"
    assert payload["professional_text"]
    assert payload["citations"]
    assert payload["citations"][0]["metadata"]
