from fastapi.testclient import TestClient

from agentic_rag.api import app

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["index_ready"] is True


def test_query_returns_grounded_answer():
    r = client.post("/query", json={"question": "What is retrieval-augmented generation?"})
    assert r.status_code == 200
    body = r.json()
    assert body["question"]
    assert body["answer"]
    assert isinstance(body["citations"], list)
    assert "retrieve" in body["path"]


def test_query_validation_rejects_empty():
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 422
