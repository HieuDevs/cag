import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client(settings, container):
    with TestClient(create_app(settings, container)) as c:
        yield c


def test_chat_streams_sse(client):
    r = client.post("/chat", json={"user_id": "u1", "message": "你好 là gì?", "level": "HSK1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-quota-used"] == "1"
    events = parse_sse(r.text)
    assert [e for e, _ in events][0] == "meta" and events[-1][0] == "done"
    assert "".join(d["text"] for e, d in events if e == "delta").strip() == "Câu trả lời mẫu."


def test_chat_non_stream(client):
    r = client.post("/chat", json={"user_id": "u1", "message": "你好", "stream": False})
    body = r.json()
    assert r.status_code == 200 and body["answer"].strip() == "Câu trả lời mẫu." and body["tier"] == "small"


def test_quota_exceeded_returns_429(client, settings):
    limit = settings.plans["free"].daily_requests
    for _ in range(limit):
        assert client.post("/chat", json={"user_id": "u9", "message": "hi", "stream": False}).status_code == 200
    r = client.post("/chat", json={"user_id": "u9", "message": "hi"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "quota_exceeded"


def test_validation(client):
    assert client.post("/chat", json={"user_id": "u1", "message": ""}).status_code == 422
    assert client.post("/chat", json={"user_id": "u1", "message": "x" * 4001}).status_code == 422


def test_feedback_and_retry(client, backend):
    first = client.post("/chat", json={"user_id": "u1", "message": "你好", "stream": False}).json()
    r = client.post("/feedback", json={"request_id": first["request_id"], "user_id": "u1", "rating": "satisfied"})
    assert r.json() == {"ok": True}
    backend.answer = "Bản tốt hơn."
    r = client.post("/feedback", json={"request_id": first["request_id"], "user_id": "u1",
                                       "rating": "unsatisfied", "retry": True, "stream": False})
    assert r.status_code == 200 and r.json()["tier"] == "large" and r.json()["answer"].strip() == "Bản tốt hơn."


def test_feedback_unknown_request(client):
    r = client.post("/feedback", json={"request_id": "nope", "user_id": "u1", "rating": "unsatisfied"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_retry_forbidden_returns_403(client):
    first = client.post("/chat", json={"user_id": "u1", "message": "你好", "plan": "basic", "stream": False}).json()
    r = client.post("/feedback", json={"request_id": first["request_id"], "user_id": "u1",
                                       "rating": "unsatisfied", "retry": True})
    assert r.status_code == 403


def test_api_key_required(settings, container):
    settings.api_key = "secret"
    with TestClient(create_app(settings, container)) as c:
        assert c.post("/chat", json={"user_id": "u1", "message": "hi"}).status_code == 401
        ok = c.post("/chat", json={"user_id": "u1", "message": "hi", "stream": False},
                    headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200, "health không cần key"


def test_health_and_stats(client):
    client.post("/chat", json={"user_id": "u1", "message": "你好", "stream": False})
    h = client.get("/health").json()
    assert h["ok"] and h["knowledge_version"] and h["tiers"]["small"][0] == "qwen/small"
    s = client.get("/stats").json()
    assert s["requests"] == 1 and s["cache_read_ratio"] == pytest.approx(7000 / 7300)
