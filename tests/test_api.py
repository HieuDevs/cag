import json

import pytest
from fastapi.testclient import TestClient

from app.main import TEST_USER_ID, create_app
from app.usage_log import hash_user


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
    r = client.post("/chat", json={"message": "你好 là gì?", "level": "HSK1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e for e, _ in events][0] == "meta" and events[-1][0] == "done"
    assert "".join(d["text"] for e, d in events if e == "delta").strip() == "Câu trả lời mẫu."


def test_chat_non_stream(client):
    r = client.post("/chat", json={"message": "你好", "stream": False})
    body = r.json()
    assert r.status_code == 200 and body["answer"].strip() == "Câu trả lời mẫu." and body["tier"] == "small"


def test_validation(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 4001}).status_code == 422


def test_feedback_and_retry(client, backend):
    first = client.post("/chat", json={"message": "你好", "stream": False}).json()
    r = client.post("/feedback", json={"request_id": first["request_id"], "rating": "satisfied"})
    assert r.json() == {"ok": True}
    backend.answer = "Bản tốt hơn."
    r = client.post("/feedback", json={"request_id": first["request_id"], "user_id": "u1",
                                       "rating": "unsatisfied", "retry": True, "stream": False})
    assert r.status_code == 200 and r.json()["tier"] == "large" and r.json()["answer"].strip() == "Bản tốt hơn."


def test_feedback_unknown_request(client):
    r = client.post("/feedback", json={"request_id": "nope", "rating": "unsatisfied"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_health_and_stats(client):
    client.post("/chat", json={"message": "你好", "stream": False})
    h = client.get("/health").json()
    assert h["ok"] and h["tiers"]["small"][0] == "qwen/small"
    assert len(h["knowledge"]["version"]) == 12 and h["knowledge"]["estimated_tokens"] > 1000
    s = client.get("/stats").json()
    assert s["requests"] == 1 and s["cache_read_ratio"] == pytest.approx(7000 / 7300)


def test_fixed_test_user_shares_session(client, container, backend):
    first = client.post("/chat", json={"message": "把 dùng thế nào", "stream": False}).json()
    client.post("/chat", json={"message": "cho thêm ví dụ", "session_id": first["session_id"], "stream": False})
    assert len(backend.calls[-1][1].messages) == 4, "Lượt 2 tiếp nối phiên của lượt 1"
    row = container.usage_log._conn.execute("SELECT user_hash FROM requests LIMIT 1").fetchone()
    assert row[0] == hash_user(TEST_USER_ID)


def test_requests_log(client, backend):
    client.post("/chat", json={"message": "你好", "stream": False})
    backend.fail_models = {"qwen/small", "deepseek/small"}
    client.post("/chat", json={"message": "谢谢", "stream": False})
    rows = client.get("/requests").json()
    assert [r["status"] for r in rows] == ["error", "ok"], "mới nhất trước"
    assert rows[1]["model"] == "qwen/small" and rows[1]["cached_input_tokens"] == 7000 and rows[1]["time"]
    assert [r["status"] for r in client.get("/requests?status=ok").json()] == ["ok"]
    assert len(client.get("/requests?limit=1").json()) == 1
    assert client.get("/requests?limit=0").status_code == 422


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "/debug/memory" in r.text


def test_debug_memory_and_clear(client):
    first = client.post("/chat", json={"message": "你好", "level": "HSK1", "stream": False}).json()
    m = client.get("/debug/memory").json()
    assert m["backend"] == "memory"
    assert m["counts"]["session"] == 1 and m["counts"]["req"] == 1 and m["counts"]["answer"] == 1
    session = next(i for i in m["items"] if i["kind"] == "session")
    assert session["key"].endswith(first["session_id"]) and len(session["value"]["messages"]) == 2
    assert session["ttl_s"] > 0

    assert m["counts"]["trace"] == 1
    assert client.post("/debug/memory/clear").json() == {"deleted": 4}
    m = client.get("/debug/memory").json()
    assert m["items"] == [] and m["semantic_index"] == {}


def test_debug_knowledge(client):
    k = client.get("/debug/knowledge").json()
    assert len(k["version"]) == 12 and k["files"][0] == "00_vai_tro.md" and len(k["text"]) > 1000


def test_trace_in_done_event_and_endpoint(client, backend):
    backend.fail_models = {"qwen/small"}
    first = client.post("/chat", json={"message": "你好", "stream": False}).json()
    steps = [(t["step"], t["status"]) for t in first["trace"]]
    assert steps == [
        ("session", "ok"), ("exact_cache", "miss"), ("router", "skip"), ("route", "ok"), ("cacheable", "ok"),
        ("embedding", "ok"), ("semantic_cache", "miss"), ("rag", "skip"), ("prompt", "ok"),
        ("llm", "fail"), ("llm", "fallback"), ("save", "ok"), ("answer_cache", "ok"),
    ]
    assert "qwen/small lỗi (503)" in first["trace"][9]["detail"]
    assert client.get(f"/debug/trace/{first['request_id']}").json() == first["trace"]
    assert client.get("/debug/trace/nope").status_code == 404

    again = client.post("/chat", json={"message": "你好", "stream": False}).json()
    assert [t["step"] for t in again["trace"]] == ["session", "exact_cache", "answer", "save"]
    assert again["trace"][1]["status"] == "hit"


def test_trace_on_error(client, backend):
    backend.fail_models = {"qwen/small", "deepseek/small"}
    r = client.post("/chat", json={"message": "你好", "stream": False})
    trace = r.json()["error"]["trace"]
    assert [(t["step"], t["status"]) for t in trace[-3:]] == [("llm", "fail")] * 3
