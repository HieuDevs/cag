import hashlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import pytest

from app.config import ModelTarget, Settings
from app.container import build_container
from app.llm.types import ChatRequest, LLMError, StreamEvent, Usage
from app.router import IntentRouter, Route
from app.store import MemoryStore


def fake_embedding(text: str, dim: int = 256) -> list[float]:
    """Embedding giả: bag-of-character-bigram băm vào `dim` chiều. Câu giống nhau cho cosine cao."""
    v = np.zeros(dim, dtype=np.float32)
    t = text.lower()
    for a, b in zip(t, t[1:] + " ", strict=True):
        v[int(hashlib.md5((a + b).encode()).hexdigest(), 16) % dim] += 1
    return (v / (np.linalg.norm(v) or 1)).tolist()


def sse_chunks(text_parts: list[str], *, usage: dict | None = None, model: str = "m", provider: str = "P",
               finish_reason: str = "stop") -> bytes:
    lines = [": OPENROUTER PROCESSING", ""]
    for part in text_parts:
        chunk = {"model": model, "provider": provider, "choices": [{"delta": {"content": part}}]}
        lines += [f"data: {json.dumps(chunk, ensure_ascii=False)}", ""]
    last = {"model": model, "provider": provider, "choices": [{"delta": {}, "finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 10}}
    lines += [f"data: {json.dumps(last)}", "", "data: [DONE]", ""]
    return "\n".join(lines).encode()


@dataclass
class FakeOpenRouter:
    """Handler cho httpx.MockTransport, giả lập /chat/completions (stream) và /embeddings."""

    # model -> (status, body) để giả lập lỗi; không có thì trả lời thành công.
    failures: dict[str, tuple[int, str]] = field(default_factory=dict)
    answer: list[str] = field(default_factory=lambda: ["Xin ", "chào"])
    usage: dict[str, Any] = field(default_factory=lambda: {
        "prompt_tokens": 8000, "completion_tokens": 50, "cost": 0.0004,
        "prompt_tokens_details": {"cached_tokens": 7600, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    })
    requests: list[dict[str, Any]] = field(default_factory=list)
    embedding_calls: int = 0
    # Câu trả lời của Jev (System One API): (intent, confidence, needs_context), hoặc (status, message) để giả lập lỗi.
    jev: tuple = ("grammar", 0.82, 0.12)
    jev_requests: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path.endswith("/systemone"):
            self.jev_requests.append({"url": str(request.url), "auth": request.headers.get("authorization"), **body})
            if isinstance(self.jev[0], int):
                return httpx.Response(self.jev[0], json={"error": {"message": self.jev[1]}})
            intent, conf, ctx = self.jev
            return httpx.Response(200, json={
                "model": body["model"], "usage": {"input_tokens": 120, "output_tokens": 3},
                "answers": {
                    "intent": {"type": "choice", "choice": intent, "confidence": conf,
                               "probabilities": {intent: conf}},
                    "needs_context": {"type": "noul", "noul": ctx},
                },
            })
        if request.url.path.endswith("/embeddings"):
            self.embedding_calls += 1
            data = [{"index": i, "embedding": fake_embedding(t)} for i, t in enumerate(body["input"])]
            return httpx.Response(200, json={"data": data})
        self.requests.append(body)
        if body["model"] in self.failures:
            status, msg = self.failures[body["model"]]
            return httpx.Response(status, json={"error": {"message": msg, "code": status}})
        return httpx.Response(200, content=sse_chunks(self.answer, usage=self.usage, model=body["model"]),
                              headers={"content-type": "text/event-stream"})


class ScriptedBackend:
    """ChatBackend giả cho test service: trả câu trả lời theo kịch bản, ghi lại request."""

    def __init__(self) -> None:
        self.calls: list[tuple[ModelTarget, ChatRequest]] = []
        self.answer = "Câu trả lời mẫu."
        self.fail_models: set[str] = set()
        self.finish_reason = "stop"
        self.usage = Usage(cached_input_tokens=7000, input_tokens=300, output_tokens=40, cost_usd=0.0005)

    async def stream_chat(self, target: ModelTarget, req: ChatRequest, *, sticky_key: str | None = None
                          ) -> AsyncIterator[StreamEvent]:
        self.calls.append((target, req))
        if target.model in self.fail_models:
            raise LLMError(f"{target.model} down", status=503)
        for word in self.answer.split(" "):
            yield StreamEvent("delta", text=word + " ")
        yield StreamEvent("done", usage=self.usage, model=target.model, provider="Fake",
                          finish_reason=self.finish_reason)


class FixedRouter(IntentRouter):
    """Router trả về route cố định (hoặc theo hàm), để test service không phụ thuộc Jev."""

    def __init__(self, route: Route | None = None):
        super().__init__([])
        self.next = route or Route("small", True, "lookup", intent="lookup", confidence=0.9, needs_context=0.1,
                                   backend="fixed")
        self.calls = 0

    async def route(self, question, embedding=None) -> Route:
        self.calls += 1
        return self.next


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        openrouter_api_key="test-key",
        tier_small=[ModelTarget(model="qwen/small", providers=["alibaba"]), ModelTarget(model="deepseek/small")],
        tier_large=[ModelTarget(model="qwen/large", providers=["alibaba"]), ModelTarget(model="deepseek/large")],
        router_backend="none",
        router_fallback="none",
        redis_url="memory://",
        usage_db_path=tmp_path / "usage.sqlite3",
        max_turns_per_session=3,
    )


@pytest.fixture
def fake_openrouter() -> FakeOpenRouter:
    return FakeOpenRouter()


@pytest.fixture
def backend() -> ScriptedBackend:
    return ScriptedBackend()


@pytest.fixture
def router() -> FixedRouter:
    return FixedRouter()


@pytest.fixture
def container(settings, backend, router, fake_openrouter):
    http = httpx.AsyncClient(base_url=settings.openrouter_base_url, transport=httpx.MockTransport(fake_openrouter))
    c = build_container(settings, http_client=http, chat_backend=backend, router=router, store=MemoryStore())
    yield c
    c.usage_log.close()


@pytest.fixture
def service(container):
    return container.service


async def collect(events) -> dict[str, Any]:
    """Gom luồng event thành dict: meta, text, done, error."""
    out: dict[str, Any] = {"text": "", "meta": None, "done": None, "error": None}
    async for ev in events:
        if ev["event"] == "delta":
            out["text"] += ev["data"]["text"]
        else:
            out[ev["event"]] = ev["data"]
    return out
