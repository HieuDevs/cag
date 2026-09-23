import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from app.embeddings import Embedder
from app.llm.openrouter import OpenRouterClient
from app.router import (
    JEV_QUESTIONS,
    Classification,
    EmbeddingClassifier,
    IntentRouter,
    JevClassifier,
    build_router,
    needs_context_heuristic,
)


class StubClassifier:
    def __init__(self, name, result=None, error=None, threshold=0.6):
        self.name, self.result, self.error, self.threshold = name, result, error, threshold
        self.calls = 0

    async def classify(self, question, embedding):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def cls(intent, conf=0.9, ctx=0.1, backend="stub"):
    return Classification(intent, conf, ctx, backend)


@pytest.mark.parametrize("c,tier,cacheable,reason", [
    (cls("lookup"), "small", True, "lookup"),
    (cls("translate"), "small", True, "translate"),
    (cls("grammar"), "large", True, "grammar"),
    (cls("culture"), "large", True, "culture"),
    (cls("correction"), "large", False, "correction"),
    (cls("off_topic"), "canned", False, "off_topic"),
    (cls("lookup", conf=0.3), "large", False, "low_confidence"),
    (cls("off_topic", conf=0.3), "large", False, "low_confidence"),
    (cls("lookup", ctx=0.8), "small", False, "lookup"),
    (cls("weird"), "large", False, "unknown_intent"),
])
async def test_routing_table(c, tier, cacheable, reason):
    r = await IntentRouter([StubClassifier("s", c)]).route("q")
    assert (r.tier, r.cacheable, r.reason) == (tier, cacheable, reason)


async def test_low_confidence_uses_default_output_budget():
    r = await IntentRouter([StubClassifier("s", cls("lookup", conf=0.1))]).route("q")
    assert r.intent is None and r.max_tokens == 1000
    r = await IntentRouter([StubClassifier("s", cls("lookup"))]).route("q")
    assert r.max_tokens == 300


async def test_falls_back_to_next_classifier():
    jev = StubClassifier("jev", error=TimeoutError())
    emb = StubClassifier("embedding", cls("grammar", backend="embedding"), threshold=0.5)
    r = await IntentRouter([jev, emb]).route("q")
    assert r.reason == "grammar" and r.backend == "embedding" and jev.calls == 1


async def test_all_classifiers_fail_goes_large():
    r = await IntentRouter([StubClassifier("jev", error=RuntimeError())]).route("q")
    assert (r.tier, r.cacheable, r.reason) == ("large", False, "router_unavailable")
    r = await IntentRouter([]).route("q")
    assert r.tier == "large"


async def test_jev_classifier_parses_sdk_response(settings):
    class FakeJev:
        def __init__(self):
            self.kwargs = None

        async def system_one(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices={"intent": SimpleNamespace(choice="grammar", confidence=0.82)},
                nouls={"needs_context": SimpleNamespace(noul=0.12)},
            )

    fake = FakeJev()
    c = await JevClassifier(settings, client=fake).classify("了 dùng khi nào", None)
    assert c == Classification("grammar", 0.82, 0.12, "jev")
    assert fake.kwargs == {"state": {"question": "了 dùng khi nào"}, "questions": JEV_QUESTIONS}


async def test_jev_timeout(settings):
    class SlowJev:
        async def system_one(self, **kwargs):
            await asyncio.sleep(5)

    settings.jev_timeout_seconds = 0.01
    with pytest.raises(asyncio.TimeoutError):
        await JevClassifier(settings, client=SlowJev()).classify("q", None)


class KeywordEmbedder(Embedder):
    """Embedding giả theo từ khóa, để kiểm tra logic phân loại mà không cần model thật."""

    AXES = ["nghĩa", "dịch", "ngữ pháp", "sửa", "thành ngữ", "thời tiết"]

    def __init__(self):
        self.calls = 0

    def _vec(self, text):
        v = np.array([1.0 if k in text.lower() else 0.0 for k in self.AXES] + [0.05])
        return v / np.linalg.norm(v)

    async def embed_many(self, texts):
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    async def embed(self, text):
        return self._vec(text)


async def test_embedding_classifier(settings, tmp_path):
    ex = tmp_path / "ex.jsonl"
    rows = [("nghĩa là gì", "lookup"), ("dịch câu", "translate"), ("ngữ pháp", "grammar"),
            ("sửa câu", "correction"), ("thành ngữ", "culture"), ("thời tiết", "off_topic")]
    ex.write_text("\n".join(f'{{"text": "{t}", "intent": "{i}"}}' for t, i in rows))
    emb = KeywordEmbedder()
    clf = EmbeddingClassifier(settings, emb, examples_path=ex, top_k=1)
    c = await clf.classify("Giải thích ngữ pháp của 把", None)
    assert c.intent == "grammar" and c.confidence > 0.9
    # Nhận embedding đang tính dở (task) từ service
    task = asyncio.ensure_future(emb.embed("dịch giúp mình"))
    assert (await clf.classify("dịch giúp mình", task)).intent == "translate"
    assert emb.calls == 1, "Câu mẫu chỉ embed một lần"


def test_needs_context_heuristic():
    assert needs_context_heuristic("Cho thêm ví dụ với từ đó") > 0.5
    assert needs_context_heuristic("câu trên sai ở đâu") > 0.5
    assert needs_context_heuristic("你好 nghĩa là gì") < 0.5


def test_build_router_without_jev_key(settings):
    settings.router_backend, settings.router_fallback = "jev", "embedding"
    embedder = Embedder(OpenRouterClient(settings), "baai/bge-m3", timeout=1)
    r = build_router(settings, embedder)
    assert [c.name for c in r.classifiers] == ["embedding"]
    settings.typesafe_api_key = "ts-key"
    r = build_router(settings, embedder)
    assert [c.name for c in r.classifiers] == ["jev", "embedding"]
    settings.router_backend, settings.router_fallback = "none", "none"
    assert build_router(settings, embedder).classifiers == []
