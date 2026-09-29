"""Định tuyến theo intent (xem docs/routing.md).

Router chính là Jev, gọi qua OpenRouter (System One API). Khi Jev lỗi hoặc timeout, dùng bộ phân loại
embedding.
Cả hai cùng lỗi thì gán `large`: tốn thêm chút tiền vẫn tốt hơn trả lời sai học thuật.

Câu nào cũng được trả lời: câu ngoài tiếng Trung (`off_topic`) đi tầng small, trả lời ngắn và kèm vài từ
tiếng Trung liên quan (knowledge/00_vai_tro.md).
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.config import Settings
from app.embeddings import Embedder
from app.llm.openrouter import OpenRouterClient
from app.llm.types import LLMError

log = logging.getLogger(__name__)

# Embedding đã có, hoặc task đang tính embedding, hoặc chưa có.
EmbeddingInput = np.ndarray | asyncio.Future | None

EXAMPLES_PATH = Path(__file__).parent / "resources" / "router_examples.jsonl"

# Viết bằng tiếng Anh vì đây là ngôn ngữ Jev làm tốt nhất. Câu hỏi của user giữ nguyên ngôn ngữ gốc.
INTENT_CRITERIA: dict[str, str] = {
    "lookup": "Meaning, pinyin, tones, stroke order or Sino-Vietnamese reading of one word or short phrase",
    "translate": "Translate one short sentence between Vietnamese and Chinese",
    "grammar": "Explain or compare grammar points, particles, measure words or sentence patterns",
    "correction": "Check or correct a sentence or paragraph the learner wrote themselves",
    "culture": "Chinese history, culture, idiom origins or classical Chinese",
    "study": (
        "How to study Chinese: learning roadmaps, study methods, HSK exam preparation, learning resources, "
        "or useful phrases for a situation or topic such as asking directions or ordering food"
    ),
    "off_topic": "Not related to the Chinese language or to learning Chinese at all",
}
INTENTS = tuple(INTENT_CRITERIA)

# intent -> (tầng, có được cache câu trả lời không)
ROUTES: dict[str, tuple[str, bool]] = {
    "lookup": ("small", True),
    "translate": ("small", True),
    "grammar": ("large", True),
    "culture": ("large", True),
    "correction": ("large", False),
    "study": ("small", True),
    # Không cache: hay cần dữ liệu thời gian thực (thời tiết, giá cả), để lâu là sai.
    "off_topic": ("small", False),
}

# Giới hạn output theo intent (chưa tính phần thinking).
MAX_OUTPUT_TOKENS: dict[str | None, int] = {
    "lookup": 600,
    "translate": 600,
    "grammar": 1000,
    "culture": 1000,
    "correction": 1200,
    "study": 800,
    "off_topic": 400,
    None: 1000,
}

JEV_QUESTIONS: dict[str, dict[str, Any]] = {
    "intent": {
        "type": "choice",
        "instructions": "What is the learner asking the Chinese-learning assistant to do?",
        "criteria": INTENT_CRITERIA,
    },
    "needs_context": {
        "type": "noul",
        "instructions": (
            "Does the question refer to something earlier in the conversation, "
            "such as 'that word' or 'the sentence above'?"
        ),
    },
}


@dataclass
class Classification:
    intent: str
    confidence: float
    needs_context: float
    backend: str


@dataclass
class Route:
    tier: str  # "small" | "large"
    cacheable: bool
    reason: str  # intent, hoặc lý do rơi vào `large`
    intent: str | None = None
    confidence: float | None = None
    needs_context: float | None = None
    backend: str = "none"
    # Từng router đã thử: backend, ngưỡng, thời gian, kết quả hoặc lỗi (hiển thị trong trace).
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def max_tokens(self) -> int:
        return MAX_OUTPUT_TOKENS.get(self.intent, MAX_OUTPUT_TOKENS[None])


class Classifier(Protocol):
    name: str
    threshold: float

    async def classify(self, question: str, embedding: EmbeddingInput) -> Classification: ...


class JevClassifier:
    name = "jev"

    def __init__(self, settings: Settings, client: OpenRouterClient):
        self.threshold = settings.jev_confidence_threshold
        self.timeout = settings.jev_timeout_seconds
        self.model = settings.jev_model
        self.client = client

    async def classify(self, question: str, embedding: EmbeddingInput) -> Classification:
        # Không retry: timeout 800ms là cho cả lần gọi, lỗi thì chuyển sang dự phòng ngay.
        r = await asyncio.wait_for(
            self.client.system_one(self.model, {"question": question}, JEV_QUESTIONS, timeout=self.timeout),
            timeout=self.timeout,
        )
        answers = r["answers"]
        intent = answers["intent"]
        return Classification(
            intent=intent["choice"],
            confidence=float(intent["confidence"]),
            needs_context=float(answers["needs_context"]["noul"]),
            backend=self.name,
        )


# Từ ngữ nhắc tới lượt trước, có cả dạng không dấu. Chỉ là heuristic cho router dự phòng.
_CONTEXT_RE = re.compile(
    r"(từ (đó|này|kia|trên|vừa rồi)|tu (do|nay|kia|tren)|câu (đó|này|trên|kia|vừa rồi|lúc nãy)|"
    r"cau (do|nay|tren|kia)|ở trên|o tren|vừa rồi|vừa nãy|lúc nãy|như trên|cái (đó|này)|"
    r"\bnó\b|thêm ví dụ|them vi du|ví dụ khác|giải thích thêm|giai thich them|"
    r"còn .{1,20} thì sao|tiếp đi|tiếp tục|chữ (đó|này)|这个词|那个词|上面|刚才)",
    re.IGNORECASE,
)


def needs_context_heuristic(question: str) -> float:
    return 0.9 if _CONTEXT_RE.search(question) else 0.1


class EmbeddingClassifier:
    """So câu hỏi với các câu mẫu có nhãn (`resources/router_examples.jsonl`).

    Điểm của mỗi intent là trung bình top-k cosine với các câu mẫu của intent đó; confidence là
    softmax của các điểm. Thay bằng logistic regression khi có đủ dữ liệu thật (docs/routing.md mục 6).
    """

    name = "embedding"

    def __init__(self, settings: Settings, embedder: Embedder, *, examples_path: Path = EXAMPLES_PATH,
                 top_k: int = 3, temperature: float = 0.02):
        self.threshold = settings.embedding_confidence_threshold
        self.embedder = embedder
        self.examples_path = examples_path
        self.top_k = top_k
        self.temperature = temperature
        self._matrix: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._lock = asyncio.Lock()

    async def _ensure_fitted(self) -> None:
        if self._matrix is not None:
            return
        async with self._lock:
            if self._matrix is not None:
                return
            rows = [json.loads(line) for line in self.examples_path.read_text("utf-8").splitlines() if line.strip()]
            self._matrix = await self.embedder.embed_many([r["text"] for r in rows])
            self._labels = np.array([r["intent"] for r in rows])
            log.info("Router embedding đã nạp %d câu mẫu", len(rows))

    async def classify(self, question: str, embedding: EmbeddingInput) -> Classification:
        await self._ensure_fitted()
        if isinstance(embedding, asyncio.Future):  # embedding đang được tính song song với Jev
            embedding = await embedding
        if embedding is None:
            embedding = await self.embedder.embed(question)
            if embedding is None:
                raise LLMError("Không lấy được embedding cho câu hỏi")
        assert self._matrix is not None and self._labels is not None
        sims = self._matrix @ embedding
        scores = []
        for intent in INTENTS:
            s = np.sort(sims[self._labels == intent])[::-1][: self.top_k]
            scores.append(float(s.mean()) if s.size else -1.0)
        logits = np.array(scores) / self.temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(probs.argmax())
        return Classification(
            intent=INTENTS[best],
            confidence=float(probs[best]),
            needs_context=needs_context_heuristic(question),
            backend=self.name,
        )


class IntentRouter:
    def __init__(self, classifiers: list[Classifier], *, needs_context_threshold: float = 0.5):
        self.classifiers = classifiers
        self.needs_context_threshold = needs_context_threshold

    async def route(self, question: str, embedding: EmbeddingInput = None) -> Route:
        attempts: list[dict[str, Any]] = []
        for clf in self.classifiers:
            started = time.monotonic()
            attempt: dict[str, Any] = {"backend": clf.name, "threshold": clf.threshold}
            attempts.append(attempt)
            try:
                c = await clf.classify(question, embedding)
            except Exception as e:  # noqa: BLE001 - router lỗi kiểu gì cũng chuyển sang dự phòng
                log.warning("Router %s lỗi: %r", clf.name, e)
                attempt.update(ok=False, ms=_ms(started), error=f"{type(e).__name__}: {e}"[:300])
                continue
            attempt.update(ok=True, ms=_ms(started), intent=c.intent, confidence=round(c.confidence, 4),
                           needs_context=round(c.needs_context, 4))
            route = self.decide(c, clf.threshold)
            route.attempts = attempts
            return route
        return Route("large", False, "router_unavailable", attempts=attempts)

    def decide(self, c: Classification, threshold: float) -> Route:
        common = dict(intent=c.intent, confidence=c.confidence, needs_context=c.needs_context, backend=c.backend)
        if c.intent not in INTENT_CRITERIA:
            return Route("large", False, "unknown_intent", **{**common, "intent": None})
        if c.confidence < threshold:
            # Không truyền intent xuống: giới hạn output và gợi ý dạng câu hỏi theo mặc định.
            return Route("large", False, "low_confidence", **{**common, "intent": None})
        tier, cacheable = ROUTES[c.intent]
        cacheable = cacheable and c.needs_context < self.needs_context_threshold
        return Route(tier, cacheable, c.intent, **common)

    @property
    def uses_embeddings(self) -> bool:
        return any(isinstance(c, EmbeddingClassifier) for c in self.classifiers)


def build_router(settings: Settings, client: OpenRouterClient, embedder: Embedder | None) -> IntentRouter:
    classifiers: list[Classifier] = []
    for backend in (settings.router_backend, settings.router_fallback):
        if backend == "jev" and not any(isinstance(c, JevClassifier) for c in classifiers):
            classifiers.append(JevClassifier(settings, client))
        elif backend == "embedding" and embedder and not any(
            isinstance(c, EmbeddingClassifier) for c in classifiers
        ):
            classifiers.append(EmbeddingClassifier(settings, embedder))
    if not classifiers:
        log.warning("Không có router nào, mọi câu hỏi sẽ vào tầng large")
    return IntentRouter(classifiers, needs_context_threshold=settings.needs_context_threshold)


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 1)
