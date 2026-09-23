"""Cache câu trả lời: khớp tuyệt đối (hash trên KV store) và khớp gần giống (embedding).

Key: `answer:{KNOWLEDGE_VERSION}:{trình độ}:{sha256(normalize(câu hỏi))}`. Đổi version là mọi key cũ
tự hết hiệu lực. Key có trình độ vì câu trả lời được viết theo trình độ người học.

Khớp gần giống giữ chỉ mục embedding trong bộ nhớ và chỉ lưu key; nội dung vẫn đọc từ KV store,
nên TTL và việc xóa cache vẫn áp dụng. Chuyển sang pgvector khi chạy nhiều worker.
"""

import hashlib
import json
import re
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from app.store import KVStore
from app.text import normalize_question

_HAN_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def han_signature(text: str) -> str:
    """Các chữ Hán trong câu hỏi, theo thứ tự xuất hiện.

    "了 dùng khi nào" và "过 dùng khi nào" có embedding rất gần nhau nhưng hỏi hai thứ khác nhau.
    Bắt buộc hai câu có cùng chữ Hán thì mới coi là gần giống.
    """
    return "".join(_HAN_RE.findall(text))


@dataclass
class CachedAnswer:
    text: str
    tier: str
    intent: str | None
    model: str
    created_at: float
    question: str = ""


class SemanticIndex:
    def __init__(self, max_entries: int):
        self.max_entries = max_entries
        self._keys: dict[str, deque[tuple[str, str]]] = {}
        self._vecs: dict[str, list[np.ndarray]] = {}

    def add(self, namespace: str, key: str, signature: str, vec: np.ndarray) -> None:
        keys = self._keys.setdefault(namespace, deque())
        vecs = self._vecs.setdefault(namespace, [])
        if any(k == key for k, _ in keys):
            return
        keys.append((key, signature))
        vecs.append(vec)
        if len(keys) > self.max_entries:
            keys.popleft()
            vecs.pop(0)

    def remove(self, namespace: str, key: str) -> None:
        keys = self._keys.get(namespace)
        if not keys:
            return
        for i, (k, _) in enumerate(keys):
            if k == key:
                del keys[i]
                del self._vecs[namespace][i]
                return

    def search(self, namespace: str, vec: np.ndarray, signature: str, threshold: float) -> tuple[str, float] | None:
        vecs = self._vecs.get(namespace)
        if not vecs:
            return None
        sims = np.stack(vecs) @ vec
        keys = self._keys[namespace]
        for i in np.argsort(-sims):
            if sims[i] < threshold:
                break
            key, sig = keys[i]
            if sig == signature:
                return key, float(sims[i])
        return None


class AnswerCache:
    def __init__(self, store: KVStore, *, version: str, ttl: int, semantic_threshold: float,
                 semantic_max_entries: int):
        self.store = store
        self.version = version
        self.ttl = ttl
        self.semantic_threshold = semantic_threshold
        self.index = SemanticIndex(semantic_max_entries)

    def _namespace(self, level: str | None) -> str:
        return f"{self.version}:{level or '-'}"

    def key(self, question: str, level: str | None) -> str:
        digest = hashlib.sha256(normalize_question(question).encode()).hexdigest()
        return f"answer:{self._namespace(level)}:{digest}"

    async def _load(self, key: str) -> CachedAnswer | None:
        raw = await self.store.get(key)
        return CachedAnswer(**json.loads(raw)) if raw else None

    async def get_exact(self, question: str, level: str | None) -> CachedAnswer | None:
        return await self._load(self.key(question, level))

    async def get_similar(self, question: str, level: str | None, vec: np.ndarray) -> tuple[CachedAnswer, float] | None:
        found = self.index.search(self._namespace(level), vec, han_signature(question), self.semantic_threshold)
        if not found:
            return None
        key, score = found
        answer = await self._load(key)
        if answer is None:  # hết TTL hoặc đã bị xóa
            self.index.remove(self._namespace(level), key)
            return None
        return answer, score

    async def put(self, question: str, level: str | None, answer: CachedAnswer, vec: np.ndarray | None) -> str:
        key = self.key(question, level)
        answer.question = question
        await self.store.set(key, json.dumps(asdict(answer), ensure_ascii=False, sort_keys=True), self.ttl)
        if vec is not None:
            self.index.add(self._namespace(level), key, han_signature(question), vec)
        return key

    async def invalidate(self, question: str, level: str | None) -> None:
        key = self.key(question, level)
        await self.store.delete(key)
        self.index.remove(self._namespace(level), key)


def now() -> float:
    return time.time()
