"""Embedding qua OpenRouter (mặc định `baai/bge-m3`), dùng cho router dự phòng và cache gần giống."""

import logging
from collections import OrderedDict

import numpy as np

from app.llm.openrouter import OpenRouterClient
from app.llm.types import LLMError

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, client: OpenRouterClient, model: str, *, timeout: float, lru_size: int = 2048):
        self.client = client
        self.model = model
        self.timeout = timeout
        self._lru: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lru_size = lru_size

    async def embed_many(self, texts: list[str]) -> np.ndarray:
        """Trả về ma trận (n, d) đã chuẩn hóa L2. Ném `LLMError` khi lỗi."""
        vectors = await self.client.embed(self.model, texts, timeout=self.timeout)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    async def embed(self, text: str) -> np.ndarray | None:
        """Embedding một câu, trả `None` khi lỗi để request vẫn chạy tiếp mà không có embedding."""
        if (hit := self._lru.get(text)) is not None:
            self._lru.move_to_end(text)
            return hit
        try:
            vec = (await self.embed_many([text]))[0]
        except (LLMError, KeyError, ValueError) as e:
            log.warning("Embedding lỗi: %s", e)
            return None
        self._lru[text] = vec
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return vec


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
