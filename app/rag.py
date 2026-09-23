"""RAG cho phần tra cứu lớn (từ vựng HSK, giáo trình). Giai đoạn 6 trong roadmap.

Đoạn tài liệu lấy được luôn đặt ở lượt user cuối, **sau** phần được cache.
"""

from typing import Protocol


class Retriever(Protocol):
    async def retrieve(self, question: str, *, intent: str | None, level: str | None) -> list[str]: ...


class NullRetriever:
    async def retrieve(self, question: str, *, intent: str | None, level: str | None) -> list[str]:
        return []
