"""Gateway: chọn model theo tầng, dự phòng sang model tiếp theo khi lỗi.

Chỉ chuyển model khi **chưa** stream token nào cho user. Đã stream một phần mà lỗi thì báo lỗi,
vì ghép hai câu trả lời từ hai model khác nhau sẽ ra nội dung lộn xộn.
"""

import logging
from collections.abc import AsyncIterator
from typing import Protocol

from app.config import ModelTarget, Settings
from app.llm.types import ChatRequest, LLMError, LLMResult, StreamEvent, Usage

log = logging.getLogger(__name__)


class ChatBackend(Protocol):
    def stream_chat(
        self, target: ModelTarget, req: ChatRequest, *, sticky_key: str | None = None
    ) -> AsyncIterator[StreamEvent]: ...


class LLMGateway:
    def __init__(self, settings: Settings, backend: ChatBackend):
        self.backend = backend
        self.tiers: dict[str, list[ModelTarget]] = {
            "small": settings.tier_small,
            "large": settings.tier_large,
        }

    def targets(self, tier: str) -> list[ModelTarget]:
        try:
            return self.tiers[tier]
        except KeyError:
            raise ValueError(f"Tầng không hợp lệ: {tier}") from None

    async def stream(
        self, tier: str, req: ChatRequest, *, sticky_key: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        failed: list[str] = []
        last_error: LLMError | None = None
        for target in self.targets(tier):
            started = False
            try:
                async for event in self.backend.stream_chat(target, req, sticky_key=sticky_key):
                    if event.type == "delta":
                        started = True
                    elif event.type == "done":
                        event.failed_attempts = failed
                    yield event
                return
            except LLMError as e:
                if started or not e.retryable:
                    raise
                log.warning("Model %s lỗi, chuyển sang model dự phòng: %s", target.model, e)
                failed.append(target.model)
                last_error = e
        raise LLMError(f"Mọi model của tầng {tier} đều lỗi: {last_error}", retryable=False)

    async def complete(self, tier: str, req: ChatRequest, *, sticky_key: str | None = None) -> LLMResult:
        parts: list[str] = []
        async for event in self.stream(tier, req, sticky_key=sticky_key):
            if event.type == "delta":
                parts.append(event.text)
            else:
                return LLMResult(
                    text="".join(parts),
                    model=event.model,
                    provider=event.provider,
                    usage=event.usage or Usage(),
                    finish_reason=event.finish_reason,
                    failed_attempts=event.failed_attempts,
                )
        raise LLMError("Stream kết thúc mà không có event done", retryable=False)
