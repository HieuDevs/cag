"""FastAPI: `/chat` (stream SSE), `/feedback`, `/health`, `/stats`, `/requests`, `/admin/warmup`.

Bản test: chưa có xác thực và quota. Cấu hình đọc trong `lifespan`, thiếu biến bắt buộc thì dừng khi start.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.container import Container, build_container
from app.service import ChatInput, Event, ServiceError

log = logging.getLogger(__name__)

# Bản test: chưa có xác thực, mọi request dùng chung một user.
TEST_USER_ID = "test-user"


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    level: str | None = Field(default=None, description="HSK1…HSK6 hoặc HSK7-9")
    stream: bool = True


class FeedbackBody(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal["satisfied", "unsatisfied"]
    # `true`: hỏi lại bằng tầng `large` và stream câu trả lời mới.
    retry: bool = False
    stream: bool = True


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or get_settings()
        c = container or build_container(s)
        app.state.container = c
        if s.warmup_on_startup:
            for r in await c.service.warmup():
                log.info("Warmup: %s", r)
        yield
        if container is None:
            await c.aclose()

    app = FastAPI(title="CAG Chinese Tutor", version="0.1.0", lifespan=lifespan)

    def get_container(request: Request) -> Container:
        return request.app.state.container

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, e: ServiceError) -> JSONResponse:
        return JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=e.status)

    @app.post("/chat")
    async def chat(body: ChatBody, c: Container = Depends(get_container)):
        events = c.service.chat(ChatInput(
            user_id=TEST_USER_ID, message=body.message, session_id=body.session_id, level=body.level,
        ))
        return _sse(events) if body.stream else await _collect(events)

    @app.post("/feedback")
    async def feedback(body: FeedbackBody, c: Container = Depends(get_container)):
        await c.service.feedback(body.request_id, TEST_USER_ID, body.rating)
        if body.rating == "unsatisfied" and body.retry:
            events = c.service.retry(body.request_id, TEST_USER_ID)
            # Lấy event đầu trước khi mở stream, để lỗi (404, 409) trả về đúng HTTP status.
            first = await anext(events)
            events = _prepend(first, events)
            return _sse(events) if body.stream else await _collect(events)
        return {"ok": True}

    @app.get("/health")
    async def health(c: Container = Depends(get_container)) -> dict[str, Any]:
        k = c.service.prompt.knowledge
        return {
            "ok": await c.store.ping(),
            "knowledge": {"version": k.version, "files": k.files, "estimated_tokens": k.estimated_tokens},
            "router": [clf.name for clf in c.router.classifiers],
            "tiers": {t: [m.model for m in ms] for t, ms in c.service.gateway.tiers.items()},
        }

    @app.get("/stats")
    async def stats(hours: float = 24, c: Container = Depends(get_container)) -> dict[str, Any]:
        return await c.usage_log.stats(time.time() - hours * 3600)

    @app.get("/requests")
    async def requests(
        limit: int = Query(20, ge=1, le=500),
        status: str | None = Query(None, description="ok | error | canned | aborted"),
        c: Container = Depends(get_container),
    ) -> list[dict[str, Any]]:
        """Log từng request, mới nhất trước."""
        return await c.usage_log.recent(limit, status)

    @app.post("/admin/warmup")
    async def warmup(c: Container = Depends(get_container)) -> list[dict[str, Any]]:
        return await c.service.warmup()

    return app


async def _prepend(first: Event, rest: AsyncIterator[Event]) -> AsyncIterator[Event]:
    yield first
    async for ev in rest:
        yield ev


def _sse(events: AsyncIterator[Event]) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        async for ev in events:
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _collect(events: AsyncIterator[Event]) -> JSONResponse:
    meta: dict[str, Any] = {}
    parts: list[str] = []
    done: dict[str, Any] = {}
    async for ev in events:
        if ev["event"] == "meta":
            meta = ev["data"]
        elif ev["event"] == "delta":
            parts.append(ev["data"]["text"])
        elif ev["event"] == "done":
            done = ev["data"]
        elif ev["event"] == "error":
            return JSONResponse({**meta, "error": ev["data"]}, status_code=503)
    return JSONResponse({**meta, "answer": "".join(parts), **done})


app = create_app()
