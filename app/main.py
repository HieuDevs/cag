"""FastAPI: `/chat` (stream SSE), `/feedback`, `/health`, `/stats`, `/admin/warmup`.

Service chạy sau backend chính của sản phẩm. Backend chính xác thực user rồi gửi `user_id` và `plan`
sang đây, kèm `Authorization: Bearer <API_KEY>`.
"""

import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.canned import QUOTA_EXCEEDED
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.service import ChatInput, Event, ServiceError

log = logging.getLogger(__name__)


class ChatBody(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    plan: str = "free"
    session_id: str | None = Field(default=None, max_length=64)
    level: str | None = Field(default=None, description="HSK1…HSK6 hoặc HSK7-9")
    stream: bool = True


class FeedbackBody(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1, max_length=128)
    rating: Literal["satisfied", "unsatisfied"]
    # `true`: hỏi lại bằng tầng `large` và stream câu trả lời mới.
    retry: bool = False
    stream: bool = True


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or build_container(settings)
        app.state.container = c
        if settings.warmup_on_startup:
            for r in await c.service.warmup():
                log.info("Warmup: %s", r)
        yield
        if container is None:
            await c.aclose()

    app = FastAPI(title="CAG Chinese Tutor", version="0.1.0", lifespan=lifespan)

    def get_container(request: Request) -> Container:
        return request.app.state.container

    def require_api_key(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_key:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(token, settings.api_key):
            raise HTTPException(401, "Sai API key")

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, e: ServiceError) -> JSONResponse:
        return JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=e.status)

    @app.post("/chat", dependencies=[Depends(require_api_key)])
    async def chat(body: ChatBody, c: Container = Depends(get_container)):
        quota = await c.service.consume_quota(body.user_id, body.plan)
        if not quota.allowed:
            return JSONResponse(
                {"error": {"code": "quota_exceeded", "message": QUOTA_EXCEEDED, "limit": quota.limit}},
                status_code=429,
            )
        events = c.service.chat(ChatInput(
            user_id=body.user_id, message=body.message, plan=body.plan,
            session_id=body.session_id, level=body.level,
        ))
        if body.stream:
            return _sse(events, headers={"X-Quota-Used": str(quota.used), "X-Quota-Limit": str(quota.limit)})
        return await _collect(events)

    @app.post("/feedback", dependencies=[Depends(require_api_key)])
    async def feedback(body: FeedbackBody, c: Container = Depends(get_container)):
        await c.service.feedback(body.request_id, body.user_id, body.rating)
        if body.rating == "unsatisfied" and body.retry:
            events = c.service.retry(body.request_id, body.user_id)
            # Kiểm tra quyền trước khi mở stream, để lỗi trả về đúng HTTP status.
            first = await anext(events)
            events = _prepend(first, events)
            return _sse(events) if body.stream else await _collect(events)
        return {"ok": True}

    @app.get("/health")
    async def health(c: Container = Depends(get_container)) -> dict[str, Any]:
        return {
            "ok": await c.store.ping(),
            "knowledge_version": c.service.prompt.version,
            "router": [clf.name for clf in c.router.classifiers],
            "tiers": {t: [m.model for m in ms] for t, ms in c.service.gateway.tiers.items()},
        }

    @app.get("/stats", dependencies=[Depends(require_api_key)])
    async def stats(hours: float = 24, c: Container = Depends(get_container)) -> dict[str, Any]:
        return await c.usage_log.stats(time.time() - hours * 3600)

    @app.post("/admin/warmup", dependencies=[Depends(require_api_key)])
    async def warmup(c: Container = Depends(get_container)) -> list[dict[str, Any]]:
        return await c.service.warmup()

    return app


async def _prepend(first: Event, rest: AsyncIterator[Event]) -> AsyncIterator[Event]:
    yield first
    async for ev in rest:
        yield ev


def _sse(events: AsyncIterator[Event], headers: dict[str, str] | None = None) -> StreamingResponse:
    async def body() -> AsyncIterator[str]:
        async for ev in events:
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **(headers or {})},
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
