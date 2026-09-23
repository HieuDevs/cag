"""Luồng xử lý một request (docs/architecture.md mục 3).

    quota → cache khớp tuyệt đối → router → cache gần giống → RAG → ghép prompt
    → gọi LLM (stream) → log → lưu cache → feedback

`chat()` và `retry()` trả về luồng event dạng `{"event": ..., "data": {...}}`, API chuyển thành SSE:
- `meta`: request_id, session_id, tầng, intent, nguồn câu trả lời
- `delta`: một đoạn câu trả lời
- `done`: usage, chi phí, độ trễ
- `error`: lỗi, kèm `code`
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from app.answer_cache import AnswerCache, CachedAnswer
from app.canned import OFF_TOPIC_ANSWER, SUMMARY_INSTRUCTION
from app.config import Settings
from app.embeddings import Embedder
from app.llm.gateway import LLMGateway
from app.llm.types import ChatRequest, LLMError, Usage
from app.prompt_builder import PromptBuilder, normalize_level
from app.quota import Quota, QuotaResult
from app.rag import NullRetriever, Retriever
from app.router import ROUTES, IntentRouter, Route
from app.sessions import Session, SessionStore
from app.store import KVStore
from app.usage_log import RequestLog, UsageLog, hash_user

log = logging.getLogger(__name__)

Event = dict[str, Any]
RECORD_TTL = 24 * 3600


@dataclass
class ChatInput:
    user_id: str
    message: str
    plan: str = "free"
    session_id: str | None = None
    level: str | None = None


@dataclass
class RequestRecord:
    """Lưu lại để xử lý feedback "chưa hài lòng" cho request này."""

    user_id: str
    plan: str
    session_id: str
    question: str
    level: str | None
    intent: str | None
    tier: str
    cacheable: bool
    turn_index: int | None  # vị trí lượt user trong lịch sử phiên; None nếu không lưu vào lịch sử
    is_retry: bool = False


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _event(name: str, **data: Any) -> Event:
    return {"event": name, "data": data}


def _usage_dict(u: Usage) -> dict[str, Any]:
    return {**asdict(u), "cache_read_ratio": round(u.cache_read_ratio, 4)}


class ChatService:
    def __init__(
        self,
        settings: Settings,
        *,
        prompt: PromptBuilder,
        gateway: LLMGateway,
        router: IntentRouter,
        answer_cache: AnswerCache,
        sessions: SessionStore,
        quota: Quota,
        usage_log: UsageLog,
        store: KVStore,
        embedder: Embedder | None = None,
        retriever: Retriever | None = None,
    ):
        self.settings = settings
        self.prompt = prompt
        self.gateway = gateway
        self.router = router
        self.answer_cache = answer_cache
        self.sessions = sessions
        self.quota = quota
        self.usage_log = usage_log
        self.store = store
        self.embedder = embedder
        self.retriever = retriever or NullRetriever()
        # Khóa sticky routing trên OpenRouter: mọi request dùng chung phần đầu prompt nên đi cùng một
        # nhà cung cấp, cache luôn nóng.
        self.sticky_key = f"cag-{prompt.version}"

    # ------------------------------------------------------------------ quota

    async def consume_quota(self, user_id: str, plan: str) -> QuotaResult:
        return await self.quota.consume(user_id, plan)

    # ------------------------------------------------------------------ chat

    async def chat(self, inp: ChatInput) -> AsyncIterator[Event]:
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        level = normalize_level(inp.level)
        question = inp.message.strip()
        row = RequestLog(
            request_id=request_id,
            user_hash=hash_user(inp.user_id),
            plan=inp.plan,
            knowledge_version=self.prompt.version,
        )

        session = await self._load_session(inp)
        row.session_id = session.id
        # Có tóm tắt phiên trước thì câu hỏi có thể phụ thuộc ngữ cảnh, không coi là câu đầu phiên.
        is_first = session.is_first_turn and not session.summary

        # 1. Cache khớp tuyệt đối: chỉ cho câu hỏi đầu phiên, chạy trước router nên không tốn tiền Jev.
        if is_first and (hit := await self.answer_cache.get_exact(question, level)):
            async for ev in self._serve_cached(inp, session, question, level, hit, "exact", row, started):
                yield ev
            return

        # 2. Router, chạy song song với embedding (dùng cho router dự phòng và cache gần giống).
        emb_task = self._start_embedding(question)
        route = await self.router.route(question, emb_task)
        row.intent, row.route_reason = route.intent, route.reason
        row.router_backend, row.router_confidence = route.backend, route.confidence
        row.needs_context = route.needs_context

        if route.tier == "canned":
            if emb_task:
                emb_task.cancel()
            row.tier, row.status = "canned", "canned"
            row.latency_ms = _ms(started)
            await self._save_record(request_id, inp, session, question, level, route, "canned", False, None)
            yield _event("meta", request_id=request_id, session_id=session.id, tier="canned",
                         intent=route.intent, reason=route.reason, answer_cache=None)
            yield _event("delta", text=OFF_TOPIC_ANSWER)
            yield _event("done", request_id=request_id, model=None, usage=None, cost_usd=0.0,
                         latency_ms=row.latency_ms)
            await self.usage_log.write(row)
            return

        tier = self._allowed_tier(route.tier, inp.plan)
        cacheable = self._cacheable(route, is_first)
        embedding = await emb_task if emb_task else None

        # 3. Cache gần giống: sau router, chỉ khi được phép cache.
        if cacheable and embedding is not None and self.settings.semantic_cache_enabled:
            if found := await self.answer_cache.get_similar(question, level, embedding):
                hit, score = found
                log.info("Trúng cache gần giống (%.3f): %r ~ %r", score, question, hit.question)
                async for ev in self._serve_cached(inp, session, question, level, hit, "semantic", row,
                                                   started, route=route):
                    yield ev
                return

        # 4. RAG, 5. ghép prompt, 6. gọi LLM.
        references = await self.retriever.retrieve(question, intent=route.intent, level=level)
        user_turn = self.prompt.compose_user_turn(
            question, level=level, intent=route.intent, references=references, summary=session.summary
        )
        history = list(session.messages)
        async for ev in self._generate(
            request_id=request_id, inp=inp, session=session, question=question, level=level, route=route,
            tier=tier, cacheable=cacheable, embedding=embedding, history=history, user_turn=user_turn,
            turn_index=len(history), row=row, started=started,
        ):
            yield ev

    # ------------------------------------------------------------------ feedback

    async def feedback(self, request_id: str, user_id: str, rating: str) -> RequestRecord:
        record = await self._get_record(request_id, user_id)
        await self.usage_log.set_feedback(request_id, rating)
        if rating == "unsatisfied" and record.cacheable:
            # Không trả câu trả lời bị chê cho người khác nữa.
            await self.answer_cache.invalidate(record.question, record.level)
        return record

    async def retry(self, request_id: str, user_id: str) -> AsyncIterator[Event]:
        """Hỏi lại bằng tầng `large` sau khi user bấm "chưa hài lòng". Không tính vào quota."""
        record = await self._get_record(request_id, user_id)
        if not self.quota.limits(record.plan).allow_large:
            raise ServiceError("plan_forbidden", "Gói hiện tại không dùng được model lớn", 403)
        # Hỏi lại không tính quota, nên mỗi request chỉ được hỏi lại một lần và không hỏi lại bản hỏi lại.
        if record.is_retry or await self.store.incr(f"retried:{request_id}", ttl=RECORD_TTL) > 1:
            raise ServiceError("already_retried", "Câu trả lời này đã được hỏi lại rồi", 409)
        started = time.monotonic()
        new_id = uuid.uuid4().hex
        session = await self.sessions.get(user_id, record.session_id) or self.sessions.new(user_id)
        idx = record.turn_index
        if idx is not None and idx < len(session.messages) and session.messages[idx]["role"] == "user":
            # Dùng lại đúng lượt user đã gửi để giữ cache lịch sử.
            history, user_turn = session.messages[:idx], session.messages[idx]["content"]
        else:
            history, idx = [], None
            user_turn = self.prompt.compose_user_turn(record.question, level=record.level, intent=record.intent)
        row = RequestLog(
            request_id=new_id, user_hash=hash_user(user_id), kind="retry", plan=record.plan,
            session_id=session.id, knowledge_version=self.prompt.version, intent=record.intent,
            route_reason="feedback_retry",
        )
        route = Route("large", record.cacheable, "feedback_retry", intent=record.intent)
        inp = ChatInput(user_id=user_id, message=record.question, plan=record.plan, session_id=session.id,
                        level=record.level)
        async for ev in self._generate(
            request_id=new_id, inp=inp, session=session, question=record.question, level=record.level,
            route=route, tier="large", cacheable=record.cacheable, embedding=None, history=history,
            user_turn=user_turn, turn_index=idx, row=row, started=started, replace_turn=idx is not None,
            refund_on_error=False, is_retry=True,
        ):
            yield ev

    # ------------------------------------------------------------------ internals

    async def _generate(
        self, *, request_id: str, inp: ChatInput, session: Session, question: str, level: str | None,
        route: Route, tier: str, cacheable: bool, embedding: np.ndarray | None,
        history: list[dict[str, str]], user_turn: str, turn_index: int | None, row: RequestLog,
        started: float, replace_turn: bool = False, refund_on_error: bool = True, is_retry: bool = False,
    ) -> AsyncIterator[Event]:
        row.tier = tier
        reasoning = tier == "large" and route.intent in self.settings.reasoning_intents
        req = ChatRequest(
            messages=self.prompt.build(history, user_turn),
            max_tokens=route.max_tokens,
            reasoning=reasoning,
            reasoning_max_tokens=self.settings.reasoning_max_tokens,
        )
        yield _event("meta", request_id=request_id, session_id=session.id, tier=tier, intent=route.intent,
                     reason=route.reason, answer_cache=None)

        parts: list[str] = []
        ttft: float | None = None
        try:
            async for ev in self.gateway.stream(tier, req, sticky_key=self.sticky_key):
                if ev.type == "delta":
                    if ttft is None:
                        ttft = _ms(started)
                    parts.append(ev.text)
                    yield _event("delta", text=ev.text)
                    continue
                usage = ev.usage or Usage()
                row.model, row.provider = ev.model, ev.provider
                row.failed_models = ",".join(ev.failed_attempts) or None
                row.cached_input_tokens, row.input_tokens = usage.cached_input_tokens, usage.input_tokens
                row.cache_write_tokens, row.output_tokens = usage.cache_write_tokens, usage.output_tokens
                row.reasoning_tokens, row.cost_usd = usage.reasoning_tokens, usage.cost_usd
                row.ttft_ms, row.latency_ms = ttft, _ms(started)
                answer = "".join(parts)
                truncated = ev.finish_reason == "length"

                await self._append_turn(session, user_turn, answer, turn_index, replace_turn)
                await self._save_record(request_id, inp, session, question, level, route, tier, cacheable,
                                        turn_index, is_retry=is_retry)
                if cacheable and answer and not truncated:
                    await self.answer_cache.put(
                        question, level,
                        CachedAnswer(text=answer, tier=tier, intent=route.intent, model=ev.model,
                                     created_at=time.time()),
                        embedding if self.settings.semantic_cache_enabled else None,
                    )
                yield _event("done", request_id=request_id, model=ev.model, provider=ev.provider,
                             usage=_usage_dict(usage), cost_usd=usage.cost_usd, ttft_ms=ttft,
                             latency_ms=row.latency_ms, truncated=truncated)
        except LLMError as e:
            log.error("Request %s lỗi: %s", request_id, e)
            row.status, row.error, row.latency_ms = "error", str(e)[:500], _ms(started)
            if refund_on_error:
                await self.quota.refund(inp.user_id)
            yield _event("error", request_id=request_id, code="llm_unavailable",
                         message="Hệ thống đang bận, bạn thử lại sau ít phút nhé.")
        except (asyncio.CancelledError, GeneratorExit):
            # User ngắt kết nối giữa chừng. Token đã sinh vẫn bị tính tiền nhưng không có usage.
            row.status, row.ttft_ms, row.latency_ms = "aborted", ttft, _ms(started)
            raise
        finally:
            # Ghi đồng bộ: khi request bị hủy, mọi `await` trong finally cũng bị hủy theo.
            self.usage_log.write_sync(row)

    async def _serve_cached(
        self, inp: ChatInput, session: Session, question: str, level: str | None, hit: CachedAnswer,
        source: str, row: RequestLog, started: float, route: Route | None = None,
    ) -> AsyncIterator[Event]:
        row.answer_cache, row.tier, row.model = source, hit.tier, hit.model
        row.intent = hit.intent
        row.route_reason = row.route_reason or f"answer_cache_{source}"
        row.cost_usd, row.latency_ms = 0.0, _ms(started)
        route = route or Route(hit.tier, True, f"answer_cache_{source}", intent=hit.intent)
        user_turn = self.prompt.compose_user_turn(question, level=level, intent=hit.intent,
                                                  summary=session.summary)
        turn_index = len(session.messages)
        await self._append_turn(session, user_turn, hit.text, turn_index, replace=False)
        await self._save_record(row.request_id, inp, session, question, level, route, hit.tier, True, turn_index)
        yield _event("meta", request_id=row.request_id, session_id=session.id, tier=hit.tier,
                     intent=hit.intent, reason=route.reason, answer_cache=source)
        yield _event("delta", text=hit.text)
        yield _event("done", request_id=row.request_id, model=hit.model, usage=None, cost_usd=0.0,
                     latency_ms=row.latency_ms)
        await self.usage_log.write(row)

    def _start_embedding(self, question: str) -> asyncio.Task | None:
        if not self.embedder:
            return None
        if not (self.settings.semantic_cache_enabled or self.router.uses_embeddings):
            return None
        return asyncio.create_task(self.embedder.embed(question))

    def _allowed_tier(self, tier: str, plan: str) -> str:
        # Quyền dùng model đắt kiểm tra bằng code, không để router quyết định.
        if tier == "large" and not self.quota.limits(plan).allow_large:
            return "small"
        return tier

    def _cacheable(self, route: Route, is_first: bool) -> bool:
        intent_ok = route.reason == route.intent and route.intent in ROUTES and ROUTES[route.intent][1]
        return intent_ok and (is_first or route.cacheable)

    async def _load_session(self, inp: ChatInput) -> Session:
        session = None
        if inp.session_id:
            session = await self.sessions.get(inp.user_id, inp.session_id)
        if session is None:
            return self.sessions.new(inp.user_id)
        if session.turns < self.settings.max_turns_per_session:
            return session
        summary = await self._summarize(session)
        return self.sessions.new(inp.user_id, summary=summary, previous_id=session.id)

    async def _summarize(self, session: Session) -> str | None:
        req = ChatRequest(
            messages=self.prompt.build(session.messages, SUMMARY_INSTRUCTION), max_tokens=400, temperature=0.2
        )
        try:
            result = await self.gateway.complete("small", req, sticky_key=self.sticky_key)
        except LLMError as e:
            log.warning("Không tóm tắt được phiên %s: %s", session.id, e)
            return None
        return result.text.strip() or None

    async def _append_turn(self, session: Session, user_turn: str, answer: str, turn_index: int | None,
                           replace: bool) -> None:
        if replace and turn_index is not None and turn_index + 1 < len(session.messages):
            session.messages[turn_index + 1] = {"role": "assistant", "content": answer}
        elif turn_index is not None:
            session.messages.append({"role": "user", "content": user_turn})
            session.messages.append({"role": "assistant", "content": answer})
            # Tóm tắt đã nằm trong lượt đầu tiên, không đưa vào các lượt sau nữa.
            session.summary = None
        else:
            return
        await self.sessions.save(session)

    async def _save_record(self, request_id: str, inp: ChatInput, session: Session, question: str,
                           level: str | None, route: Route, tier: str, cacheable: bool,
                           turn_index: int | None, *, is_retry: bool = False) -> None:
        record = RequestRecord(
            user_id=inp.user_id, plan=inp.plan, session_id=session.id, question=question, level=level,
            intent=route.intent, tier=tier, cacheable=cacheable, turn_index=turn_index, is_retry=is_retry,
        )
        await self.store.set(f"req:{request_id}", json.dumps(asdict(record), ensure_ascii=False), RECORD_TTL)

    async def _get_record(self, request_id: str, user_id: str) -> RequestRecord:
        raw = await self.store.get(f"req:{request_id}")
        record = RequestRecord(**json.loads(raw)) if raw else None
        if record is None or record.user_id != user_id:
            raise ServiceError("not_found", "Không tìm thấy request", 404)
        return record

    # ------------------------------------------------------------------ warmup

    async def warmup(self) -> list[dict[str, Any]]:
        """Gửi 1 request cho model chính của mỗi tầng để ghi cache phần đầu prompt trước khi mở traffic."""
        results = []
        for tier in ("small", "large"):
            req = ChatRequest(messages=self.prompt.build([], "ping"), max_tokens=1, temperature=0)
            target = self.gateway.targets(tier)[0]
            try:
                r = await self.gateway.complete(tier, req, sticky_key=self.sticky_key)
                results.append({"tier": tier, "model": r.model, "provider": r.provider,
                                "usage": _usage_dict(r.usage)})
            except LLMError as e:
                results.append({"tier": tier, "model": target.model, "error": str(e)})
        return results


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 1)
