"""Luồng xử lý một request (docs/architecture.md mục 3).

    cache khớp tuyệt đối → router → cache gần giống → RAG → ghép prompt
    → gọi LLM (stream) → log → lưu cache → feedback

`chat()` và `retry()` trả về luồng event dạng `{"event": ..., "data": {...}}`, API chuyển thành SSE:
- `meta`: request_id, session_id, tầng, intent, nguồn câu trả lời
- `delta`: một đoạn câu trả lời
- `done`: usage, chi phí, độ trễ, `trace` (từng bước đã chạy, xem `app/trace.py`)
- `error`: lỗi, kèm `code` và `trace`
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
from app.canned import SUMMARY_INSTRUCTION
from app.config import Settings
from app.embeddings import Embedder
from app.llm.gateway import LLMGateway
from app.llm.types import ChatRequest, LLMError, Usage
from app.prompt_builder import PromptBuilder, normalize_level
from app.rag import NullRetriever, Retriever
from app.router import ROUTES, IntentRouter, Route
from app.sessions import Session, SessionStore
from app.store import KVStore
from app.trace import Trace, elapsed_ms
from app.usage_log import RequestLog, UsageLog, hash_user

log = logging.getLogger(__name__)

Event = dict[str, Any]
RECORD_TTL = 24 * 3600


@dataclass
class ChatInput:
    user_id: str
    message: str
    session_id: str | None = None
    level: str | None = None


@dataclass
class RequestRecord:
    """Lưu lại để xử lý feedback "chưa hài lòng" cho request này."""

    user_id: str
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
        self.usage_log = usage_log
        self.store = store
        self.embedder = embedder
        self.retriever = retriever or NullRetriever()
        # Khóa sticky routing trên OpenRouter: mọi request dùng chung phần đầu prompt nên đi cùng một
        # nhà cung cấp, cache luôn nóng.
        self.sticky_key = f"cag-{prompt.version}"

    # ------------------------------------------------------------------ chat

    async def chat(self, inp: ChatInput) -> AsyncIterator[Event]:
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        level = normalize_level(inp.level)
        question = inp.message.strip()
        row = RequestLog(
            request_id=request_id,
            user_hash=hash_user(inp.user_id),
            knowledge_version=self.prompt.version,
        )

        trace = Trace(started)

        t = time.monotonic()
        session = await self._load_session(inp)
        row.session_id = session.id
        self._trace_session(trace, inp, session, elapsed_ms(t))
        # Có tóm tắt phiên trước thì câu hỏi có thể phụ thuộc ngữ cảnh, không coi là câu đầu phiên.
        is_first = session.is_first_turn and not session.summary

        # 1. Cache khớp tuyệt đối: chỉ cho câu hỏi đầu phiên, chạy trước router nên không tốn tiền Jev.
        if not is_first:
            trace.add("exact_cache", "skip", "Bỏ qua: không phải câu đầu phiên, câu hỏi có thể phụ thuộc ngữ cảnh")
        else:
            t = time.monotonic()
            hit = await self.answer_cache.get_exact(question, level)
            if hit:
                trace.add("exact_cache", "hit", "Trúng: câu hỏi giống hệt đã được trả lời, không cần router và LLM",
                          ms=elapsed_ms(t))
                async for ev in self._serve_cached(inp, session, question, level, hit, "exact", row, started,
                                                   trace):
                    yield ev
                return
            trace.add("exact_cache", "miss", f"Chưa có câu hỏi giống hệt (trình độ {level or 'không có'})",
                      ms=elapsed_ms(t))

        # 2. Router, chạy song song với embedding (dùng cho router dự phòng và cache gần giống).
        t_emb = time.monotonic()
        emb_ms: list[float] = []
        emb_task = self._start_embedding(question)
        if emb_task:
            emb_task.add_done_callback(lambda _: emb_ms.append(elapsed_ms(t_emb)))
        t = time.monotonic()
        route = await self.router.route(question, emb_task)
        self._trace_route(trace, route, elapsed_ms(t))
        row.intent, row.route_reason = route.intent, route.reason
        row.router_backend, row.router_confidence = route.backend, route.confidence
        row.needs_context = route.needs_context

        tier = route.tier
        cacheable = self._cacheable(route, is_first)
        trace.add("cacheable", "ok" if cacheable else "skip", self._cache_note(route, is_first, cacheable))
        embedding = await emb_task if emb_task else None
        if not emb_task:
            trace.add("embedding", "skip", "Không cần: tắt cache gần giống và router không dùng embedding")
        elif embedding is None:
            trace.add("embedding", "fail", "Lỗi khi tính embedding, bỏ qua cache gần giống",
                      ms=emb_ms[0] if emb_ms else None)
        else:
            trace.add("embedding", "ok", f"Tính embedding ({self.embedder.model}) song song với router",
                      ms=emb_ms[0] if emb_ms else None)

        # 3. Cache gần giống: sau router, chỉ khi được phép cache.
        threshold = self.settings.semantic_cache_threshold
        if not cacheable:
            trace.add("semantic_cache", "skip", "Bỏ qua: câu này không được cache")
        elif not self.settings.semantic_cache_enabled:
            trace.add("semantic_cache", "skip", "Tắt (SEMANTIC_CACHE_ENABLED=false)")
        elif embedding is None:
            trace.add("semantic_cache", "skip", "Bỏ qua: không có embedding")
        else:
            t = time.monotonic()
            if found := await self.answer_cache.get_similar(question, level, embedding):
                hit, score = found
                log.info("Trúng cache gần giống (%.3f): %r ~ %r", score, question, hit.question)
                trace.add("semantic_cache", "hit", f"Trúng câu gần giống (cosine {score:.3f} ≥ {threshold}): "
                          f"“{hit.question}”", ms=elapsed_ms(t), score=round(score, 4))
                async for ev in self._serve_cached(inp, session, question, level, hit, "semantic", row,
                                                   started, trace, route=route):
                    yield ev
                return
            trace.add("semantic_cache", "miss", f"Không có câu nào cùng chữ Hán với cosine ≥ {threshold}",
                      ms=elapsed_ms(t))

        # 4. RAG, 5. ghép prompt, 6. gọi LLM.
        t = time.monotonic()
        references = await self.retriever.retrieve(question, intent=route.intent, level=level)
        if isinstance(self.retriever, NullRetriever):
            trace.add("rag", "skip", "Chưa bật RAG, chỉ dùng kiến thức cố định trong system prompt")
        else:
            trace.add("rag", "ok" if references else "miss", f"Lấy được {len(references)} đoạn tham khảo",
                      ms=elapsed_ms(t))
        user_turn = self.prompt.compose_user_turn(
            question, level=level, intent=route.intent, references=references, summary=session.summary
        )
        history = list(session.messages)
        async for ev in self._generate(
            request_id=request_id, inp=inp, session=session, question=question, level=level, route=route,
            tier=tier, cacheable=cacheable, embedding=embedding, history=history, user_turn=user_turn,
            turn_index=len(history), row=row, started=started, trace=trace,
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
        """Hỏi lại bằng tầng `large` sau khi user bấm "chưa hài lòng"."""
        record = await self._get_record(request_id, user_id)
        # Mỗi request chỉ được hỏi lại một lần, và không hỏi lại bản hỏi lại.
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
            request_id=new_id, user_hash=hash_user(user_id), kind="retry",
            session_id=session.id, knowledge_version=self.prompt.version, intent=record.intent,
            route_reason="feedback_retry",
        )
        route = Route("large", record.cacheable, "feedback_retry", intent=record.intent)
        inp = ChatInput(user_id=user_id, message=record.question, session_id=session.id, level=record.level)
        trace = Trace(started)
        trace.add("feedback_retry", "ok", f"User chưa hài lòng với request {request_id[:8]}: hỏi lại bằng tầng "
                  "large, không qua router và cache", original_request_id=request_id)
        trace.add("session", "ok", "Dùng lại đúng lượt user cũ trong lịch sử, thay câu trả lời cũ bằng câu mới"
                  if idx is not None else "Không tìm thấy lượt cũ trong phiên: hỏi lại không kèm lịch sử")
        async for ev in self._generate(
            request_id=new_id, inp=inp, session=session, question=record.question, level=record.level,
            route=route, tier="large", cacheable=record.cacheable, embedding=None, history=history,
            user_turn=user_turn, turn_index=idx, row=row, started=started, trace=trace,
            replace_turn=idx is not None, is_retry=True,
        ):
            yield ev

    # ------------------------------------------------------------------ internals

    async def _generate(
        self, *, request_id: str, inp: ChatInput, session: Session, question: str, level: str | None,
        route: Route, tier: str, cacheable: bool, embedding: np.ndarray | None,
        history: list[dict[str, str]], user_turn: str, turn_index: int | None, row: RequestLog,
        started: float, trace: Trace, replace_turn: bool = False, is_retry: bool = False,
    ) -> AsyncIterator[Event]:
        row.tier = tier
        reasoning = tier == "large" and route.intent in self.settings.reasoning_intents
        req = ChatRequest(
            messages=self.prompt.build(history, user_turn),
            max_tokens=route.max_tokens,
            reasoning=reasoning,
            reasoning_max_tokens=self.settings.reasoning_max_tokens,
        )
        k = self.prompt.knowledge
        trace.add("prompt", "ok",
                  f"[system: kiến thức {k.version}, ~{k.estimated_tokens:,} token, cố định nên cache được] "
                  f"+ [lịch sử {len(history)} message] + [lượt user {len(user_turn)} ký tự]; "
                  f"max_tokens {req.max_tokens}" + (", bật thinking" if reasoning else ""),
                  messages=len(req.messages), max_tokens=req.max_tokens, reasoning=reasoning)
        yield _event("meta", request_id=request_id, session_id=session.id, tier=tier, intent=route.intent,
                     reason=route.reason, answer_cache=None)

        parts: list[str] = []
        ttft: float | None = None
        attempts: list[dict[str, Any]] = []
        llm_started = time.monotonic()
        try:
            async for ev in self.gateway.stream(tier, req, sticky_key=self.sticky_key, attempts=attempts):
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

                self._trace_llm_failures(trace, attempts, llm_started)
                cost = f"${usage.cost_usd:.5f}" if usage.cost_usd is not None else "chưa rõ chi phí"
                trace.add("llm", "fallback" if attempts else "ok",
                          f"{ev.model} qua {ev.provider or '?'}: TTFT {ttft or 0:.0f} ms, "
                          f"đọc cache {usage.cached_input_tokens:,}/{usage.total_input_tokens:,} token input "
                          f"({usage.cache_read_ratio:.0%}), {usage.output_tokens:,} token output, {cost}"
                          + (", bị cắt do max_tokens" if truncated else ""),
                          ms=elapsed_ms(llm_started), model=ev.model, provider=ev.provider,
                          finish_reason=ev.finish_reason)

                await self._append_turn(session, user_turn, answer, turn_index, replace_turn)
                await self._save_record(request_id, inp, session, question, level, route, tier, cacheable,
                                        turn_index, is_retry=is_retry)
                trace.add("save", "ok", f"Lưu lịch sử phiên ({session.turns} lượt) và bản ghi request để xử lý "
                          "feedback" if turn_index is not None else "Lưu bản ghi request, không lưu lịch sử phiên")
                if cacheable and answer and not truncated:
                    with_vec = embedding is not None and self.settings.semantic_cache_enabled
                    await self.answer_cache.put(
                        question, level,
                        CachedAnswer(text=answer, tier=tier, intent=route.intent, model=ev.model,
                                     created_at=time.time()),
                        embedding if self.settings.semantic_cache_enabled else None,
                    )
                    trace.add("answer_cache", "ok", "Lưu cache câu trả lời"
                              + (" và thêm embedding vào chỉ mục gần giống" if with_vec else ""))
                elif truncated:
                    trace.add("answer_cache", "skip", "Không lưu cache: câu trả lời bị cắt do max_tokens")
                else:
                    trace.add("answer_cache", "skip", "Không lưu cache: câu này không được cache")
                await self._save_trace(request_id, trace)
                yield _event("done", request_id=request_id, model=ev.model, provider=ev.provider,
                             usage=_usage_dict(usage), cost_usd=usage.cost_usd, ttft_ms=ttft,
                             latency_ms=row.latency_ms, truncated=truncated, trace=trace.steps)
        except LLMError as e:
            log.error("Request %s lỗi: %s", request_id, e)
            row.status, row.error, row.latency_ms = "error", str(e)[:500], _ms(started)
            self._trace_llm_failures(trace, attempts, llm_started)
            trace.add("llm", "fail", "Đã stream một phần thì lỗi, không chuyển model" if parts
                      else f"Mọi model của tầng {tier} đều lỗi, trả lỗi cho user")
            await self._save_trace(request_id, trace)
            yield _event("error", request_id=request_id, code="llm_unavailable",
                         message="Hệ thống đang bận, bạn thử lại sau ít phút nhé.", trace=trace.steps)
        except (asyncio.CancelledError, GeneratorExit):
            # User ngắt kết nối giữa chừng. Token đã sinh vẫn bị tính tiền nhưng không có usage.
            row.status, row.ttft_ms, row.latency_ms = "aborted", ttft, _ms(started)
            raise
        finally:
            # Ghi đồng bộ: khi request bị hủy, mọi `await` trong finally cũng bị hủy theo.
            self.usage_log.write_sync(row)

    async def _serve_cached(
        self, inp: ChatInput, session: Session, question: str, level: str | None, hit: CachedAnswer,
        source: str, row: RequestLog, started: float, trace: Trace, route: Route | None = None,
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
        age = time.time() - hit.created_at
        trace.add("answer", "hit", f"Trả câu trả lời đã cache (tầng {hit.tier}, model {hit.model}, tạo "
                  f"{age / 60:.0f} phút trước), không gọi LLM, $0")
        trace.add("save", "ok", f"Lưu lịch sử phiên ({session.turns} lượt) và bản ghi request")
        await self._save_trace(row.request_id, trace)
        yield _event("meta", request_id=row.request_id, session_id=session.id, tier=hit.tier,
                     intent=hit.intent, reason=route.reason, answer_cache=source)
        yield _event("delta", text=hit.text)
        yield _event("done", request_id=row.request_id, model=hit.model, usage=None, cost_usd=0.0,
                     latency_ms=row.latency_ms, trace=trace.steps)
        await self.usage_log.write(row)

    def _start_embedding(self, question: str) -> asyncio.Task | None:
        if not self.embedder:
            return None
        if not (self.settings.semantic_cache_enabled or self.router.uses_embeddings):
            return None
        return asyncio.create_task(self.embedder.embed(question))

    def _trace_session(self, trace: Trace, inp: ChatInput, session: Session, ms: float) -> None:
        if not inp.session_id:
            trace.add("session", "ok", "Không có session_id: mở phiên mới", ms=ms)
        elif session.id == inp.session_id:
            trace.add("session", "ok", f"Tiếp phiên cũ, đã có {session.turns} lượt", ms=ms)
        elif session.previous_id == inp.session_id:
            note = "đã tóm tắt phiên cũ bằng tầng small" if session.summary else "tóm tắt lỗi, bỏ qua tóm tắt"
            trace.add("session", "fallback",
                      f"Phiên cũ đủ {self.settings.max_turns_per_session} lượt: {note}, mở phiên mới", ms=ms)
        else:
            trace.add("session", "miss", "session_id không tồn tại hoặc đã hết hạn: mở phiên mới", ms=ms)

    @staticmethod
    def _trace_route(trace: Trace, route: Route, ms: float) -> None:
        if not route.attempts:
            trace.add("router", "skip", "Không cấu hình router nào", ms=ms)
        for i, a in enumerate(route.attempts):
            if a["ok"]:
                trace.add("router", "ok", f"{a['backend']}: intent {a['intent']}, độ tin cậy {a['confidence']:.2f} "
                          f"(ngưỡng {a['threshold']}), needs_context {a['needs_context']:.2f}", ms=a["ms"],
                          **{k: a[k] for k in ("backend", "intent", "confidence", "needs_context")})
            else:
                then = "chuyển router dự phòng" if i + 1 < len(route.attempts) else "hết router để thử"
                trace.add("router", "fail", f"{a['backend']} lỗi: {a['error'] or 'không rõ'} → {then}", ms=a["ms"],
                          backend=a["backend"])
        conf = f"{route.confidence:.2f}" if route.confidence is not None else "?"
        detail, status = {
            "off_topic": ("Câu hỏi ngoài tiếng Trung → tầng small, trả lời ngắn kèm vài từ tiếng Trung liên quan",
                          "ok"),
            "low_confidence": (f"Độ tin cậy {conf} dưới ngưỡng → tầng large, bỏ intent (an toàn hơn đẩy nhầm "
                               "câu khó xuống small)", "fallback"),
            "router_unavailable": ("Mọi router đều lỗi → tầng large", "fallback"),
            "unknown_intent": ("Router trả intent lạ → tầng large", "fallback"),
        }.get(route.reason, (f"intent {route.intent} → tầng {route.tier} theo bảng định tuyến", "ok"))
        trace.add("route", status, detail, tier=route.tier, reason=route.reason)

    def _cache_note(self, route: Route, is_first: bool, cacheable: bool) -> str:
        if route.reason != route.intent or route.intent not in ROUTES:
            return f"Không cache: route là {route.reason}, không có intent chắc chắn"
        if not ROUTES[route.intent][1]:
            why = {"correction": "câu trả lời riêng cho bài viết của từng user",
                   "off_topic": "câu ngoài chủ đề hay cần dữ liệu thời gian thực"}.get(route.intent, "")
            return f"Không cache: intent {route.intent}" + (f", {why}" if why else "")
        if is_first:
            return "Được cache: câu đầu phiên, không phụ thuộc ngữ cảnh"
        nc = f"{route.needs_context:.2f}" if route.needs_context is not None else "?"
        if cacheable:
            return f"Được cache: needs_context {nc} dưới ngưỡng {self.settings.needs_context_threshold}"
        return f"Không cache: câu hỏi phụ thuộc lượt trước (needs_context {nc})"

    @staticmethod
    def _trace_llm_failures(trace: Trace, attempts: list[dict[str, Any]], llm_started: float) -> None:
        start = elapsed_ms(trace.started) - elapsed_ms(llm_started)
        for a in attempts:
            start += a["ms"]
            then = "thử model dự phòng" if a["retryable"] and not a["after_first_token"] else "dừng"
            code = f" ({a['status']})" if a["status"] else ""
            trace.add("llm", "fail", f"{a['model']} lỗi{code}: {a['error']} → {then}", ms=a["ms"],
                      at_ms=round(start, 1), model=a["model"])

    async def _save_trace(self, request_id: str, trace: Trace) -> None:
        await self.store.set(f"trace:{request_id}", json.dumps(trace.steps, ensure_ascii=False), RECORD_TTL)

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
            user_id=inp.user_id, session_id=session.id, question=question, level=level,
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
