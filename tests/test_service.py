import json

import pytest

from app.canned import OFF_TOPIC_ANSWER
from app.router import Route
from app.service import ChatInput, ServiceError
from tests.conftest import collect


def rows(container):
    conn = container.usage_log._conn
    return [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY ts")]


async def test_first_question_calls_llm_and_logs_usage(service, backend, container):
    out = await collect(service.chat(ChatInput(user_id="u1", message="你好 là gì?", level="hsk1")))
    assert out["text"].strip() == "Câu trả lời mẫu."
    assert out["meta"]["tier"] == "small" and out["meta"]["answer_cache"] is None
    assert out["done"]["usage"]["cached_input_tokens"] == 7000
    target, req = backend.calls[0]
    assert target.model == "qwen/small"
    assert req.max_tokens == 300 and req.reasoning is False
    assert "Trình độ người học: HSK1" in req.messages[-1]["content"]
    [row] = rows(container)
    assert row["intent"] == "lookup" and row["cost_usd"] == pytest.approx(0.0005) and row["status"] == "ok"
    assert row["user_hash"] != "u1", "Không lưu user_id gốc"


async def test_exact_cache_hit_skips_router_and_llm(service, backend, router, container):
    await collect(service.chat(ChatInput(user_id="u1", message="你好 là gì?")))
    out = await collect(service.chat(ChatInput(user_id="u2", message="  你好 LÀ GÌ ")))
    assert out["meta"]["answer_cache"] == "exact"
    assert out["done"]["cost_usd"] == 0.0
    assert len(backend.calls) == 1 and router.calls == 1
    assert rows(container)[-1]["answer_cache"] == "exact"


async def test_semantic_cache_hit(service, backend, router):
    service.answer_cache.semantic_threshold = 0.85  # embedding giả thô hơn bge-m3
    await collect(service.chat(ChatInput(user_id="u1", message="打算 nghĩa là gì vậy bạn ơi")))
    out = await collect(service.chat(ChatInput(user_id="u2", message="打算 nghĩa là gì vậy bạn")))
    assert out["meta"]["answer_cache"] == "semantic"
    assert len(backend.calls) == 1 and router.calls == 2


async def test_follow_up_keeps_history_prefix_byte_identical(service, backend):
    first = await collect(service.chat(ChatInput(user_id="u1", message="把 dùng thế nào")))
    sid = first["meta"]["session_id"]
    await collect(service.chat(ChatInput(user_id="u1", message="cho thêm ví dụ", session_id=sid)))
    (_, r1), (_, r2) = backend.calls
    # Lượt 2 = toàn bộ lượt 1 (system + user) + câu trả lời + câu hỏi mới
    assert json.dumps(r2.messages[:2]) == json.dumps(r1.messages)
    assert r2.messages[2]["role"] == "assistant"
    assert len(r2.messages) == 4


async def test_follow_up_is_not_served_from_exact_cache(service, backend, router):
    service.settings.semantic_cache_enabled = False
    await collect(service.chat(ChatInput(user_id="u1", message="你好 là gì")))
    first = await collect(service.chat(ChatInput(user_id="u2", message="谢谢 là gì")))
    # Lượt sau phụ thuộc ngữ cảnh: không phải câu đầu phiên nên bỏ qua cache khớp tuyệt đối
    router.next = Route("small", False, "lookup", intent="lookup", confidence=0.9, needs_context=0.9)
    sid = first["meta"]["session_id"]
    out = await collect(service.chat(ChatInput(user_id="u2", message="你好 là gì", session_id=sid)))
    assert out["meta"]["answer_cache"] is None and len(backend.calls) == 3


async def test_session_of_other_user_is_not_reused(service, backend):
    first = await collect(service.chat(ChatInput(user_id="u1", message="把 dùng thế nào")))
    out = await collect(service.chat(ChatInput(user_id="u2", message="tiếp", session_id=first["meta"]["session_id"])))
    assert out["meta"]["session_id"] != first["meta"]["session_id"]
    assert len(backend.calls[-1][1].messages) == 2


async def test_off_topic_is_canned(service, backend, router, container):
    router.next = Route("canned", False, "off_topic", intent="off_topic", confidence=0.9)
    out = await collect(service.chat(ChatInput(user_id="u1", message="giá bitcoin?")))
    assert out["text"] == OFF_TOPIC_ANSWER and out["meta"]["tier"] == "canned"
    assert backend.calls == []
    assert rows(container)[-1]["status"] == "canned"


async def test_correction_is_not_cached(service, backend, router):
    router.next = Route("large", False, "correction", intent="correction", confidence=0.9)
    for uid in ("u1", "u2"):
        await collect(service.chat(ChatInput(user_id=uid, message="Sửa câu: 我是很忙")))
    assert len(backend.calls) == 2
    assert backend.calls[0][1].max_tokens == 1200


async def test_low_confidence_is_not_cached_and_uses_large(service, backend, router):
    router.next = Route("large", False, "low_confidence", intent=None, confidence=0.3)
    for uid in ("u1", "u2"):
        await collect(service.chat(ChatInput(user_id=uid, message="???")))
    assert len(backend.calls) == 2 and backend.calls[0][0].model == "qwen/large"


async def test_grammar_on_large_enables_reasoning(service, backend, router):
    router.next = Route("large", True, "grammar", intent="grammar", confidence=0.9)
    await collect(service.chat(ChatInput(user_id="u1", message="了 và 过")))
    assert backend.calls[0][1].reasoning is True


async def test_plan_without_large_is_downgraded(service, backend, router):
    router.next = Route("large", True, "grammar", intent="grammar", confidence=0.9)
    out = await collect(service.chat(ChatInput(user_id="u1", message="了 và 过", plan="basic")))
    assert out["meta"]["tier"] == "small" and backend.calls[0][0].model == "qwen/small"


async def test_truncated_answer_is_not_cached(service, backend):
    backend.finish_reason = "length"
    await collect(service.chat(ChatInput(user_id="u1", message="你好 là gì")))
    out = await collect(service.chat(ChatInput(user_id="u2", message="你好 là gì")))
    assert out["meta"]["answer_cache"] is None and len(backend.calls) == 2


async def test_fallback_model_is_logged(service, backend, container):
    backend.fail_models = {"qwen/small"}
    out = await collect(service.chat(ChatInput(user_id="u1", message="你好")))
    assert out["done"]["model"] == "deepseek/small"
    assert rows(container)[-1]["failed_models"] == "qwen/small"


async def test_llm_failure_refunds_quota(service, backend, container):
    backend.fail_models = {"qwen/small", "deepseek/small"}
    assert (await service.consume_quota("u1", "free")).used == 1
    out = await collect(service.chat(ChatInput(user_id="u1", message="你好")))
    assert out["error"]["code"] == "llm_unavailable"
    assert (await service.consume_quota("u1", "free")).used == 1, "Lượt lỗi đã được trả lại"
    assert rows(container)[-1]["status"] == "error"


async def test_session_rolls_over_with_summary(service, backend, settings):
    sid = None
    for i in range(settings.max_turns_per_session):
        out = await collect(service.chat(ChatInput(user_id="u1", message=f"câu {i} về 了", session_id=sid)))
        sid = out["meta"]["session_id"]
    backend.answer = "Tóm tắt: đang học 了."
    out = await collect(service.chat(ChatInput(user_id="u1", message="câu tiếp theo", session_id=sid)))
    assert out["meta"]["session_id"] != sid
    summary_req = backend.calls[-2][1]
    assert "Tóm tắt cuộc hội thoại" in summary_req.messages[-1]["content"]
    answer_req = backend.calls[-1][1]
    assert len(answer_req.messages) == 2, "Phiên mới chỉ có system + lượt đầu"
    assert "Tóm tắt: đang học 了." in answer_req.messages[-1]["content"]


async def test_feedback_invalidates_cache_and_retry_uses_large(service, backend):
    first = await collect(service.chat(ChatInput(user_id="u1", message="你好 là gì")))
    rid, sid = first["meta"]["request_id"], first["meta"]["session_id"]
    await service.feedback(rid, "u1", "unsatisfied")
    backend.answer = "Câu trả lời tốt hơn."
    out = await collect(service.retry(rid, "u1"))
    assert out["meta"]["tier"] == "large" and backend.calls[-1][0].model == "qwen/large"
    # Dùng lại đúng lượt user cũ, và thay câu trả lời cuối trong lịch sử
    assert backend.calls[-1][1].messages == backend.calls[0][1].messages
    session = await service.sessions.get("u1", sid)
    assert len(session.messages) == 2 and session.messages[1]["content"].strip() == "Câu trả lời tốt hơn."
    # Câu trả lời mới của tầng large được cache lại
    hit = await service.answer_cache.get_exact("你好 là gì", None)
    assert hit.tier == "large"


async def test_feedback_for_other_user_is_rejected(service):
    first = await collect(service.chat(ChatInput(user_id="u1", message="你好")))
    with pytest.raises(ServiceError) as ei:
        await service.feedback(first["meta"]["request_id"], "u2", "unsatisfied")
    assert ei.value.status == 404


async def test_retry_forbidden_for_plan_without_large(service):
    first = await collect(service.chat(ChatInput(user_id="u1", message="你好", plan="basic")))
    with pytest.raises(ServiceError) as ei:
        await collect(service.retry(first["meta"]["request_id"], "u1"))
    assert ei.value.code == "plan_forbidden"


async def test_warmup_hits_primary_model_of_each_tier(service, backend):
    results = await service.warmup()
    assert [r["model"] for r in results] == ["qwen/small", "qwen/large"]
    assert all(req.max_tokens == 1 for _, req in backend.calls)


async def test_retry_is_limited_to_once(service):
    first = await collect(service.chat(ChatInput(user_id="u1", message="你好")))
    rid = first["meta"]["request_id"]
    second = await collect(service.retry(rid, "u1"))
    for target in (rid, second["meta"]["request_id"]):
        with pytest.raises(ServiceError) as ei:
            await collect(service.retry(target, "u1"))
        assert ei.value.code == "already_retried"
