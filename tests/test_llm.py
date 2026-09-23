import httpx
import pytest

from app.config import ModelTarget
from app.llm.gateway import LLMGateway
from app.llm.openrouter import OpenRouterClient
from app.llm.types import ChatRequest, LLMError, Usage
from tests.conftest import FakeOpenRouter, sse_chunks


def client_for(settings, handler) -> OpenRouterClient:
    http = httpx.AsyncClient(base_url=settings.openrouter_base_url, transport=httpx.MockTransport(handler))
    return OpenRouterClient(settings, http_client=http)


REQ = ChatRequest(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}], max_tokens=300)


def test_usage_normalization():
    u = Usage.from_openrouter({
        "prompt_tokens": 22000, "completion_tokens": 800, "cost": 0.0021,
        "prompt_tokens_details": {"cached_tokens": 20000, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 120},
    })
    assert (u.cached_input_tokens, u.input_tokens, u.output_tokens, u.reasoning_tokens) == (20000, 2000, 800, 120)
    assert u.cost_usd == pytest.approx(0.0021)
    assert u.cache_read_ratio == pytest.approx(20000 / 22000)
    assert Usage.from_openrouter(None) == Usage()


def test_build_body_small_tier_disables_reasoning(settings):
    c = client_for(settings, FakeOpenRouter())
    body = c.build_body(ModelTarget(model="qwen/x", providers=["alibaba"]), REQ, sticky_key="cag-v1")
    assert body["reasoning"] == {"enabled": False}
    assert body["max_tokens"] == 300
    assert body["provider"] == {"order": ["alibaba"], "allow_fallbacks": True, "data_collection": "allow"}
    assert body["session_id"] == "cag-v1"
    assert body["stream"] is True


def test_build_body_reasoning_budget(settings):
    settings.openrouter_data_collection = "deny"
    c = client_for(settings, FakeOpenRouter())
    req = ChatRequest(messages=REQ.messages, max_tokens=1000, reasoning=True, reasoning_max_tokens=1024)
    body = c.build_body(ModelTarget(model="qwen/x"), req, sticky_key=None)
    assert body["reasoning"] == {"max_tokens": 1024, "exclude": True}
    assert body["max_tokens"] == 2024, "max_tokens phải chừa chỗ cho phần thinking"
    assert body["provider"] == {"data_collection": "deny"}
    assert "session_id" not in body


async def test_stream_parses_deltas_usage_and_comments(settings):
    fake = FakeOpenRouter(answer=["你", "好"])
    c = client_for(settings, fake)
    events = [e async for e in c.stream_chat(ModelTarget(model="qwen/x"), REQ)]
    assert "".join(e.text for e in events if e.type == "delta") == "你好"
    done = events[-1]
    assert done.type == "done" and done.model == "qwen/x" and done.provider == "P"
    assert done.usage.cached_input_tokens == 7600 and done.usage.cost_usd == pytest.approx(0.0004)
    assert done.finish_reason == "stop"
    assert fake.requests[0]["messages"] == REQ.messages


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (404, True), (401, False), (402, False)])
async def test_http_errors(settings, status, retryable):
    c = client_for(settings, FakeOpenRouter(failures={"qwen/x": (status, "boom")}))
    with pytest.raises(LLMError) as ei:
        _ = [e async for e in c.stream_chat(ModelTarget(model="qwen/x"), REQ)]
    assert ei.value.status == status and ei.value.retryable is retryable
    assert "boom" in str(ei.value)


async def test_mid_stream_error_chunk(settings):
    body = b'data: {"choices":[{"delta":{"content":"A"}}]}\n\ndata: {"error":{"message":"provider died"}}\n\n'
    c = client_for(settings, lambda r: httpx.Response(200, content=body))
    got = []
    with pytest.raises(LLMError, match="provider died"):
        async for e in c.stream_chat(ModelTarget(model="qwen/x"), REQ):
            got.append(e)
    assert got[0].text == "A"


async def test_connection_error_is_retryable(settings):
    def handler(request):
        raise httpx.ConnectError("refused")

    c = client_for(settings, handler)
    with pytest.raises(LLMError) as ei:
        _ = [e async for e in c.stream_chat(ModelTarget(model="qwen/x"), REQ)]
    assert ei.value.retryable


async def test_embed(settings):
    c = client_for(settings, FakeOpenRouter())
    vecs = await c.embed("baai/bge-m3", ["a", "b"], timeout=1)
    assert len(vecs) == 2 and len(vecs[0]) == 256


# ---------------------------------------------------------------- gateway

async def test_gateway_falls_back_before_first_token(settings):
    fake = FakeOpenRouter(failures={"qwen/small": (503, "overloaded")})
    gw = LLMGateway(settings, client_for(settings, fake))
    result = await gw.complete("small", REQ)
    assert result.model == "deepseek/small"
    assert result.failed_attempts == ["qwen/small"]
    assert [r["model"] for r in fake.requests] == ["qwen/small", "deepseek/small"]


async def test_gateway_does_not_fall_back_on_auth_error(settings):
    fake = FakeOpenRouter(failures={"qwen/small": (401, "bad key")})
    gw = LLMGateway(settings, client_for(settings, fake))
    with pytest.raises(LLMError, match="bad key"):
        await gw.complete("small", REQ)
    assert len(fake.requests) == 1


async def test_gateway_does_not_fall_back_after_streaming_started(settings):
    body = b'data: {"choices":[{"delta":{"content":"A"}}]}\n\ndata: {"error":{"message":"died"}}\n\n'
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=body)

    gw = LLMGateway(settings, client_for(settings, handler))
    with pytest.raises(LLMError, match="died"):
        await gw.complete("large", REQ)
    assert len(calls) == 1


async def test_gateway_all_models_fail(settings):
    fake = FakeOpenRouter(failures={"qwen/large": (503, "x"), "deepseek/large": (500, "y")})
    gw = LLMGateway(settings, client_for(settings, fake))
    with pytest.raises(LLMError, match="Mọi model của tầng large"):
        await gw.complete("large", REQ)


async def test_gateway_success_passes_through_usage(settings):
    body = sse_chunks(["ok"], usage={"prompt_tokens": 10, "completion_tokens": 1,
                                     "prompt_tokens_details": {"cached_tokens": 8}})
    gw = LLMGateway(settings, client_for(settings, lambda r: httpx.Response(200, content=body)))
    r = await gw.complete("small", REQ)
    assert r.text == "ok" and r.usage.cached_input_tokens == 8 and r.usage.input_tokens == 2


def test_gateway_unknown_tier(settings):
    with pytest.raises(ValueError):
        LLMGateway(settings, None).targets("huge")
