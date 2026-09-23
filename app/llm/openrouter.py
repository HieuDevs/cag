"""Client OpenRouter (API tương thích OpenAI) cho chat stream và embedding.

Tài liệu liên quan:
- Usage luôn có trong chunk cuối của stream, gồm `prompt_tokens_details.cached_tokens` và `cost`.
- `provider.order` ghim nhà cung cấp, `session_id` làm khóa sticky routing để cache luôn nóng.
- `reasoning.enabled=false` tắt thinking; `reasoning.max_tokens` giới hạn phần suy nghĩ.
- Jev (TypeSafe) gọi qua System One API `POST /api/v1/systemone`, cùng API key.
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.config import ModelTarget, Settings
from app.llm.types import ChatRequest, LLMError, StreamEvent, Usage

log = logging.getLogger(__name__)

# 401 sai key, 402 hết credit, 403 bị moderation chặn: đổi model cũng không giải quyết được.
NON_RETRYABLE_STATUS = {401, 402, 403}


class OpenRouterClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self.settings = settings
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": settings.openrouter_app_url,
            "X-Title": settings.openrouter_app_name,
        }
        self._http = http_client or httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=settings.llm_connect_timeout_seconds),
        )
        self._headers = headers

    async def aclose(self) -> None:
        await self._http.aclose()

    def build_body(self, target: ModelTarget, req: ChatRequest, *, sticky_key: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": target.model,
            "messages": req.messages,
            "stream": True,
            "max_tokens": req.max_tokens + (req.reasoning_max_tokens if req.reasoning else 0),
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.reasoning:
            body["reasoning"] = {"max_tokens": req.reasoning_max_tokens, "exclude": True}
        else:
            body["reasoning"] = {"enabled": False}
        provider: dict[str, Any] = {}
        if target.providers:
            provider["order"] = target.providers
            provider["allow_fallbacks"] = True
        provider["data_collection"] = self.settings.openrouter_data_collection
        body["provider"] = provider
        if sticky_key:
            body["session_id"] = sticky_key[:256]
        return body

    async def stream_chat(
        self, target: ModelTarget, req: ChatRequest, *, sticky_key: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        body = self.build_body(target, req, sticky_key=sticky_key)
        try:
            async with self._http.stream(
                "POST", "/chat/completions", json=body, headers=self._headers
            ) as resp:
                if resp.status_code >= 400:
                    raw = (await resp.aread()).decode("utf-8", "replace")
                    raise LLMError(
                        f"OpenRouter {resp.status_code}: {_error_message(raw)}",
                        status=resp.status_code,
                        retryable=resp.status_code not in NON_RETRYABLE_STATUS,
                    )
                usage: Usage | None = None
                model = target.model
                provider = ""
                finish_reason: str | None = None
                async for line in resp.aiter_lines():
                    # OpenRouter gửi dòng comment `: OPENROUTER PROCESSING` để giữ kết nối.
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        log.warning("Bỏ qua chunk không phải JSON: %.200s", data)
                        continue
                    if err := chunk.get("error"):
                        raise LLMError(f"OpenRouter stream error: {err.get('message', err)}", retryable=True)
                    model = chunk.get("model") or model
                    provider = chunk.get("provider") or provider
                    if chunk.get("usage"):
                        usage = Usage.from_openrouter(chunk["usage"])
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if text := delta.get("content"):
                            yield StreamEvent("delta", text=text)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                yield StreamEvent(
                    "done", usage=usage or Usage(), model=model, provider=provider, finish_reason=finish_reason
                )
        except httpx.TimeoutException as e:
            raise LLMError(f"OpenRouter timeout: {e!r}", retryable=True) from e
        except httpx.TransportError as e:
            raise LLMError(f"OpenRouter connection error: {e!r}", retryable=True) from e

    async def system_one(self, model: str, state: Any, questions: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        """Gọi Jev. Trả về body JSON: `{"model", "usage", "answers": {tên câu hỏi: câu trả lời}}`."""
        try:
            resp = await self._http.post(
                self.settings.jev_url,
                json={"model": model, "state": state, "questions": questions},
                headers=self._headers,
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"OpenRouter systemone error: {e!r}") from e
        if resp.status_code >= 400:
            raise LLMError(f"OpenRouter systemone {resp.status_code}: {_error_message(resp.text)}",
                           status=resp.status_code, retryable=resp.status_code not in NON_RETRYABLE_STATUS)
        return resp.json()

    async def embed(self, model: str, inputs: list[str], *, timeout: float) -> list[list[float]]:
        try:
            resp = await self._http.post(
                "/embeddings", json={"model": model, "input": inputs}, headers=self._headers, timeout=timeout
            )
        except httpx.HTTPError as e:
            raise LLMError(f"OpenRouter embedding error: {e!r}") from e
        if resp.status_code >= 400:
            raise LLMError(f"OpenRouter embedding {resp.status_code}: {_error_message(resp.text)}",
                           status=resp.status_code)
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


def _error_message(raw: str) -> str:
    try:
        err = json.loads(raw).get("error") or {}
        return str(err.get("message") or err)[:500]
    except (json.JSONDecodeError, AttributeError):
        return raw[:500]
