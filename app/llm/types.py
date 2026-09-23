from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Usage:
    """`usage` đã chuẩn hóa. `input_tokens` là phần input **không** đọc từ cache."""

    cached_input_tokens: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None

    @property
    def total_input_tokens(self) -> int:
        return self.cached_input_tokens + self.input_tokens

    @property
    def cache_read_ratio(self) -> float:
        total = self.total_input_tokens
        return self.cached_input_tokens / total if total else 0.0

    @classmethod
    def from_openrouter(cls, raw: dict[str, Any] | None) -> "Usage":
        if not raw:
            return cls()
        prompt_details = raw.get("prompt_tokens_details") or {}
        completion_details = raw.get("completion_tokens_details") or {}
        prompt = int(raw.get("prompt_tokens") or 0)
        cached = int(prompt_details.get("cached_tokens") or 0)
        cost = raw.get("cost")
        return cls(
            cached_input_tokens=cached,
            input_tokens=max(prompt - cached, 0),
            cache_write_tokens=int(prompt_details.get("cache_write_tokens") or 0),
            output_tokens=int(raw.get("completion_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            cost_usd=float(cost) if cost is not None else None,
        )


@dataclass
class ChatRequest:
    messages: list[dict[str, Any]]
    max_tokens: int
    reasoning: bool = False
    reasoning_max_tokens: int = 1024
    temperature: float | None = 0.3


@dataclass
class StreamEvent:
    type: Literal["delta", "done"]
    text: str = ""
    usage: Usage | None = None
    model: str = ""
    provider: str = ""
    finish_reason: str | None = None
    # Các model đã thử và lỗi trước khi tới model trả lời (chỉ có ở event "done" từ gateway).
    failed_attempts: list[str] = field(default_factory=list)


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    failed_attempts: list[str] = field(default_factory=list)


class LLMError(Exception):
    """Lỗi từ nhà cung cấp. `retryable` = có nên thử model tiếp theo trong tầng không."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
