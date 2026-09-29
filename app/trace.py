"""Ghi lại từng bước một request đã đi qua, để trang test hiển thị luồng chạy.

Mỗi bước: tên, trạng thái, mô tả tiếng Việt, thời điểm tính từ lúc nhận request, thời gian của bước.
Trace đi kèm event `done`/`error` và được lưu ở KV store dưới key `trace:{request_id}`.
"""

import time
from typing import Any, Literal

Status = Literal["ok", "hit", "miss", "skip", "fail", "fallback"]


class Trace:
    def __init__(self, started: float | None = None) -> None:
        self.started = started if started is not None else time.monotonic()
        self.steps: list[dict[str, Any]] = []

    def add(self, step: str, status: Status, detail: str, *, ms: float | None = None, at_ms: float | None = None,
            **data: Any) -> None:
        """`at_ms`: thời điểm bước kết thúc, mặc định là lúc gọi `add`."""
        self.steps.append({
            "step": step,
            "status": status,
            "detail": detail,
            "at_ms": at_ms if at_ms is not None else elapsed_ms(self.started),
            "ms": ms,
            **({"data": data} if data else {}),
        })


def elapsed_ms(since: float) -> float:
    return round((time.monotonic() - since) * 1000, 1)
