"""Log mỗi request: model, intent, token đọc cache, token mới, output, chi phí, độ trễ.

Dùng SQLite cho MVP để chạy được ngay. Bảng giữ nguyên tên cột khi chuyển sang Postgres.
Không lưu user_id gốc hay nội dung câu hỏi, chỉ lưu hash của user_id.
"""

import asyncio
import hashlib
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    user_hash TEXT NOT NULL,
    session_id TEXT,
    kind TEXT NOT NULL,              -- chat | retry
    knowledge_version TEXT,
    intent TEXT,
    route_reason TEXT,
    router_backend TEXT,
    router_confidence REAL,
    needs_context REAL,
    tier TEXT,
    model TEXT,
    provider TEXT,
    failed_models TEXT,
    answer_cache TEXT,               -- exact | semantic | NULL
    cached_input_tokens INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cost_usd REAL,
    ttft_ms REAL,
    latency_ms REAL,
    status TEXT NOT NULL,            -- ok | error | canned
    error TEXT,
    feedback TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


def hash_user(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


@dataclass
class RequestLog:
    request_id: str
    user_hash: str
    kind: str = "chat"
    ts: float = field(default_factory=time.time)
    session_id: str | None = None
    knowledge_version: str | None = None
    intent: str | None = None
    route_reason: str | None = None
    router_backend: str | None = None
    router_confidence: float | None = None
    needs_context: float | None = None
    tier: str | None = None
    model: str | None = None
    provider: str | None = None
    failed_models: str | None = None
    answer_cache: str | None = None
    cached_input_tokens: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    ttft_ms: float | None = None
    latency_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    feedback: str | None = None


class UsageLog:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def write_sync(self, row: RequestLog) -> None:
        data = asdict(row)
        cols = ", ".join(data)
        marks = ", ".join(f":{k}" for k in data)
        with self._lock, self._conn:
            self._conn.execute(f"INSERT OR REPLACE INTO requests ({cols}) VALUES ({marks})", data)

    async def write(self, row: RequestLog) -> None:
        await asyncio.to_thread(self.write_sync, row)

    def _set_feedback(self, request_id: str, feedback: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE requests SET feedback = ? WHERE request_id = ?", (feedback, request_id))

    async def set_feedback(self, request_id: str, feedback: str) -> None:
        await asyncio.to_thread(self._set_feedback, request_id, feedback)

    def _stats(self, since: float) -> dict[str, Any]:
        with self._lock:
            total = self._conn.execute(
                """SELECT COUNT(*) AS requests,
                          SUM(cost_usd) AS cost_usd,
                          SUM(cached_input_tokens) AS cached_input_tokens,
                          SUM(input_tokens) AS input_tokens,
                          SUM(output_tokens) AS output_tokens,
                          SUM(answer_cache IS NOT NULL) AS answer_cache_hits,
                          SUM(status = 'error') AS errors,
                          SUM(route_reason IN ('low_confidence', 'router_unavailable', 'unknown_intent'))
                              AS router_fallbacks,
                          SUM(feedback = 'unsatisfied') AS unsatisfied
                   FROM requests WHERE ts >= ?""",
                (since,),
            ).fetchone()
            by = self._conn.execute(
                """SELECT COALESCE(intent, route_reason) AS intent, tier, COUNT(*) AS requests,
                          SUM(cost_usd) AS cost_usd, AVG(latency_ms) AS avg_latency_ms,
                          SUM(cached_input_tokens) AS cached_input_tokens, SUM(input_tokens) AS input_tokens
                   FROM requests WHERE ts >= ? GROUP BY 1, 2 ORDER BY 3 DESC""",
                (since,),
            ).fetchall()
            ttfts = [r[0] for r in self._conn.execute(
                "SELECT ttft_ms FROM requests WHERE ts >= ? AND ttft_ms IS NOT NULL AND answer_cache IS NULL "
                "ORDER BY ttft_ms", (since,)
            )]
        t = dict(total)
        n = t["requests"] or 0
        cached, fresh = t["cached_input_tokens"] or 0, t["input_tokens"] or 0
        return {
            **t,
            "cache_read_ratio": cached / (cached + fresh) if cached + fresh else None,
            "answer_cache_hit_rate": (t["answer_cache_hits"] or 0) / n if n else None,
            "avg_cost_usd": (t["cost_usd"] or 0) / n if n else None,
            "ttft_p95_ms": ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.95))] if ttfts else None,
            "by_intent_tier": [dict(r) for r in by],
        }

    async def stats(self, since: float) -> dict[str, Any]:
        return await asyncio.to_thread(self._stats, since)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
