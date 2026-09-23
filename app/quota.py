"""Giới hạn lượt hỏi mỗi ngày theo gói. Ngày tính theo giờ Việt Nam."""

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import PlanLimits
from app.store import KVStore


@dataclass
class QuotaResult:
    allowed: bool
    used: int
    limit: int


class Quota:
    def __init__(self, store: KVStore, plans: dict[str, PlanLimits], *, timezone: str):
        self.store = store
        self.plans = plans
        self.tz = ZoneInfo(timezone)

    def limits(self, plan: str) -> PlanLimits:
        # Gói không xác định thì áp giới hạn của gói free, không bao giờ mở rộng quyền.
        return self.plans.get(plan) or self.plans["free"]

    def _key(self, user_id: str) -> str:
        return f"quota:{user_id}:{datetime.now(self.tz):%Y%m%d}"

    async def consume(self, user_id: str, plan: str) -> QuotaResult:
        limit = self.limits(plan).daily_requests
        used = await self.store.incr(self._key(user_id), ttl=2 * 24 * 3600)
        if used > limit:
            await self.store.decr(self._key(user_id))
            return QuotaResult(False, limit, limit)
        return QuotaResult(True, used, limit)

    async def refund(self, user_id: str) -> None:
        """Trả lại lượt khi request lỗi phía hệ thống."""
        await self.store.decr(self._key(user_id))
