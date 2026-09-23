from app.config import PlanLimits
from app.quota import Quota
from app.store import MemoryStore


async def test_memory_store_ttl(monkeypatch):
    s = MemoryStore()
    await s.set("k", "v", ttl=10)
    assert await s.get("k") == "v"
    import app.store as store_mod

    real = store_mod.time.monotonic
    monkeypatch.setattr(store_mod.time, "monotonic", lambda: real() + 11)
    assert await s.get("k") is None


async def test_incr_decr():
    s = MemoryStore()
    assert await s.incr("c", ttl=5) == 1
    assert await s.incr("c", ttl=5) == 2
    assert await s.decr("c") == 1


async def test_quota_limits_and_refund():
    q = Quota(MemoryStore(), {"free": PlanLimits(daily_requests=2)}, timezone="Asia/Ho_Chi_Minh")
    assert (await q.consume("u", "free")).allowed
    assert (await q.consume("u", "free")).used == 2
    denied = await q.consume("u", "free")
    assert not denied.allowed and denied.limit == 2
    await q.refund("u")
    assert (await q.consume("u", "free")).allowed
    assert (await q.consume("other", "free")).used == 1


def test_unknown_plan_falls_back_to_free():
    q = Quota(MemoryStore(), {"free": PlanLimits(daily_requests=2, allow_large=False)}, timezone="UTC")
    assert q.limits("enterprise-hack").allow_large is False
