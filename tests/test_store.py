from app.store import MemoryStore, make_store


async def test_memory_store_ttl(monkeypatch):
    s = MemoryStore()
    await s.set("k", "v", ttl=10)
    assert await s.get("k") == "v"
    import app.store as store_mod

    real = store_mod.time.monotonic
    monkeypatch.setattr(store_mod.time, "monotonic", lambda: real() + 11)
    assert await s.get("k") is None


async def test_incr():
    s = MemoryStore()
    assert await s.incr("c", ttl=5) == 1
    assert await s.incr("c", ttl=5) == 2


def test_make_store():
    assert isinstance(make_store("memory://"), MemoryStore)

