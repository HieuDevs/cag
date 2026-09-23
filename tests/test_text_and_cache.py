import numpy as np

from app.answer_cache import AnswerCache, CachedAnswer, han_signature
from app.store import MemoryStore
from app.text import normalize_question
from tests.conftest import fake_embedding


def test_normalize_question():
    assert normalize_question("  了  DÙNG khi nào？ ") == "了 dùng khi nào"
    assert normalize_question("你好，世界！") == "你好,世界"
    # NFD (dấu tổ hợp) và NFC cho cùng kết quả
    assert normalize_question("Tiếng") == normalize_question("Tiếng")


def test_han_signature():
    assert han_signature("了 dùng khi nào") == "了"
    assert han_signature("so sánh 了 và 过") == "了过"
    assert han_signature("không có chữ Hán") == ""


def make_cache(**kw) -> AnswerCache:
    return AnswerCache(MemoryStore(), version=kw.get("version", "v1"), ttl=60, semantic_threshold=0.9,
                       semantic_max_entries=kw.get("max_entries", 100))


def answer(text="A") -> CachedAnswer:
    return CachedAnswer(text=text, tier="small", intent="lookup", model="m", created_at=0)


async def test_exact_hit_ignores_case_space_and_punctuation():
    c = make_cache()
    await c.put("你好 là gì?", "HSK1", answer(), None)
    assert (await c.get_exact("  你好 LÀ GÌ ", "HSK1")).text == "A"
    assert await c.get_exact("你好 là gì", "HSK2") is None, "Khác trình độ thì khác key"


async def test_version_change_invalidates():
    store = MemoryStore()
    c1 = AnswerCache(store, version="v1", ttl=60, semantic_threshold=0.9, semantic_max_entries=10)
    c2 = AnswerCache(store, version="v2", ttl=60, semantic_threshold=0.9, semantic_max_entries=10)
    await c1.put("q", None, answer(), None)
    assert await c2.get_exact("q", None) is None
    assert c1.key("q", None).startswith("answer:v1:-:")


async def test_semantic_hit_requires_same_han_chars():
    c = make_cache()
    q1 = "了 dùng khi nào vậy bạn"
    await c.put(q1, None, answer("về 了"), np.array(fake_embedding(q1)))
    near = "了 dùng khi nào vậy"
    hit = await c.get_similar(near, None, np.array(fake_embedding(near)))
    assert hit and hit[0].text == "về 了"
    other = "过 dùng khi nào vậy bạn"  # embedding rất gần nhưng hỏi chữ khác
    assert await c.get_similar(other, None, np.array(fake_embedding(other))) is None


async def test_invalidate_removes_exact_and_semantic():
    c = make_cache()
    q = "打算 là gì"
    vec = np.array(fake_embedding(q))
    await c.put(q, None, answer(), vec)
    await c.invalidate(q, None)
    assert await c.get_exact(q, None) is None
    assert await c.get_similar(q, None, vec) is None


async def test_semantic_index_is_bounded():
    c = make_cache(max_entries=2)
    for i in range(3):
        q = f"câu hỏi số {i}"
        await c.put(q, None, answer(str(i)), np.array(fake_embedding(q)))
    assert len(c.index._keys["v1:-"]) == 2
