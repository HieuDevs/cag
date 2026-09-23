import json
import re

from app.config import ROOT_DIR
from app.prompt_builder import (
    CHECKSUM_FILE,
    PromptBuilder,
    compute_checksum,
    load_knowledge,
    normalize_level,
)

KNOWLEDGE_DIR = ROOT_DIR / "knowledge"


def test_knowledge_checksum_matches_version():
    """CI: sửa knowledge/ thì phải chạy `cag knowledge bump` (tăng KNOWLEDGE_VERSION)."""
    recorded = (KNOWLEDGE_DIR / CHECKSUM_FILE).read_text().strip()
    assert recorded == compute_checksum(KNOWLEDGE_DIR), "Chạy `cag knowledge bump` sau khi sửa knowledge/"


def test_system_prefix_is_byte_stable():
    a = PromptBuilder(load_knowledge(KNOWLEDGE_DIR))
    b = PromptBuilder(load_knowledge(KNOWLEDGE_DIR))
    m1 = a.build([], a.compose_user_turn("你好 là gì", level="HSK1", intent="lookup"))
    m2 = b.build([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                 b.compose_user_turn("了 dùng khi nào", level="HSK5", intent="grammar", references=["RAG"]))
    assert json.dumps(m1[0], sort_keys=True) == json.dumps(m2[0], sort_keys=True)
    # Cùng một builder thì dùng lại đúng một object.
    assert a.build([], "q1")[0] is a.build([], "q2")[0]


def test_system_prefix_has_no_dynamic_data():
    text = load_knowledge(KNOWLEDGE_DIR).text
    assert not re.search(r"\b20\d\d-\d\d-\d\dT\d\d:", text), "Không được có timestamp trong phần được cache"
    assert "{{" not in text  # không còn placeholder template chưa điền


def test_dynamic_parts_go_after_cached_prefix():
    pb = PromptBuilder(load_knowledge(KNOWLEDGE_DIR))
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    turn = pb.compose_user_turn("Câu hỏi?", level="HSK3", intent="grammar", references=["Đoạn RAG"], summary="TT")
    msgs = pb.build(history, turn)
    assert msgs[0]["role"] == "system"
    assert "Trình độ người học" not in msgs[0]["content"] and "Đoạn RAG" not in msgs[0]["content"]
    assert msgs[1:3] == history
    last = msgs[-1]["content"]
    assert last.index("TT") < last.index("HSK3") < last.index("Đoạn RAG") < last.index("Câu hỏi?")


def test_compose_without_optional_parts():
    pb = PromptBuilder(load_knowledge(KNOWLEDGE_DIR))
    assert pb.compose_user_turn("  hi  ") == "<cau_hoi>\nhi\n</cau_hoi>"


def test_cache_control_option():
    pb = PromptBuilder(load_knowledge(KNOWLEDGE_DIR), cache_control=True)
    content = pb.system_message["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_normalize_level():
    assert normalize_level("hsk 3") == "HSK3"
    assert normalize_level("HSK7-9") == "HSK7-9"
    assert normalize_level("HSK10") is None
    assert normalize_level("") is None
    assert normalize_level("A1") is None
