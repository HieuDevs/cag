"""Ghép prompt theo đúng thứ tự để giữ cache phần đầu (xem docs/caching.md).

    [system: vai trò + kiến thức cốt lõi]   CỐ ĐỊNH, build một lần lúc khởi động
    [lịch sử hội thoại của phiên]           chỉ nối thêm vào cuối
    [ngữ cảnh + đoạn RAG + câu hỏi mới]     thay đổi mỗi request

Không đưa vào phần system bất cứ thứ gì khác nhau giữa các request: ngày giờ, user, trình độ, RAG.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEVEL_RE = re.compile(r"^HSK(?:[1-6]|7-9)$")

INTENT_HINTS = {
    "lookup": "tra từ",
    "translate": "dịch câu",
    "grammar": "giải thích ngữ pháp",
    "culture": "văn hóa, thành ngữ, cổ văn",
    "correction": "chữa bài viết",
}


@dataclass(frozen=True)
class Knowledge:
    # 12 ký tự đầu của sha256 nội dung: sửa kiến thức là version tự đổi, cache câu trả lời cũ hết hiệu lực.
    version: str
    files: tuple[str, ...]
    text: str

    @property
    def estimated_tokens(self) -> int:
        # Ước lượng thô: ~1,2 chữ Hán mỗi token, ~3 ký tự Latin (tiếng Việt) mỗi token.
        han = sum(1 for ch in self.text if "\u4e00" <= ch <= "\u9fff")
        return int(han / 1.2 + (len(self.text) - han) / 3)


def load_knowledge(directory: Path) -> Knowledge:
    files = sorted(p for p in directory.glob("*.md") if p.is_file())
    if not files:
        raise RuntimeError(f"Không có file kiến thức nào trong {directory}")
    # Chuẩn hóa xuống dòng và bỏ khoảng trắng cuối file để kết quả không phụ thuộc editor.
    parts = [p.read_text(encoding="utf-8").replace("\r\n", "\n").strip() for p in files]
    text = "\n\n---\n\n".join(parts)
    return Knowledge(
        version=hashlib.sha256(text.encode()).hexdigest()[:12],
        files=tuple(p.name for p in files),
        text=text,
    )


def normalize_level(level: str | None) -> str | None:
    if not level:
        return None
    level = level.strip().upper().replace(" ", "")
    return level if LEVEL_RE.match(level) else None


class PromptBuilder:
    def __init__(self, knowledge: Knowledge, *, cache_control: bool = False):
        self.knowledge = knowledge
        if cache_control:
            content: Any = [{"type": "text", "text": knowledge.text, "cache_control": {"type": "ephemeral"}}]
        else:
            content = knowledge.text
        # Build một lần. Mọi request dùng lại đúng object này để phần đầu prompt giống nhau từng byte.
        self._system_message: dict[str, Any] = {"role": "system", "content": content}

    @property
    def version(self) -> str:
        return self.knowledge.version

    @property
    def system_message(self) -> dict[str, Any]:
        return self._system_message

    def compose_user_turn(
        self,
        question: str,
        *,
        level: str | None = None,
        intent: str | None = None,
        references: list[str] | None = None,
        summary: str | None = None,
    ) -> str:
        """Nội dung lượt user. Được lưu nguyên văn vào lịch sử để các lượt sau vẫn trúng cache lịch sử."""
        context: list[str] = []
        if level:
            context.append(f"Trình độ người học: {level}")
        if intent in INTENT_HINTS:
            context.append(f"Dạng câu hỏi: {INTENT_HINTS[intent]}")
        blocks: list[str] = []
        if summary:
            blocks.append(f"<tom_tat_phien_truoc>\n{summary.strip()}\n</tom_tat_phien_truoc>")
        if context:
            blocks.append("<ngu_canh>\n" + "\n".join(context) + "\n</ngu_canh>")
        if references:
            joined = "\n\n".join(r.strip() for r in references)
            blocks.append(f"<tai_lieu_tham_khao>\n{joined}\n</tai_lieu_tham_khao>")
        blocks.append(f"<cau_hoi>\n{question.strip()}\n</cau_hoi>")
        return "\n\n".join(blocks)

    def build(self, history: list[dict[str, str]], user_turn: str) -> list[dict[str, Any]]:
        return [self._system_message, *history, {"role": "user", "content": user_turn}]
