"""Chuẩn hóa câu hỏi để làm key cho cache câu trả lời."""

import re
import unicodedata

# Dấu câu toàn góc của tiếng Trung -> bán góc.
_PUNCT = str.maketrans({
    "，": ",", "。": ".", "？": "?", "！": "!", "：": ":", "；": ";",
    "（": "(", "）": ")", "【": "[", "】": "]", "“": '"', "”": '"',
    "‘": "'", "’": "'", "、": ",", "～": "~", "　": " ",
})
_SPACE_RE = re.compile(r"\s+")
_TRAILING_RE = re.compile(r"[\s.?!,;:~]+$")


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_PUNCT).lower()
    text = _SPACE_RE.sub(" ", text).strip()
    # "了 dùng khi nào?" và "了 dùng khi nào" là cùng một câu hỏi.
    return _TRAILING_RE.sub("", text)
