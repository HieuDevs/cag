"""Các chỉ dẫn cố định gửi kèm cho model."""

# Gửi ở cuối lịch sử để dùng lại cache phần đầu prompt và cache lịch sử.
SUMMARY_INSTRUCTION = (
    "<yeu_cau_he_thong>\n"
    "Tóm tắt cuộc hội thoại ở trên trong tối đa 120 từ tiếng Việt để dùng làm ngữ cảnh cho phiên mới: "
    "trình độ người học (nếu biết), các từ, mẫu câu, lỗi sai đã được bàn tới, và câu hỏi đang dang dở. "
    "Chỉ trả về bản tóm tắt, không chào hỏi.\n"
    "</yeu_cau_he_thong>"
)
