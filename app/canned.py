"""Các câu trả lời mẫu, không tốn tiền gọi LLM."""

OFF_TOPIC_ANSWER = (
    "Mình là trợ lý học tiếng Trung nên chỉ hỗ trợ được các câu hỏi về tiếng Trung thôi: "
    "tra từ, pinyin, dịch câu, giải thích ngữ pháp, chữa bài viết, văn hóa và thành ngữ.\n\n"
    "Bạn thử hỏi mình kiểu như: \"了 và 过 khác nhau thế nào?\" hoặc "
    "\"Dịch giúp mình câu: Cuối tuần này bạn có rảnh không?\" nhé!"
)

# Gửi ở cuối lịch sử để dùng lại cache phần đầu prompt và cache lịch sử.
SUMMARY_INSTRUCTION = (
    "<yeu_cau_he_thong>\n"
    "Tóm tắt cuộc hội thoại ở trên trong tối đa 120 từ tiếng Việt để dùng làm ngữ cảnh cho phiên mới: "
    "trình độ người học (nếu biết), các từ, mẫu câu, lỗi sai đã được bàn tới, và câu hỏi đang dang dở. "
    "Chỉ trả về bản tóm tắt, không chào hỏi.\n"
    "</yeu_cau_he_thong>"
)
