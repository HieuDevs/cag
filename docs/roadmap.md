# Roadmap

## Các giai đoạn

| Giai đoạn | Nội dung | Tiêu chí hoàn thành |
|---|---|---|
| **0. Dữ liệu và eval** | Gom khoảng 300 câu hỏi thật, gán nhãn intent. Soạn đáp án tham chiếu cho khoảng 100 câu. Viết script chấm | Có `eval/` chạy được bằng 1 lệnh, ra điểm theo từng intent |
| **1. MVP** | FastAPI `/chat` stream, 1 nhà cung cấp (Qwen), kiến thức cốt lõi trong `knowledge/`, cache phần đầu prompt, log `usage` | Tỉ lệ đọc cache > 80%, TTFT p95 < 2s |
| **2. Chọn model** | Chạy eval trên các bản Qwen và DeepSeek, lấy Claude làm mốc so sánh. Chấm theo: đúng học thuật, tiếng Việt tự nhiên, không lẫn ngôn ngữ | Chốt model cho tầng `small` và `large` |
| **3. Router** | Tích hợp Jev, bộ embedding dự phòng, eval router, chọn ngưỡng confidence | Câu khó bị đẩy nhầm sang `small` < 3% |
| **4. Cache câu trả lời + quota** | Redis, pgvector, giới hạn theo gói | Tỉ lệ hit đo được, không lẫn câu trả lời giữa các user |
| **5. Gateway dự phòng** | Thêm DeepSeek làm dự phòng, thêm feedback "chưa hài lòng" để hỏi lại bằng `large` | Tắt Qwen giả lập mà hệ thống vẫn trả lời được |
| **6. RAG + Batch** | RAG cho từ vựng HSK và giáo trình. Tạo sẵn nội dung bằng Batch API | |
| **7. (Tùy chọn) Tự host** | vLLM + Qwen trên GPU, xem [cost.md](cost.md#7-phương-án-tự-host-giai-đoạn-sau) | Đạt điều kiện chuyển trong cost.md |

## Tiêu chí chấm câu trả lời (eval)

| Tiêu chí | Cách chấm |
|---|---|
| Đúng học thuật: pinyin, thanh điệu, nghĩa, ngữ pháp | Người chấm, hoặc so với đáp án tham chiếu |
| Tiếng Việt tự nhiên, không lẫn chữ Trung hay tiếng Anh sai chỗ | LLM chấm, kèm người chấm mẫu |
| Ví dụ đúng, phù hợp trình độ | Người chấm |
| Độ dài phù hợp intent | Tự động |
| Chi phí và độ trễ | Tự động, từ log |

## Câu hỏi mở

1. **Kiến thức cốt lõi** gồm những tài liệu nào (giáo trình, đề cương), và có bản quyền để đưa vào hệ thống không?
2. **Tuân thủ dữ liệu cá nhân (Nghị định 13/2023):** có chuyển dữ liệu ra nước ngoài không, có cần hồ sơ đánh giá tác động không, và user có phải trẻ vị thành niên không? Việc này quyết định chọn region hoặc nhà cung cấp.
3. **Mô hình gói:** gói free được bao nhiêu câu mỗi ngày, và gói trả phí có dùng tầng `large` hoặc Claude cho câu khó không?
4. **Jev với tiếng Việt:** chờ kết quả eval ở giai đoạn 3.
5. **Chủ đề nhạy cảm** (lịch sử, chính trị Trung Quốc): Qwen và DeepSeek có thể né tránh hoặc trả lời một chiều. Có cần chuyển intent `culture` sang model khác không?

## Rủi ro

| Rủi ro | Mức độ | Cách giảm |
|---|---|---|
| Model trả lời sai kiến thức học thuật | Cao | Eval trước khi ra mắt, chọn `large` khi router không chắc, có nút feedback, CAG với tài liệu chuẩn |
| Chất lượng tiếng Việt của Qwen và DeepSeek | Trung bình | Có tiêu chí riêng trong eval. Có phương án thay thế cho tầng `large` |
| Nhà cung cấp quá tải hoặc đổi giá | Trung bình | Gateway có dự phòng, log chi phí theo ngày, cảnh báo ngân sách |
| Jev (startup mới) thay đổi rate limit hoặc giá | Trung bình | Timeout, dự phòng bằng embedding, ghim version |
| Chi phí vượt dự kiến | Trung bình | Quota theo gói, cảnh báo khi tỉ lệ đọc cache tụt, dashboard chi phí theo intent |
| Rủi ro pháp lý về dữ liệu | Cao nếu có trẻ vị thành niên | Làm rõ câu hỏi mở số 2 trước khi ra mắt |
