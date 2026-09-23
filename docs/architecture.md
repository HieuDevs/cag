# Kiến trúc

## 1. Bối cảnh và yêu cầu

- **Người dùng:** người Việt học tiếng Trung, từ HSK sơ cấp tới đại học. Hỏi bằng tiếng Việt, tiếng Trung, hoặc trộn cả hai.
- **Quy mô mục tiêu:** khoảng 10.000 user, tương đương 300k–1M request mỗi tháng (xem [cost.md](cost.md)).
- **Yêu cầu chất lượng:** đúng về học thuật (pinyin, thanh điệu, ngữ pháp, ví dụ), và giải thích bằng tiếng Việt tự nhiên.
- **Yêu cầu chi phí:** chi phí AI dưới khoảng $0.5 mỗi user mỗi tháng.

## 2. Vì sao CAG kết hợp RAG

CAG (Cache-Augmented Generation) nạp kiến thức vào context **một lần**, cache lại KV/prefix, rồi dùng lại cho mọi câu hỏi. Cách này không cần bước retrieve, nên nhanh hơn và không bị lỗi retrieve sai.

Giới hạn của CAG là kiến thức phải vừa context và càng dài càng tốn tiền đọc cache. Vì vậy hệ thống chia kiến thức như sau:

| Loại tài liệu | Kích thước ước lượng | Cách nạp |
|---|---|---|
| Vai trò, phong cách trả lời, đề cương, bảng ngữ pháp cốt lõi | 5–30K token | **CAG**, nằm trong phần đầu prompt được cache |
| Giáo trình chi tiết từng bài | 50–150K token | RAG |
| Từ vựng HSK 1–6 kèm pinyin, nghĩa Việt, ví dụ | 150K+ token | RAG |
| Từ điển, kho câu ví dụ | Hàng triệu token | RAG |

Ước lượng thô: khoảng 1–1.5 chữ Hán mỗi token. Tiếng Việt tốn token hơn.

## 3. Luồng xử lý một request

1. **Quota:** kiểm tra số lượt còn lại theo gói (Redis). Hết lượt thì trả thông báo nâng cấp.
2. **Cache câu trả lời, khớp tuyệt đối:** hash câu hỏi đã chuẩn hóa. Chỉ áp dụng cho câu hỏi đầu phiên.
3. **Router (Jev):** phân loại intent và kiểm tra câu hỏi có phụ thuộc ngữ cảnh không. Xem [routing.md](routing.md).
4. **Cache câu trả lời, gần giống:** chỉ chạy khi router đánh dấu `cacheable`.
5. **RAG (tùy chọn):** lấy đoạn tài liệu liên quan cho các intent cần tra cứu.
6. **Ghép prompt:** theo đúng thứ tự ở mục 5.
7. **Gọi LLM qua gateway:** stream câu trả lời, dự phòng sang nhà cung cấp khác khi lỗi.
8. **Ghi log:** model, intent, token đọc cache, token mới, token output, chi phí, độ trễ.
9. **Lưu cache câu trả lời:** nếu đủ điều kiện.
10. **Feedback:** user bấm "chưa hài lòng" thì hỏi lại bằng tầng `large`.

## 4. Các thành phần

| Thành phần | Nhiệm vụ | Công nghệ dự kiến |
|---|---|---|
| API | `/chat` (stream), `/feedback` | FastAPI |
| Quota | Giới hạn lượt hỏi theo gói | Redis |
| Router | Phân loại intent, kiểm tra phụ thuộc ngữ cảnh, lọc câu ngoài phạm vi | Jev; dự phòng `bge-m3` + logistic regression |
| Prompt builder | Giữ phần đầu prompt cố định từng byte, quản lý `KNOWLEDGE_VERSION` | Module nội bộ |
| LLM gateway | Gọi nhà cung cấp, retry, dự phòng, chuẩn hóa `usage` | Client tương thích OpenAI |
| Cache câu trả lời | Khớp tuyệt đối và gần giống | Redis + pgvector |
| RAG | Tra từ vựng, giáo trình | pgvector hoặc Qdrant, embedding `bge-m3` |
| Batch job | Tạo sẵn giải thích từ vựng, bài tập | Batch API của nhà cung cấp |
| Observability | Chi phí mỗi request, tỉ lệ cache hit, phân bố intent | Postgres + Grafana/Metabase |

## 5. Thứ tự các phần trong prompt

```
┌───────────────────────────────────────────────┐
│ system: vai trò + quy tắc trả lời            │  CỐ ĐỊNH, được cache
│ system: kiến thức cốt lõi (KNOWLEDGE_VERSION) │  (giống nhau cho mọi user)
├───────────────────────────────────────────────┤
│ lịch sử hội thoại của phiên                   │  cache theo phiên
├───────────────────────────────────────────────┤
│ [Trình độ: HSK3] + đoạn RAG + câu hỏi mới     │  thay đổi mỗi request
└───────────────────────────────────────────────┘
```

**Nguyên tắc:** mọi thứ khác nhau giữa các user hoặc các request (ngày giờ, tên, trình độ, đoạn RAG) phải nằm **sau** phần được cache. Xem [caching.md](caching.md).

## 6. LLM gateway

Gateway không phụ thuộc vào một nhà cung cấp cụ thể. Mỗi tầng có một danh sách nhà cung cấp theo thứ tự ưu tiên, nhà cung cấp đứng sau là dự phòng.

```python
class LLMProvider(Protocol):
    def chat(self, model: str, system: str, messages: list[dict], **opts) -> LLMResult: ...

PROVIDERS = {
    "qwen": OpenAICompatProvider(base_url=QWEN_BASE_URL, api_key=...),
    "deepseek": OpenAICompatProvider(base_url=DEEPSEEK_BASE_URL, api_key=...),
}

# tầng -> [(nhà cung cấp, model)], theo thứ tự ưu tiên
TIERS = {
    "small": [("qwen", QWEN_SMALL), ("deepseek", DEEPSEEK_CHAT)],
    "large": [("qwen", QWEN_LARGE), ("deepseek", DEEPSEEK_CHAT)],
}
```

- `LLMResult` chuẩn hóa `usage` về một dạng chung: `cached_input_tokens`, `input_tokens`, `output_tokens`. Tên trường của mỗi nhà cung cấp khác nhau.
- Tầng `small` **tắt chế độ thinking**. Tầng `large` chỉ bật thinking với intent cần suy luận (ví dụ `grammar`).
- **Dữ liệu:** dùng region quốc tế (Alibaba Cloud Singapore), hoặc nhà cung cấp ngoài Trung Quốc cho DeepSeek. Không gửi thông tin định danh user vào prompt. Xem mục tuân thủ trong [roadmap.md](roadmap.md).

## 7. Cấu trúc thư mục dự kiến

```
cag/
├── app/
│   ├── main.py            # FastAPI: /chat (stream), /feedback
│   ├── router.py          # Jev + embedding dự phòng
│   ├── prompt_builder.py  # SYSTEM_BLOCKS, KNOWLEDGE_VERSION
│   ├── llm/               # gateway, providers, chuẩn hóa usage
│   ├── answer_cache.py    # khớp tuyệt đối + gần giống
│   ├── rag.py
│   └── quota.py
├── knowledge/             # kiến thức cốt lõi (.md), quản lý bằng git
├── jobs/batch_generate.py # tạo sẵn nội dung
├── eval/                  # bộ câu hỏi có nhãn + script chấm
└── docs/
```

## 8. Các chỉ số cần theo dõi

| Chỉ số | Mục tiêu | Ghi chú |
|---|---|---|
| Chi phí trung bình mỗi request | Theo [cost.md](cost.md) | Tách theo intent và tầng |
| Tỉ lệ đọc cache (token đọc cache / tổng input) | > 80% | Tụt đột ngột nghĩa là phần đầu prompt đã bị thay đổi |
| Tỉ lệ hit cache câu trả lời | 15–30% | |
| Phân bố intent và tầng | ~70% `small` | |
| Tỉ lệ confidence thấp và tỉ lệ Jev lỗi | < 10% | Tăng lên thì cần tune lại criteria |
| Tỉ lệ bấm "chưa hài lòng" | Theo dõi xu hướng | Tách theo tầng |
| Độ trễ tới token đầu tiên (TTFT), p95 | < 2s | |
