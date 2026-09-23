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

1. **Quota:** kiểm tra số lượt còn lại theo gói (Redis). Hết lượt thì trả thông báo nâng cấp. *(Chưa làm ở bản test.)*
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

| Thành phần | Nhiệm vụ | Hiện tại (MVP) | Dự kiến khi mở rộng |
|---|---|---|---|
| API | `/chat` (stream), `/feedback` | FastAPI | |
| Quota | Giới hạn lượt hỏi theo gói | Chưa làm (bản test) | Redis |
| Router | Phân loại intent, kiểm tra phụ thuộc ngữ cảnh, lọc câu ngoài phạm vi | Jev qua OpenRouter; dự phòng `bge-m3` + so khớp câu mẫu có nhãn | Logistic regression trên dữ liệu thật |
| Prompt builder | Giữ phần đầu prompt cố định từng byte, quản lý `KNOWLEDGE_VERSION` | Module nội bộ | |
| LLM gateway | Gọi model theo tầng, dự phòng, chuẩn hóa `usage` | OpenRouter | vLLM tự host (cùng API) |
| Cache câu trả lời | Khớp tuyệt đối và gần giống | Redis + chỉ mục embedding trong bộ nhớ | Redis + pgvector (khi chạy nhiều worker) |
| RAG | Tra từ vựng, giáo trình | Chưa có (interface `Retriever`) | pgvector hoặc Qdrant, embedding `bge-m3` |
| Batch job | Tạo sẵn giải thích từ vựng, bài tập | Chưa có | Batch API của nhà cung cấp |
| Observability | Chi phí mỗi request, tỉ lệ cache hit, phân bố intent | SQLite + `/stats`, `cag stats` | Postgres + Grafana/Metabase |

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

## 6. LLM gateway (OpenRouter)

Mọi model đều gọi qua **OpenRouter** (API tương thích OpenAI), gồm cả model trả lời, router Jev và embedding, nên chỉ cần một API key và một client. Mỗi tầng có một danh sách model theo thứ tự ưu tiên, model đứng sau là dự phòng (`app/config.py`, đổi được bằng biến môi trường `TIER_SMALL`, `TIER_LARGE`).

| Tầng | Model chính | Dự phòng |
|---|---|---|
| `small` | `qwen/qwen3.8-flash` (ghim nhà cung cấp `alibaba`) | `deepseek/deepseek-v4-flash` |
| `large` | `qwen/qwen3.7-plus` (ghim nhà cung cấp `alibaba`) | `deepseek/deepseek-v4-pro` |

- **Dự phòng:** gateway chỉ chuyển sang model tiếp theo khi **chưa** stream token nào cho user, và lỗi có thể thử lại (429, 5xx, timeout, 404 không có endpoint). Lỗi 401, 402 (hết credit), 403 (bị moderation chặn) thì dừng luôn. Đã stream một phần mà lỗi thì báo lỗi, không ghép hai câu trả lời.
- **Giữ cache nóng:** ghim nhà cung cấp bằng `provider.order`, và gửi `session_id = "cag-{KNOWLEDGE_VERSION}"` làm khóa sticky routing, để mọi request dùng chung phần đầu prompt đi tới cùng một nhà cung cấp.
- **Usage:** OpenRouter luôn trả `usage` ở chunk cuối của stream. Gateway chuẩn hóa về `cached_input_tokens`, `input_tokens` (phần không đọc cache), `cache_write_tokens`, `output_tokens`, `reasoning_tokens`, `cost_usd`. `cost_usd` là số tiền OpenRouter tính thật, không cần tự nhân giá.
- **Thinking:** tầng `small` gửi `reasoning: {enabled: false}`. Tầng `large` chỉ bật với intent trong `REASONING_INTENTS` (mặc định `grammar`), với `reasoning: {max_tokens: 1024, exclude: true}`. `max_tokens` được cộng thêm phần thinking để câu trả lời không bị cắt.
- **Dữ liệu:** không gửi thông tin định danh user vào prompt. OpenRouter có `provider.data_collection = "deny"` (biến `OPENROUTER_DATA_COLLECTION`) để chỉ dùng nhà cung cấp không lưu dữ liệu để train. Kiểm tra lại danh sách nhà cung cấp còn lại sau khi bật. Xem mục tuân thủ trong [roadmap.md](roadmap.md).
- **Tự host về sau:** vLLM có API tương thích OpenAI. Chỉ cần đổi `OPENROUTER_BASE_URL` sang vLLM (chạy với `--enable-prefix-caching`), code không đổi.
- **Jev** (`typesafe/jev-1.13`) cũng gọi qua OpenRouter, ở System One API `POST /api/v1/systemone`, cùng API key. Xem [routing.md](routing.md).
- **Embedding** (`baai/bge-m3`, $0.01/1M token) cũng gọi qua OpenRouter, dùng cho router dự phòng và cache câu trả lời gần giống.
- Khi trỏ `OPENROUTER_BASE_URL` sang vLLM tự host, Jev vẫn gọi OpenRouter qua `JEV_URL` riêng.

## 7. Cấu trúc thư mục

```
cag/
├── app/
│   ├── main.py            # FastAPI: /chat (SSE), /feedback, /health, /stats, /admin/warmup
│   ├── service.py         # luồng xử lý một request (mục 3)
│   ├── container.py       # khởi tạo các thành phần từ Settings
│   ├── config.py          # cấu hình, đọc từ biến môi trường / .env
│   ├── router.py          # Jev + bộ phân loại embedding dự phòng, bảng định tuyến
│   ├── prompt_builder.py  # phần đầu prompt cố định, KNOWLEDGE_VERSION, ghép lượt user
│   ├── llm/               # openrouter.py (client), gateway.py (tầng + dự phòng), types.py (usage)
│   ├── answer_cache.py    # khớp tuyệt đối + gần giống
│   ├── sessions.py        # lịch sử hội thoại, chỉ nối thêm, tóm tắt khi quá 10 lượt
│   ├── usage_log.py       # log mỗi request (SQLite, chuyển Postgres sau)
│   ├── store.py           # Redis / bộ nhớ trong tiến trình
│   ├── rag.py             # interface cho giai đoạn 6
│   ├── cli.py             # cag knowledge check|bump|info, cag warmup, cag stats
│   └── resources/router_examples.jsonl  # câu mẫu có nhãn cho router embedding
├── knowledge/             # kiến thức cốt lõi (.md) + VERSION + CHECKSUM
├── eval/                  # router_eval.py, answer_eval.py, datasets/
├── tests/
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
