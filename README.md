# CAG — Trợ lý học thuật tiếng Trung

Trợ lý AI trả lời câu hỏi học thuật tiếng Trung cho người Việt: tra từ, pinyin, dịch, giải thích và so sánh ngữ pháp, chữa bài viết, văn hóa và thành ngữ.

Hệ thống kết hợp **Cache-Augmented Generation (CAG)** cho phần kiến thức cốt lõi với **định tuyến theo intent** để giữ chi phí thấp ở quy mô khoảng 10.000 user.

Mọi model (Qwen, DeepSeek, router Jev, embedding) đều gọi qua **OpenRouter**, nên chỉ cần một API key.

> **Trạng thái:** đã có MVP chạy được (giai đoạn 1–5 trong [roadmap](docs/roadmap.md#tiến-độ-2026-09-23)). Chưa chạy với traffic thật, nên các con số chi phí vẫn là ước lượng.

## Kiến trúc tổng quan

```
Câu hỏi
  │
  ▼
Cache câu trả lời (khớp tuyệt đối) ──► Router: Jev (~200ms, ~$0)
                                                 │
        ┌───────────┬──────────────┬─────────────┼───────────────────┐
    off_topic     small          large       confidence thấp      Jev lỗi
    câu mẫu     Qwen nhỏ    Qwen lớn / DeepSeek    gán large     router embedding
                  │              │
                  ▼              ▼
          Prompt = [system + kiến thức cố định: CACHE] + [lịch sử] + [câu hỏi]
                  │
                  ▼
          Gọi LLM (stream) ─► log token / cache hit / chi phí ─► lưu cache câu trả lời
```

## Các quyết định chính

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| Model trả lời | **Qwen** (chính), **DeepSeek** (dự phòng / A/B), gọi qua **OpenRouter** | Mạnh tiếng Trung, giá rẻ, tự cache phần đầu prompt. Một API key cho mọi model. Qwen có đường chuyển sang tự host |
| Router | **Jev** (TypeSafe, gọi qua OpenRouter), dự phòng bằng bộ phân loại embedding | Rẻ (~$17/tháng cho 1M request), nhanh, trả về confidence để quyết định có chuyển câu hỏi lên model mạnh hơn không |
| Kiến thức | CAG cho phần lõi (~20K token) + RAG cho phần tra cứu lớn | Tài liệu lớn như từ vựng HSK hay từ điển không vừa context |
| Hạ tầng | Dùng API trước, tự host sau khi traffic ổn định | Ra sản phẩm nhanh, chi phí tỉ lệ theo lượng dùng |

## Chạy thử

Cần Python 3.11+ và một [OpenRouter API key](https://openrouter.ai/keys).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env            # thay CHANGE_ME bằng OPENROUTER_API_KEY
uvicorn app.main:app --reload
```

**Cấu hình:** `.env.example` liệt kê mọi biến, biến nào cũng có giá trị. Chỉ `OPENROUTER_API_KEY` là bắt buộc (dùng cho model trả lời, Jev và embedding); trống hoặc còn `CHANGE_ME` thì server dừng khi khởi động.

> **Bản test:** chưa có xác thực request và quota theo gói. Sẽ thêm sau.

Không có Redis thì đặt `REDIS_URL=memory://`: dữ liệu nằm trong bộ nhớ tiến trình, chỉ hợp cho dev. Chạy đủ bộ với Redis:

```bash
docker compose up --build
```

Router Jev (`typesafe/jev-1.13`) gọi qua OpenRouter bằng cùng key, không cần key TypeSafe riêng. Jev lỗi hoặc timeout thì router tự chuyển sang bộ phân loại embedding (`baai/bge-m3`, cũng qua OpenRouter).

### Gọi API

```bash
curl -N localhost:8000/chat -H 'content-type: application/json' \
  -d '{"user_id": "u1", "message": "了 và 过 khác nhau thế nào?", "level": "HSK3"}'
```

Kết quả là luồng SSE:

```
event: meta   data: {"request_id": "...", "session_id": "...", "tier": "large", "intent": "grammar", "answer_cache": null}
event: delta  data: {"text": "..."}
event: done   data: {"model": "qwen/qwen3.7-plus", "usage": {"cached_input_tokens": 7400, ...}, "cost_usd": 0.0021, "ttft_ms": 640}
```

| Endpoint | Mô tả |
|---|---|
| `POST /chat` | `user_id`, `message`, `session_id` (lấy từ event `meta` để hỏi tiếp), `level` (`HSK1`…`HSK6`, `HSK7-9`), `stream` (mặc định `true`, `false` thì trả JSON) |
| `POST /feedback` | `request_id`, `user_id`, `rating` (`satisfied`/`unsatisfied`). Thêm `retry: true` để hỏi lại bằng tầng `large` |
| `GET /health` | Version kiến thức, router đang dùng, model của từng tầng |
| `GET /stats?hours=24` | Chi phí, tỉ lệ đọc cache, tỉ lệ hit cache câu trả lời, TTFT p95, phân bố intent/tầng |
| `POST /admin/warmup` | Làm nóng cache phần đầu prompt cho model chính của mỗi tầng |

### Lệnh quản trị

| Lệnh | Mô tả |
|---|---|
| `cag knowledge bump` | Chạy sau khi sửa `knowledge/`: tăng `KNOWLEDGE_VERSION`, ghi checksum. CI chạy `cag knowledge check` |
| `cag knowledge info` | Số file và số token ước lượng của kiến thức cốt lõi |
| `cag warmup` | Làm nóng cache sau deploy |
| `cag stats --hours 24` | Chỉ số từ log |
| `python -m eval.router_eval` | Eval router: độ chính xác, tỉ lệ câu khó bị đẩy sang `small`, quét ngưỡng confidence |
| `python -m eval.answer_eval --judge anthropic/claude-sonnet-5` | Chạy bộ câu hỏi qua toàn bộ pipeline, chấm tự động và bằng model (gọi API thật, tốn tiền) |
| `pytest` | Test, không cần API key (OpenRouter được giả lập) |

Cấu hình đầy đủ ở [app/config.py](app/config.py) và [.env.example](.env.example).

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | Hệ thống chạy như thế nào: khởi động, từng bước của một request, prompt, OpenRouter, dữ liệu lưu ở đâu, xử lý lỗi |
| [docs/architecture.md](docs/architecture.md) | Luồng xử lý, các thành phần, gateway OpenRouter, cấu trúc thư mục |
| [docs/routing.md](docs/routing.md) | Định tuyến bằng Jev: câu hỏi, bảng định tuyến, dự phòng, rủi ro |
| [docs/caching.md](docs/caching.md) | 3 tầng cache, CAG khi gọi qua API (KV cache ở nhà cung cấp), các lỗi hay gặp |
| [docs/cost.md](docs/cost.md) | Ước lượng chi phí dùng API và tự host, các cách tối ưu |
| [docs/roadmap.md](docs/roadmap.md) | Các giai đoạn triển khai, kế hoạch eval, câu hỏi mở, rủi ro |

## Tham khảo

- [Cache-Augmented Generation (CAG) from Scratch](https://medium.com/@sabaybiometzger/cache-augmented-generation-cag-from-scratch-441adf71c6a3): bài gốc, dùng Llama 3.1 8B 4-bit và `DynamicCache` của HF
- [TypeSafe: System One models và Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [Jev docs](https://docs.typesafe.ai/)
