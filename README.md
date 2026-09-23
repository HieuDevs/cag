# CAG — Trợ lý học thuật tiếng Trung

Trợ lý AI trả lời câu hỏi học thuật tiếng Trung cho người Việt: tra từ, pinyin, dịch, giải thích và so sánh ngữ pháp, chữa bài viết, văn hóa và thành ngữ.

Hệ thống kết hợp **Cache-Augmented Generation (CAG)** cho phần kiến thức cốt lõi với **định tuyến theo intent** để giữ chi phí thấp ở quy mô khoảng 10.000 user.

> **Trạng thái:** đang ở giai đoạn thiết kế, chưa có code. Các con số chi phí là ước lượng, cần đo lại bằng traffic thật.

## Kiến trúc tổng quan

```
Câu hỏi
  │
  ▼
Quota ──► Cache câu trả lời (khớp tuyệt đối) ──► Router: Jev (~200ms, ~$0)
                                                 │
        ┌───────────┬──────────────┬─────────────┼───────────────────┐
    off_topic     small          large       confidence thấp      Jev lỗi
    câu mẫu     Qwen nhỏ    Qwen lớn / DeepSeek     └────── gán large ──────┘
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
| Model trả lời | **Qwen** (chính), **DeepSeek** (dự phòng / A/B) | Mạnh tiếng Trung, giá rẻ, tự cache phần đầu prompt. Qwen có đường chuyển sang tự host |
| Router | **Jev** (TypeSafe), dự phòng bằng bộ phân loại embedding | Rẻ (~$17/tháng cho 1M request), nhanh, trả về confidence để quyết định có chuyển câu hỏi lên model mạnh hơn không |
| Kiến thức | CAG cho phần lõi (~20K token) + RAG cho phần tra cứu lớn | Tài liệu lớn như từ vựng HSK hay từ điển không vừa context |
| Hạ tầng | Dùng API trước, tự host sau khi traffic ổn định | Ra sản phẩm nhanh, chi phí tỉ lệ theo lượng dùng |

## Tài liệu

| File | Nội dung |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Luồng xử lý, các thành phần, gateway nhà cung cấp, cấu trúc thư mục |
| [docs/routing.md](docs/routing.md) | Định tuyến bằng Jev: câu hỏi, bảng định tuyến, dự phòng, rủi ro |
| [docs/caching.md](docs/caching.md) | 3 tầng cache, quy tắc giữ phần đầu prompt cố định, các lỗi hay gặp |
| [docs/cost.md](docs/cost.md) | Ước lượng chi phí dùng API và tự host, các cách tối ưu |
| [docs/roadmap.md](docs/roadmap.md) | Các giai đoạn triển khai, kế hoạch eval, câu hỏi mở, rủi ro |

## Tham khảo

- [Cache-Augmented Generation (CAG) from Scratch](https://medium.com/@sabaybiometzger/cache-augmented-generation-cag-from-scratch-441adf71c6a3): bài gốc, dùng Llama 3.1 8B 4-bit và `DynamicCache` của HF
- [TypeSafe: System One models và Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [Jev docs](https://docs.typesafe.ai/)
