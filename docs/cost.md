# Chi phí

> Mọi con số là **ước lượng để định hướng**. Cần đo lại bằng log token thật sau 1–2 tháng chạy.
> Giá Claude, Jev, Qwen và DeepSeek tra ngày 2026-09-23 (Qwen và DeepSeek lấy từ API `/models` của OpenRouter). Giá GPU là giá tham khảo.

## 1. Giả định

| Thông số | Mức Nhẹ | Mức Vừa |
|---|---|---|
| Tỉ lệ user hoạt động mỗi ngày (trong 10k) | 20% = 2.000 | 30% = 3.000 |
| Số câu hỏi mỗi người mỗi ngày | 5 | ~11 |
| **Số request mỗi tháng** | **~300k** | **~1M** |

Một request điển hình:

| Phần | Token | Cách tính giá |
|---|---|---|
| Kiến thức cốt lõi (CAG) | 20.000 | Giá đọc cache |
| Lịch sử hội thoại + câu hỏi | 2.000 | Giá input thường |
| Output, đã tính phần suy nghĩ | 800 | Giá output |

## 2. Công thức

```
chi phí / request = 20K × giá_đọc_cache + 2K × giá_input + 800 × giá_output
chi phí / tháng   = Σ theo tầng (số request của tầng × chi phí / request của tầng) + router
```

## 3. Bảng tính cho Qwen và DeepSeek (qua OpenRouter)

Giá mỗi 1M token. OpenRouter tính đúng giá của nhà cung cấp; phí nạp credit của OpenRouter tính riêng, xem openrouter.ai/pricing.

| Model | Tầng | Input | Đọc cache | Output | Chi phí / request | 300k | 1M |
|---|---|---|---|---|---|---|---|
| `qwen/qwen3.8-flash` (Alibaba) | small | $0.15 | $0.016 | $0.47 | ~$0.0010 | ~$300 | ~$1.000 |
| `qwen/qwen3.7-plus` (Alibaba) | large | $0.32 | $0.064 | $1.28 | ~$0.0029 | ~$880 | ~$2.940 |
| `deepseek/deepseek-v4-flash` | dự phòng small | $0.089 | $0.018 | $0.18 | ~$0.0007 | ~$200 | ~$670 |
| `deepseek/deepseek-v4-pro` | dự phòng large | $0.96 | $0.080 | $1.91 | ~$0.0050 | ~$1.510 | ~$5.030 |

Giá DeepSeek khác nhau theo nhà cung cấp trên OpenRouter (ví dụ `deepseek-v4-flash` có input từ $0.04 tới $0.21). Bảng dùng giá niêm yết của model.

Với cách chia ~70% `small`, ~30% `large`: 0.7 × $0.0010 + 0.3 × $0.0029 ≈ **$0.0016 mỗi request**, tức **~$475/tháng (300k)** và **~$1.580/tháng (1M)**, chưa trừ phần tiết kiệm từ cache câu trả lời. Ở mức Vừa, chi phí khoảng $0.16 mỗi user mỗi tháng, dưới mục tiêu $0.5.

Kiến thức cốt lõi hiện có khoảng 7.600 token (`cag knowledge info`), ít hơn mức 20K giả định ở trên, nên chi phí thật sẽ thấp hơn bảng. Con số cuối cùng lấy từ `cost_usd` trong log (`cag stats`).

**Embedding** (`baai/bge-m3`, $0.01/1M token): khoảng 30 token mỗi câu hỏi, dưới $1/tháng ở mức 1M request.

## 4. Tham chiếu: Claude API

Bảng này dùng làm mốc so sánh chất lượng và giá, và là phương án thay thế cho tầng `large` nếu eval cho thấy Qwen và DeepSeek chưa đạt. Đọc cache tính khoảng 10% giá input. Ghi cache tính 1.25 lần giá input với cache 5 phút.

| Model | Input / Output (mỗi 1M) | Chi phí / request | 300k | 1M |
|---|---|---|---|---|
| Claude Haiku 4.5 | $1 / $5 | ~$0.008 | ~$2.400 | ~$8.000 |
| Claude Sonnet 5 | $2 / $10 | ~$0.016 | ~$4.800 | ~$16.000 |
| Sonnet 5, **không cache** | | ~$0.052 | ~$15.600 | ~$52.000 |

Dòng cuối cho thấy caching (CAG) giúp giảm khoảng 3 lần chi phí. Nhà cung cấp nào cũng vậy, cache là bắt buộc.

## 5. Router (Jev)

~400 token mỗi request × $0.042/1M token → **~$5/tháng (300k) đến ~$17/tháng (1M)**. Không đáng kể. Jev gọi qua OpenRouter nên tính chung vào hóa đơn OpenRouter, không cần tài khoản TypeSafe riêng.

## 6. Các cách tối ưu

Mức tiết kiệm dưới đây tính trên chi phí trả lời trong kịch bản tham chiếu Sonnet 5, nơi output chiếm 50%, input mới 25%, đọc cache 25%.

| # | Cách làm | Tiết kiệm | Tài liệu |
|---|---|---|---|
| 1 | Chia model theo intent, ~70% sang `small` | ~35% | [routing.md](routing.md) |
| 2 | Giới hạn output theo intent, tắt thinking ở `small` | ~20% | [routing.md](routing.md) |
| 3 | Cache lịch sử hội thoại | ~15% | [caching.md](caching.md) |
| 4 | Cache câu trả lời (15–30% câu hỏi trùng) | 15–30% | [caching.md](caching.md) |
| 5 | Batch API cho việc không cần trả lời ngay (tạo sẵn nội dung, chấm bài qua đêm) | ~50% phần đó | |
| 6 | Rút gọn kiến thức cốt lõi 20K → 10K, phần còn lại chuyển sang RAG | ~10% | [architecture.md](architecture.md) |
| 7 | Câu `off_topic` trả câu mẫu | 100% phần đó | [routing.md](routing.md) |

Kết hợp cách 1–4 trên kịch bản Claude: $0.016 xuống khoảng **$0.0055–0.007 mỗi request (giảm 57–65%)**. Mức Vừa giảm từ $16.000 xuống khoảng $5.500–7.000 mỗi tháng.

## 7. Phương án tự host (giai đoạn sau)

### Ước lượng VRAM (Qwen3, fp16 KV cache)

| Model (AWQ 4-bit) | Trọng số | KV cache / token | KV cho 20K token kiến thức |
|---|---|---|---|
| Qwen3-8B | ~6 GB | ~144 KB | ~2.9 GB |
| Qwen3-14B | ~10 GB | ~160 KB | ~3.2 GB |
| Qwen3-32B | ~19 GB | ~256 KB | ~5.1 GB |

Dùng vLLM với `--enable-prefix-caching` thì phần KV của kiến thức được **dùng chung** giữa các request. Mỗi request chỉ tốn thêm KV cho lịch sử, câu hỏi và câu trả lời.

### Cấu hình và chi phí (Qwen3-14B)

| | Mức Nhẹ | Mức Vừa |
|---|---|---|
| Lượng output | 240M token/tháng, cao điểm ~300 token/giây | 800M token/tháng, cao điểm ~900 token/giây |
| GPU | 2× L4 24GB (1 chạy, 1 dự phòng) | 2× A100 80GB, hoặc 3–4× L4 |
| Hạ tầng / tháng | ~$700–1.300 | ~$2.000–4.000 |

**Chưa tính:** người vận hành (MLOps), GPU tính tiền 24/7 kể cả lúc không có traffic, và thời gian khởi động model khi autoscale.

**Chỉ chuyển sang tự host khi:** hóa đơn API ổn định trên khoảng $3–5k/tháng, **và** đã có người vận hành, **và** eval cho thấy model tự host đạt chất lượng. Qwen là lựa chọn phù hợp vì chuyển từ API sang tự host vẫn giữ nguyên họ model và prompt.
