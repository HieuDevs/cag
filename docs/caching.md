# Caching

Hệ thống có 3 tầng cache. Tầng 1 là CAG, và cũng là tầng tiết kiệm nhiều nhất.

| Tầng | Cache cái gì | Ở đâu | Tiết kiệm |
|---|---|---|---|
| 1. Cache phần đầu prompt (CAG) | System prompt + kiến thức cốt lõi | Phía nhà cung cấp LLM, hoặc prefix cache của vLLM nếu tự host | Phần input này chỉ tốn khoảng 10–20% giá thường |
| 2. Cache lịch sử hội thoại | Các lượt trước trong phiên | Phía nhà cung cấp LLM | Lịch sử không bị tính giá đầy đủ ở mỗi lượt |
| 3. Cache câu trả lời | Câu trả lời hoàn chỉnh | Redis + pgvector | 100% chi phí của request trúng cache |

## 1. Cache phần đầu prompt (CAG)

### CAG khi gọi qua API: KV cache nằm ở nhà cung cấp

Bài CAG gốc dùng `DynamicCache` của HuggingFace: tự chạy model, tính KV của phần kiến thức một lần, giữ tensor trong GPU và dùng lại cho mọi câu hỏi. Khi gọi model qua OpenRouter thì không lấy hay truyền tensor KV được. Việc đó do nhà cung cấp làm, dưới tên **prompt caching**: Alibaba (Qwen) và DeepSeek tự giữ KV của phần đầu prompt đã gặp, và tính phần đọc lại khoảng 10% giá input.

Vì vậy code **không tự cache KV**. Việc của code là làm cho KV cache phía nhà cung cấp trúng:

| Việc | Ở đâu |
|---|---|
| Phần system + kiến thức build một lần, giống nhau từng byte cho mọi request | `app/prompt_builder.py` |
| Ghim nhà cung cấp (`provider.order`) và gửi `session_id` cố định theo `KNOWLEDGE_VERSION`, để mọi request tới cùng nơi đang giữ cache | `app/llm/openrouter.py`, `app/service.py` |
| Ghi `cached_tokens` của từng request để đo tỉ lệ trúng cache thật | `app/usage_log.py`, `/stats` |
| Làm nóng cache sau deploy | `POST /admin/warmup`, `WARMUP_ON_STARTUP` |

Khi tự host bằng vLLM với `--enable-prefix-caching`, vLLM giữ KV của phần đầu dùng chung giữa các request (giống `DynamicCache` nhưng dùng được cho nhiều request song song). Code chỉ cần đổi `OPENROUTER_BASE_URL`.

`answer_cache.py` **không phải** KV cache. Nó là tầng 3: lưu câu trả lời hoàn chỉnh để câu hỏi trùng không phải gọi LLM.

### Quy tắc giữ cache

Cache của nhà cung cấp hoạt động theo kiểu **khớp phần đầu (prefix match)**. Chỉ cần một byte ở phần đầu prompt thay đổi là mọi thứ phía sau mất cache.

**Quy tắc bắt buộc:**

1. Phần đầu prompt (`SYSTEM_BLOCKS`) được build **một lần** lúc khởi động, từ file trong `knowledge/`. Không build lại theo từng request.
2. **Không** chèn vào phần đầu prompt: ngày giờ, tên hay ID user, trình độ, request ID, hay đoạn RAG.
3. Mọi dữ liệu có cấu trúc đưa vào prompt phải được serialize cố định: `json.dumps(..., sort_keys=True)`, và sắp xếp list theo thứ tự cố định.
4. `KNOWLEDGE_VERSION` tự tính từ nội dung `knowledge/*.md`. Sửa kiến thức thì deploy rồi làm nóng cache.
5. Mỗi model có cache riêng. Tầng `small` và `large` mỗi bên tự ghi cache của mình, đây là hành vi bình thường.

**Kiểm tra điều kiện của từng nhà cung cấp** (độ dài tối thiểu để được cache, thời gian sống của cache, giá khi trúng cache) trong tài liệu chính thức của Qwen và DeepSeek trước khi chốt. Các điều kiện này khác nhau giữa các nhà cung cấp và thay đổi theo thời gian.

## 2. Cache lịch sử hội thoại

- Giữ **toàn bộ** lịch sử của phiên và chỉ nối thêm vào cuối. Không cắt kiểu cửa sổ trượt `history[-6:]`, vì mỗi lần cửa sổ trượt thì phần đầu `messages` đổi và cache lịch sử mất.
- Giới hạn độ dài phiên khoảng 10 lượt. Quá mức đó thì tóm tắt phiên cũ và mở phiên mới, đưa bản tóm tắt vào lượt đầu.

## 3. Cache câu trả lời

**Chỉ cache khi đủ tất cả điều kiện:**
- Là câu hỏi đầu phiên, hoặc router trả `needs_context < 0.5`.
- Intent có `cacheable = true` theo [routing.md](routing.md). Không cache `correction`.
- Câu trả lời không bị user đánh giá "chưa hài lòng".

**Key:**

```
answer:{KNOWLEDGE_VERSION}:{level}:{sha256(normalize(question))}
```

`normalize` gồm: bỏ khoảng trắng thừa, chuyển về chữ thường, chuẩn hóa Unicode NFC, thống nhất dấu câu toàn góc/bán góc của tiếng Trung, và bỏ dấu câu ở cuối.

**Tra cứu theo 2 bước:**
1. **Khớp tuyệt đối** bằng hash trên Redis. Bước này chạy trước router nên không tốn cả tiền Jev.
2. **Khớp gần giống** bằng embedding `bge-m3` trên pgvector, cosine ≥ 0.95. Bước này chạy sau router, và chỉ khi `cacheable`.

Ngưỡng 0.95 phải kiểm tra bằng eval. Hai câu gần giống nhau về từ ngữ có thể hỏi hai thứ khác nhau, ví dụ "了 dùng khi nào" và "过 dùng khi nào". Vì vậy khớp gần giống còn bắt buộc **hai câu có đúng cùng các chữ Hán** (`han_signature` trong `app/answer_cache.py`).

Key có thêm **trình độ** (`answer:{KNOWLEDGE_VERSION}:{trình độ}:{hash}`) vì câu trả lời được viết theo trình độ người học. Key không có tầng, vì khớp tuyệt đối chạy trước router nên chưa biết tầng. Câu trả lời bị cắt do hết `max_tokens` thì không cache.

**Vô hiệu hóa cache:** khi đổi `KNOWLEDGE_VERSION`, toàn bộ key cũ tự hết hiệu lực. Đặt TTL cho key khoảng 30 ngày.

## 4. Làm nóng cache sau deploy

Cache của nhà cung cấp thường chỉ dùng được sau khi request đầu tiên đã được xử lý. Nếu ngay sau deploy có N request cùng lúc, cả N đều trả giá đầy đủ.

→ Sau mỗi lần deploy hoặc đổi `KNOWLEDGE_VERSION`, gửi 1 request làm nóng cho mỗi model đang dùng, trước khi mở traffic.

## 5. Các lỗi hay gặp làm mất cache mà không có cảnh báo

| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| `datetime.now()` hoặc tên user trong system prompt | Không request nào dùng được cache, chi phí tăng khoảng gấp 3 | Đưa vào message cuối |
| `json.dumps` không `sort_keys` | Thứ tự key thay đổi ngẫu nhiên nên mất cache | `sort_keys=True` |
| Đoạn RAG đặt trước kiến thức cốt lõi | Mất cache toàn bộ phần kiến thức | RAG đặt sau phần được cache |
| Cắt lịch sử bằng cửa sổ trượt | Mất cache lịch sử ở mỗi lượt | Xem mục 2 |
| Sửa file trong `knowledge/` mà version không đổi | Cache câu trả lời trả về nội dung cũ | Không xảy ra: version là hash của nội dung kiến thức |
| Cache câu trả lời cho `correction` | User A nhận câu trả lời dành cho user B | Chỉ cache intent có `cacheable = true` |

## 6. Giám sát

- Mỗi request ghi log số token đọc cache (từ `usage` đã chuẩn hóa ở gateway).
- Cảnh báo khi **tỉ lệ đọc cache < 80%** trong 15 phút. Gần như chắc chắn có một lỗi trong bảng trên vừa được deploy.
